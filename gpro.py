# fmt: off
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
# fmt: on

# ruff: noqa: E402
import argparse
import re
import time

from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
# from unsloth.tokenizer_utils import add_new_tokens

_CGREEN, _CRED, _CMAGENTA, _CYELLOW, _CCYAN, _CGRAY, _CEND = \
    "\033[92m", "\033[91m", "\033[95m", "\033[93m", "\033[96m", "\033[90m", "\033[0m"

_MAX_SEQ_LENGTH = 1024
_MAX_LORA_RANK = 64


def _train(_args: argparse.Namespace):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"unsloth/{_args.base_model}",
        max_seq_length=_MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=_MAX_LORA_RANK,
        gpu_memory_utilization=0.85,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=_MAX_LORA_RANK,
        # Remove QKVO if out of memory
        target_modules=[
            # "q_proj",
            # "k_proj",
            # "v_proj",
            # "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=_MAX_LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    _SYSTEM_PROMPT = "Respond in the following format: <think>{some thinking}</think> final answer here"

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen2.5",
        mapping={
            "role": "role",
            "content": "content",
            "user": "user",
            "assistant": "assistant",
        },
        map_eos_token=True,
        system_message=_SYSTEM_PROMPT,
    )
    tokenizer.pad_token = tokenizer.eos_token
    # add_new_tokens(model, tokenizer, ["<think>", "</think>"]) ## CUDA OOM

    def extract_xml_answer(text: str) -> str:
        answer = text.split("<think>")[-1]
        answer = answer.split("</think>")[-1]
        answer = answer.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        answer = answer.replace("final answer here", "")
        return answer.strip()

    # def extract_hash_answer(text: str) -> str | None:
    #     if "####" not in text:
    #         return None
    #     return text.split("####")[1].strip()

    def get_gsm8k_questions(split="train") -> Dataset:
        # data = load_dataset("openai/gsm8k", "main")[split]
        data = load_dataset("swulling/gsm8k_chinese")[split]  ## CN
        data = data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    # {"role": "user", "content": x["question"]},
                    {"role": "user", "content": x["question_zh-cn"]},  ## CN
                ],
                "answer": x["answer_only"],  ## CN
                # "answer": extract_hash_answer(x["answer"]),
            }
        )
        return data

    dataset = get_gsm8k_questions()

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        q = prompts[0][-1]["content"]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        print(
            f"\n{_CYELLOW}{'-' * 20}{_CEND}",
            f"\n{_CRED}Question:{_CEND}\n{q}",
            f"\n{_CGREEN}Answer:{_CEND}\n{answer[0]}",
            f"\n{_CMAGENTA}Response:{_CEND}\n{responses[0]}",
            f"\n{_CCYAN}Extracted:{_CEND}\n{extracted_responses[0]}",
            f"\n{_CGRAY}{'-' * 20}{_CEND}\n",
        )
        return [2.0 if str(a) in r else 0.0 for r, a in zip(extracted_responses, answer)]

    def int_reward_func(completions, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think>.*?$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"<think>.*?</think>.*?"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def count_xml(text) -> float:
        count = 0.0
        if text.count("final answer here") > 0:
            count -= 0.125
        if text.count("<think>") == 1:
            count += 0.125
        if text.count("</think>") == 1:
            count += 0.125
            count += len(text.split("</think>")[-1]) * 0.001
        if text.count("<answer>") > 0:
            count -= 0.125
            count -= len(text.split("<answer>")[-1]) * 0.001
        if text.count("</answer>") > 0:
            count -= 0.125
            count -= (len(text.split("</answer>")[-1])) * 0.001
        return count

    def xmlcount_reward_func(completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [count_xml(c) for c in contents]

    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,  # Decrease if out of memory
        max_prompt_length=256,
        max_completion_length=256,
        # num_train_epochs = 1,
        max_steps=900,
        save_steps=300,
        max_grad_norm=0.1,
        report_to="none",
        output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    if _args.save_lora:
        model.save_pretrained(f"{_args.saved_path}/{_args.saved_name}-LoRA")
        tokenizer.save_pretrained(f"{_args.saved_path}/{_args.saved_name}-LoRA")

    if _args.save_vllm:
        model.save_pretrained_merged(
            f"{_args.saved_path}/{_args.saved_name}-ft",
            tokenizer,
            # save_method="merged_4bit_forced",
        )

    if _args.save_gguf:
        model.save_pretrained_gguf(
            f"{_args.saved_path}/{_args.saved_name}-q5_K_M",
            tokenizer,
            quantization_method="q5_k_m",
        )
        print(f"\n\n{tokenizer._ollama_modelfile}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen2.5-14B-Instruct")
    parser.add_argument("--sys-prompt", type=str, default="system_message.md")
    parser.add_argument("--save-lora", action="store_true", default=True)
    parser.add_argument("--save-vllm", action="store_true", default=True)
    parser.add_argument("--save-gguf", action="store_true", default=False)
    parser.add_argument("--saved-path", type=str, default="./models")
    parser.add_argument("--saved-name", type=str, default="Qwen2.5-14B-GPRO")
    _args = parser.parse_args()

    start_time = time.time()
    _train(_args)
    end_time = time.time()
    print(f"Training took: {end_time - start_time} seconds.")
