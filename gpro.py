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

_MAX_SEQ_LENGTH = 1024
_LORA_RANK = 64


def _train(_args: argparse.Namespace):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"unsloth/{_args.base_model}",
        max_seq_length=_MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=_LORA_RANK,
        gpu_memory_utilization=0.5,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=_LORA_RANK,
        # Remove QKVO if out of memory
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=_LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    SYSTEM_PROMPT = """
    Respond in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """

    # ruff: noqa: F841
    XML_COT_FORMAT = """\
    <reasoning>
    {reasoning}
    </reasoning>
    <answer>
    {answer}
    </answer>
    """

    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    def get_gsm8k_questions(split="train") -> Dataset:
        data = load_dataset("openai/gsm8k", "main")[split]
        data = data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }
        )
        return data

    dataset = get_gsm8k_questions()

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        q = prompts[0][-1]["content"]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        print(
            "-" * 20,
            f"Question:\n{q}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    def int_reward_func(completions, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def count_xml(text) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1]) * 0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
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
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=200,
        # num_train_epochs = 1,
        max_steps=250,
        save_steps=250,
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
    parser.add_argument("--base-model", type=str, default="Qwen2.5-3B-Instruct")
    parser.add_argument("--sys-prompt", type=str, default="system_message.md")
    parser.add_argument("--save-lora", action="store_true", default=True)
    parser.add_argument("--save-vllm", action="store_true", default=True)
    parser.add_argument("--save-gguf", action="store_true", default=False)
    parser.add_argument("--saved-path", type=str, default="./models")
    parser.add_argument("--saved-name", type=str, default="Qwen2.5-3B-GPRO")
    _args = parser.parse_args()

    start_time = time.time()
    _train(_args)
    end_time = time.time()
    print(f"Training took: {end_time - start_time} seconds.")
