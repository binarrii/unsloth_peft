import argparse
import time
# import textwrap

from pprint import pprint

import pandas as pd
import torch

from datasets import Dataset, load_dataset

# fmt: off
# Unsloth should be imported before trl
from unsloth import (
    FastLanguageModel,
    # UnslothTrainer,
    # UnslothTrainingArguments,
    # is_bfloat16_supported,
)
# from unsloth.chat_templates import (
#     get_chat_template,
#     standardize_sharegpt,
# )
from trl import SFTConfig, SFTTrainer
# fmt: on

_MAX_SEQ_LEN = 1024
_MAX_LORA_RANK = 256


def _train(_args: argparse.Namespace):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=_args.base_model,
        max_seq_length=_MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )
    # print(f">>>>>> {tokenizer.chat_template}")

    with open("pretrained_model.txt", "w") as modelf:
        pprint(vars(model), modelf)

    model = FastLanguageModel.get_peft_model(
        model,
        r=_MAX_LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=_MAX_LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    model.config.repetition_penalty = 1.15

    with open("peft_model.txt", "w") as modelf:
        pprint(vars(model), modelf)

    # system_prompt = _get_system_prompt(_args.sys_prompt)
    # tokenizer._system_message = system_prompt
    # tokenizer = get_chat_template(
    #     tokenizer,
    #     chat_template="qwen2.5",
    #     mapping={
    #         "role": "role",
    #         "content": "content",
    #         "user": "user",
    #         "assistant": "assistant",
    #     },
    #     map_eos_token=True,
    #     system_message=system_prompt,
    # )
    # tokenizer.pad_token = tokenizer.eos_token

    # def formatting_prompts_func(inputs):
    #     samples = inputs["text"]
    #     print(f">>>>> len(samples): {len(samples)}")
    #     texts = []
    #     for text in samples:
    #         texts.append(f"{text}{tokenizer.eos_token}")
    #     return {"text": texts}

    # # `sample_by` line, paragraph, document
    # dataset = load_dataset(
    #     "text",
    #     data_files=["files/wenlv/md/*.md"],
    #     sample_by="document",
    #     split="train",
    # )
    # # print(f">>>>>>>> {dataset.column_names}")
    # # dataset = dataset["train"].train_test_split(train_size=0.9)["train"]
    # dataset = dataset.map(formatting_prompts_func, batched=True)

    # Continued Pretraining
    # trainer = UnslothTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=dataset,
    #     dataset_text_field="text",
    #     max_seq_length=_MAX_SEQ_LEN,
    #     dataset_num_proc=1,
    #     packing=False,  # Can make training 5x faster for short sequences.
    #     args=UnslothTrainingArguments(
    #         per_device_train_batch_size=2,
    #         gradient_accumulation_steps=8,
    #         warmup_steps=len(dataset) // 10,
    #         # num_train_epochs=35,
    #         max_steps=len(dataset),
    #         learning_rate=4e-5,
    #         embedding_learning_rate=4e-6,
    #         fp16=not is_bfloat16_supported(),
    #         bf16=is_bfloat16_supported(),
    #         logging_steps=1,
    #         optim="adamw_8bit",
    #         weight_decay=0.01,
    #         lr_scheduler_type="linear",
    #         seed=3407,
    #         output_dir="outputs",
    #         report_to="none",  # Use this for WandB etc
    #     ),
    # )

    def generate_conversation(examples):
        problems = examples["question"]
        solutions = examples["answer"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append(
                [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ]
            )
        return {"conversations": conversations}

    provider_dataset = load_dataset("json", data_files="WASU_QA.jsonl", split="train")
    provider_dataset = provider_dataset.map(generate_conversation, batched=True)
    provider_series = pd.Series(
        tokenizer.apply_chat_template(provider_dataset["conversations"], tokenize=False)
    )

    identity_dataset = load_dataset("json", data_files="identity.jsonl", split="train")
    identity_dataset = identity_dataset.map(generate_conversation, batched=True)
    identity_series = pd.Series(
        tokenizer.apply_chat_template(identity_dataset["conversations"], tokenize=False)
    )

    combined_data = pd.concat([provider_series, identity_series])
    combined_data.name = "text"

    combined_dataset = Dataset.from_pandas(pd.DataFrame(combined_data))
    combined_dataset = combined_dataset.shuffle(seed=3407)

    # print(f">>>>> {combined_dataset[0]}")
    # print(f">>>>> {identity_dataset.column_names}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=combined_dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=len(combined_dataset) // 8,
            max_steps=len(combined_dataset) * 2,
            learning_rate=4e-6,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )

    start_memory, max_memory = _show_start_memory_stats()
    trainer_stats = trainer.train()
    _show_final_memory_stats(trainer_stats, start_memory, max_memory)

    # saving Lora
    if _args.save_lora:
        model.save_pretrained(f"{_args.saved_path}/{_args.saved_name}-wenlv-LoRA")
        tokenizer.save_pretrained(f"{_args.saved_path}/{_args.saved_name}-wenlv-LoRA")
        # model.push_to_hub("your_name/Qwen2.5-7B-bnb-4bit-ft", token = "...") # Online saving
        # tokenizer.push_to_hub("your_name/Qwen2.5-7B-bnb-4bit-ft", token = "...") # Online saving

    # Saving to float16 for vLLM
    if _args.save_vllm:
        model.save_pretrained_merged(
            f"{_args.saved_path}/{_args.saved_name}-wenlv-ft",
            tokenizer,
            # save_method="merged_4bit_forced",
        )
        # model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

    # GGUF / llama.cpp Conversion
    if _args.save_gguf:
        model.save_pretrained_gguf(
            f"{_args.saved_path}/{_args.saved_name}-wenlv-q5_K_M",
            tokenizer,
            quantization_method="q5_k_m",
        )
        print(f"\n\n{tokenizer._ollama_modelfile}\n\n")
        # model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q5_k_m", token = "")


def _get_system_prompt(file):
    with open(file, "r") as f:
        return f.read()


def _show_start_memory_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return start_gpu_memory, max_memory


def _show_final_memory_stats(trainer_stats, start_gpu_memory, max_memory):
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="unsloth/Qwen3-14B")
    parser.add_argument("--sys-prompt", type=str, default="system_message.md")
    parser.add_argument("--save-lora", action="store_true", default=False)
    parser.add_argument("--save-vllm", action="store_true", default=True)
    parser.add_argument("--save-gguf", action="store_true", default=False)
    parser.add_argument("--saved-path", type=str, default="./models")
    parser.add_argument("--saved-name", type=str, default="Qwen3-14B")
    _args = parser.parse_args()

    start_time = time.time()
    _train(_args)
    end_time = time.time()
    print(f"Training took: {end_time - start_time} seconds.")
