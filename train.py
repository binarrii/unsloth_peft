import argparse
import time

import torch
from datasets import load_dataset
from transformers import TextStreamer, TrainingArguments
from trl import SFTTrainer
from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
    UnslothTrainer,
    UnslothTrainingArguments,
)
from unsloth.chat_templates import get_chat_template


def _train(_args: argparse.Namespace):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"unsloth/{_args.base_model}",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

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
        system_message="""
        - You are 剧匠, created by WASU. You are a helpful assistant, good at writing screenplays. Apart from the screenplays, please answer other questions briefly.
        - 你是剧匠，由华数创建。你是一个有用的助手，擅长编写剧本。除了剧本之外，请简要回答其他问题。
        """,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def formatting_prompts_func(inputs):
        samples = inputs["text"]
        print(f">>>>> len(samples): {len(samples)}")
        texts = []
        for t in samples:
            text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            write a screenplay(写一个剧本)

            ### Response:
            {}
            """
            text = text.format(t) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    # `sample_by` line, paragraph, document
    dataset = load_dataset(
        "text", data_files="files/*.txt", sample_by="document", split="train"
    )
    # print(f">>>>>>>> {dataset.column_names}")
    # dataset = dataset["train"].train_test_split(train_size=0.9)["train"]
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Continued Pretraining
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=6,
            # num_train_epochs=1,  # Set this for 1 full training run.
            max_steps=60,
            learning_rate=4e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
    )

    start_memory, max_memory = _show_start_memory_stats()
    trainer_stats = trainer.train()
    _show_final_memory_stats(trainer_stats, start_memory, max_memory)

    def formatting_instructions_func(conversations):
        inputs = conversations["instruction"]
        outputs = conversations["output"]
        print(f">>>>> len(inputs): {len(inputs)}, len(outputs): {len(outputs)}")

        texts = []
        for _in, _out in zip(inputs, outputs):
            text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

                   ### Instruction:
                   {}

                   ### Response:
                   {}
                   """
            text = text.format(_in, _out) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    identity_dataset = load_dataset("json", data_files="identity.json", split="train")
    identity_dataset = identity_dataset.map(formatting_instructions_func, batched=True)

    # Instruction Finetuning
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=identity_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            max_steps=40,
            warmup_steps=4,
            # warmup_ratio = 0.1,
            # num_train_epochs = 1,
            learning_rate=5e-5,
            embedding_learning_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.00,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )

    start_memory, max_memory = _show_start_memory_stats()
    trainer_stats = trainer.train()
    _show_final_memory_stats(trainer_stats, start_memory, max_memory)

    # Local saving
    if _args.save_lora:
        model.save_pretrained(f"{_args.saved_path}/{_args.saved_name}-Screenplay-LoRA")
        tokenizer.save_pretrained(
            f"{_args.saved_path}/{_args.saved_name}-Screenplay-LoRA"
        )
        # model.push_to_hub("your_name/Qwen2.5-7B-bnb-4bit-ft", token = "...") # Online saving
        # tokenizer.push_to_hub("your_name/Qwen2.5-7B-bnb-4bit-ft", token = "...") # Online saving

    # Inference
    if _args.exec_infr:
        FastLanguageModel.for_inference(model)
        inputs = tokenizer(["写一个剧本"], return_tensors="pt").to("cuda")
        text_streamer = TextStreamer(tokenizer)
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

    # Saving to float16 for vLLM
    if _args.save_vllm:
        model.save_pretrained_merged(
            f"{_args.saved_path}/{_args.saved_name}-Screenplay-ft",
            tokenizer,
            # save_method="merged_4bit_forced",
        )
        # model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

    # GGUF / llama.cpp Conversion
    if _args.save_gguf:
        model.save_pretrained_gguf(
            f"{_args.saved_path}/{_args.saved_name}-Screenplay-q5_K_M",
            tokenizer,
            quantization_method="q5_k_m",
        )
        print(f"\n\n{tokenizer._ollama_modelfile}\n\n")
        # model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q5_k_m", token = "")


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
    parser.add_argument("--base-model", type=str, default="Qwen2.5-14B-Instruct")
    parser.add_argument("--exec-infr", action="store_true", default=False)
    parser.add_argument("--save-lora", action="store_true", default=True)
    parser.add_argument("--save-vllm", action="store_true", default=True)
    parser.add_argument("--save-gguf", action="store_true", default=False)
    parser.add_argument("--saved-path", type=str, default="./models")
    parser.add_argument("--saved-name", type=str, default="Qwen2.5-14B")
    _args = parser.parse_args()

    start_time = time.time()
    _train(_args)
    end_time = time.time()
    print(f"Training took: {end_time - start_time} seconds.")
