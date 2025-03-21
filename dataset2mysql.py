import json
import textwrap

import mysql.connector
from datasets import load_dataset


if __name__ == "__main__":
    dataset = load_dataset("Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese", split="train")

    conn = mysql.connector.connect(
        host="10.252.25.251",
        port=3309,
        user="root",
        password="",
        database="devel",
    )
    cursor = conn.cursor()

    insert_query = textwrap.dedent("""
    INSERT IGNORE INTO magpie_qwen2_pro_200k_chinese (
        uuid, model, gen_input_configs, instruction, response, conversations, 
        task_category, other_task_category, task_category_generator, difficulty, 
        intent, knowledge, difficulty_generator, input_quality, quality_explanation, 
        quality_generator, llama_guard_2, reward_model, instruct_reward, 
        min_neighbor_distance, repeat_count, min_similar_uuid, instruction_length, 
        response_length, language
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """)

    for i, row in enumerate(dataset):
        try:
            cursor.execute(
                insert_query,
                (
                    row["uuid"],
                    row["model"],
                    json.dumps(row["gen_input_configs"], ensure_ascii=False),
                    row["instruction"],
                    row["response"],
                    json.dumps(row["conversations"], ensure_ascii=False),
                    row["task_category"],
                    json.dumps(row["other_task_category"], ensure_ascii=False),
                    row["task_category_generator"],
                    row["difficulty"],
                    row["intent"],
                    row["knowledge"],
                    row["difficulty_generator"],
                    row["input_quality"],
                    row["quality_explanation"],
                    row["quality_generator"],
                    row["llama_guard_2"],
                    row["reward_model"],
                    row["instruct_reward"],
                    row["min_neighbor_distance"],
                    row["repeat_count"],
                    row["min_similar_uuid"],
                    row["instruction_length"],
                    row["response_length"],
                    row["language"],
                ),
            )
            print(f"{i + 1} rows inserted...")
        except Exception as ex:
            print(f"row skipped: {i + 1}, {row['uuid']}")
            print(f"{str(ex)}")

    conn.commit()
    print("final transaction committed")

    cursor.close()
    conn.close()
    print("resources released")
    print("data insert finished")
