import json
import os
import shutil
import textwrap
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from openai import OpenAI

_IMAGE_PATH = "images"

_openai = OpenAI()


def translate_text(text):
    try:
        response = _openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": textwrap.dedent("""
                    - Translate the following text to Chinese: `{}`
                    - Just do translation, no other outputs (IMPORTANT!!!)
                    """).format(text),
                }
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text


if __name__ == "__main__":
    if os.path.exists(_IMAGE_PATH):
        shutil.rmtree(_IMAGE_PATH)
    os.makedirs(_IMAGE_PATH, exist_ok=True)

    _data = load_dataset("LLAAMM/text2image10k")["train"]

    def _process_row(row_dict, i):
        copied_row = row_dict.copy()
        if "text" in copied_row:
            print(f"Translating item {i + 1}/1000...")
            copied_row["text"] = translate_text(copied_row["text"])
        if "image" in copied_row:
            image = copied_row["image"]
            if hasattr(image, "save"):
                image_path = f"{_IMAGE_PATH}/image_{i:04d}.jpg"
                image.save(image_path, format="JPEG")
                copied_row["image"] = image_path
        return copied_row

    with ThreadPoolExecutor(max_workers=4) as executor:
        _futures = []
        for i, row in enumerate(_data):
            _row_dict = dict(row)
            if i % 5 == 0 and len(_row_dict["text"]) >= 30:
                _futures.append(executor.submit(_process_row, _row_dict, i))
            if len(_futures) >= 1000:
                break
        json_data = [f.result() for f in _futures]

    with open("text2image10k.json", "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)

    print("Dataset successfully converted to JSON.")
