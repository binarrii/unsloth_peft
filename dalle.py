import base64
import csv
import os
import re
import textwrap
import time
from io import BytesIO

from datasets import load_dataset
from openai import OpenAI
from PIL import Image

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


def generate_image(prompt, size="1792x1024", quality="hd", n=1):
    try:
        response = _openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
            response_format="b64_json",
        )

        images = []
        for data in response.data:
            image_data = base64.b64decode(data.b64_json)
            image = Image.open(BytesIO(image_data))
            images.append(image)

        return images
    except Exception as e:
        print(f"Error generating image: {e}")
        return []


def save_images(images, output_dir="generated_images", prefix="image"):
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    for i, image in enumerate(images):
        filename = f"{prefix}-{i}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        saved_paths.append(filepath)

    return saved_paths


if __name__ == "__main__":
    # _data = load_dataset("LLAAMM/text2image10k")["train"]
    _data = load_dataset("wanng/midjourney-v5-202304-clean")["train"]

    with open("text2image.csv", "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["prompt", "image"])

        for i, row in enumerate(_data):
            row_dict = dict(row)
            # prompt = row_dict["text"]
            prompt = row_dict["Content"]
            match = re.match('\\*\\*.+?\\*\\*', prompt)
            if match:
                prompt = match.group().strip('*')
            
            if i % 20000 == 0 and len(prompt) >= 50:
                images = generate_image(prompt)
                if images:
                    # paths = save_images(images, prefix=f"image_{i:04d}")
                    paths = save_images(images, prefix=f"image_{i:07d}")
                    translated_prompt = translate_text(prompt)
                    csv_writer.writerow([translated_prompt, paths[0]])
                    print(f"Generated {len(paths)} images:")
                    for path in paths:
                        print(f"- {path}")
                else:
                    print("No images were generated.")
                time.sleep(13.5)
