import csv
import json


if __name__ == "__main__":
    with open("text2image10k.json", 'r', encoding='utf-8') as json_file:
        json_array = json.load(json_file)

    with open("text2image10k.csv", 'w', newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['prompt', 'image'])
        for obj in json_array:
            csv_writer.writerow([obj["text"], obj["image"]])
