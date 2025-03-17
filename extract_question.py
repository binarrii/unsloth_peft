import csv
import os


_final_txt_file = "normal_questions.txt"


if __name__ == "__main__":
    if os.path.exists(_final_txt_file):
        os.remove(_final_txt_file)

    total, correct, incorrect, i = 0, 0, 0, 0

    with (
        open("chinese_simpleqa.csv", newline="", encoding="utf-8") as csv_file,
        open(_final_txt_file, "a") as txt_file,
    ):
        csv_reader = csv.reader(csv_file)
        rows = []
        for n, row in enumerate(csv_reader):
            if n == 0:
                continue
            rows.append(row)
        rows = sorted(rows, key=lambda r: (r[1], r[2]))

        primary_category, secondary_category = None, None
        i, j = 64, 0
        for m, row in enumerate(rows):
            pc, sc = row[1].strip(), row[2].strip()
            if primary_category != pc:
                primary_category = pc
                i += 1
                j = 0
            if secondary_category != sc:
                secondary_category = sc
                j += 1
                txt_file.write(
                    f"{chr(i)}.{j // 26 + 1} {chr((j % 26) + 96)}){primary_category}-{secondary_category}\n"
                )

            txt_file.write(f"  - {row[3]}\n")
            print(f"{m + 1}. {row[3]}")
