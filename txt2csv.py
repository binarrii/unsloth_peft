import csv


if __name__ == "__main__":
    with open("sensitive_words.txt.c", "r", encoding="utf-8") as txt_file:
        txt_lines = txt_file.readlines()

    with open("sensitive_words.csv", "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        for line in txt_lines:
            if line.count("@@") != 1:
                continue
            w, c = line.split("@@")
            csv_writer.writerow([w.strip(), c.strip()])
