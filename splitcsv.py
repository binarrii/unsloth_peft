import csv

_input_file = "normal_20250324.csv"
_with_header = True
_rows_per_file = 10000


def new_csv_file(prefix, suffix, header, rows):
    with open(f"{prefix}-{suffix:03d}.csv", 'w', newline="", encoding="utf-8") as out_csv:
        csv_writer = csv.writer(out_csv)
        csv_writer.writerow(header)
        csv_writer.writerows(rows)


if __name__ == "__main__":
    prefix = _input_file.split('.csv')[0]

    with open(_input_file, newline="", encoding="utf-8") as in_csv:
        csv_reader = csv.reader(in_csv)
        header, out_lines = None, []
        for i, row in enumerate(csv_reader):
            if i == 0 and _with_header:
                header = row
                continue
            out_lines.append(row)
            if len(out_lines) == _rows_per_file:
                suffix = (i - 1) // _rows_per_file + 1
                new_csv_file(prefix, suffix, header, out_lines)
                out_lines = []
        
        if len(out_lines) != 0:
            suffix = (i - 1) // _rows_per_file + 1
            new_csv_file(prefix, suffix, header, out_lines)
            out_lines = []
