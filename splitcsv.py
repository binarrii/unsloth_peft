import argparse
import csv


def new_csv_file(prefix, suffix, header, rows):
    fname = f"{prefix}-{suffix:03d}.csv"
    with open(fname, "w", newline="", encoding="utf-8") as out_csv:
        csv_writer = csv.writer(out_csv)
        csv_writer.writerow(header)
        csv_writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, default=None)
    parser.add_argument("--output-prefix", type=str, required=False)
    parser.add_argument("--rows-per-file", type=int, default=10000)
    parser.add_argument("--using-header", action="store_true", default=True)
    _args = parser.parse_args()

    input_file = _args.input_file
    prefix = _args.output_prefix or input_file.split(".csv")[0]
    rows_per_file = _args.rows_per_file or 10000
    using_header = _args.using_header or True

    with open(input_file, newline="", encoding="utf-8") as in_csv:
        csv_reader = csv.reader(in_csv)
        header, out_lines = None, []
        for i, row in enumerate(csv_reader):
            if i == 0 and using_header:
                header = row
                continue
            out_lines.append(row)
            if len(out_lines) == rows_per_file:
                suffix = (i - 1) // rows_per_file + 1
                new_csv_file(prefix, suffix, header, out_lines)
                out_lines = []

        if len(out_lines) != 0:
            suffix = (i - 1) // rows_per_file + 1
            new_csv_file(prefix, suffix, header, out_lines)
            out_lines = []
