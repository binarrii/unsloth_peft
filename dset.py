import os
import docx2txt


def docx_to_txt(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            docx_path = os.path.join(directory, filename)
            txt_path = os.path.join(directory, filename.replace(".docx", ".txt"))

            text = docx2txt.process(docx_path)
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert .docx files to .txt files in a directory."
    )
    parser.add_argument(
        "--dir", type=str, help="The directory containing .docx files to convert."
    )
    args = parser.parse_args()

    docx_to_txt(args.dir)
