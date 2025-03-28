import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm


def _str_to_json(s):
    return json.loads(s) if isinstance(s, str) and len(s) > 0 else None


def _run(args: argparse.Namespace):
    out_file = args.outfile or args.dataset.split("/")[-1].replace("-", "_")
    if not out_file.endswith(".jsonl"):
        out_file = f"{out_file}.jsonl"

    if os.path.exists(out_file):
        os.remove(out_file)

    dataset = load_dataset(args.dataset, split="train")
    with open(out_file, "a") as jsonl:
        for i in tqdm(range(0, args.maxrows), desc="Converting"):
            row = dataset[i]
            jsonl.write(
                json.dumps(
                    {
                        **row,
                        "dialog_act": _str_to_json(row["dialog_act"]),
                        "history": _str_to_json(row["history"]),
                        "user_state": _str_to_json(row["user_state"]),
                        "goal": _str_to_json(row["goal"]),
                        "sys_usr": _str_to_json(row["sys_usr"]),
                        "sys_state": _str_to_json(row["sys_state"]),
                        "sys_state_init": _str_to_json(row["sys_state_init"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BruceNju/crosswoz-sft")
    parser.add_argument("--maxrows", type=int, default=1000)
    parser.add_argument("--outfile", type=str, required=False)

    _run(parser.parse_args())
