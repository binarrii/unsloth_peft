import ast
import json
import os
from urllib.parse import quote

from datasets import load_dataset
from sqlalchemy import create_engine, types
from tqdm import tqdm

_BATCH_SIZE = 1000

_STRING_AS_JSON = types.JSON(none_as_null=True)

username = quote("root")
password = quote(os.getenv("MYSQL_PASSWORD"))
host = "10.252.25.251"
port = 3309
database = "devel"


engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}",
    json_serializer=lambda o: ast.literal_eval(json.dumps(o)) if o else None,
)

# import pandas as pd
# df = pd.read_parquet("train.parquet")

dataset = load_dataset("BruceNju/crosswoz-sft", split="train")

df = dataset.to_pandas()
for start in tqdm(range(0, len(df), _BATCH_SIZE), desc="Writing to MySQL"):
    df_chunk = df.iloc[start : start + _BATCH_SIZE]
    df_chunk.to_sql(
        "crosswoz_sft",
        con=engine,
        if_exists="append",
        index=False,
        dtype={
            "dialog_id": types.BIGINT,
            "dialog_act": _STRING_AS_JSON,
            "history": _STRING_AS_JSON,
            "user_state": _STRING_AS_JSON,
            "goal": _STRING_AS_JSON,
            "sys_usr": _STRING_AS_JSON,
            "sys_state": _STRING_AS_JSON,
            "sys_state_init": _STRING_AS_JSON,
        },
    )
