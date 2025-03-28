import os
from urllib.parse import quote

from datasets import load_dataset
from sqlalchemy import create_engine
from tqdm import tqdm

_BATCH_SIZE = 1000


username = quote("root")
password = quote(os.getenv("MYSQL_PASSWORD"))
host = "10.252.25.251"
port = 3309
database = "devel"

engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
)

# import pandas as pd
# df = pd.read_parquet("train.parquet")

dataset = load_dataset("BruceNju/crosswoz-sft", split="train")

df = dataset.to_pandas()
for start in tqdm(range(0, len(df), _BATCH_SIZE), desc="Writing to MySQL"):
    df_chunk = df.iloc[start : start + _BATCH_SIZE]
    df_chunk.to_sql("crosswoz_sft", con=engine, if_exists="append", index=False)
