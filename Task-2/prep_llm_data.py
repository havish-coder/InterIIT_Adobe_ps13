"""
Converts train_enriched.csv → llm_train_data.jsonl using the shared
prompt builder from prompt_utils.py.  Re-run this after enrich_vlm.py
to pick up any new vlm_description values.
"""

import json
import os
import pandas as pd
from prompt_utils import build_messages


def main():
    print("Loading data...")
    for path in ["train_enriched.csv", "train.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded {len(df)} rows from {path}")
            break
    else:
        print("ERROR: no training CSV found.")
        return

    df = df[df["content"].notna() & (df["content"].str.strip() != "")]
    print(f"  After content filter: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    records = []
    for _, row in df.iterrows():
        messages = build_messages(row.to_dict(), include_response=True)
        records.append({"messages": messages})

    out_path = "llm_train_data.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(records)} records to {out_path}")
    print("\n--- Example ---")
    # ensure_ascii=True avoids Windows cp1252 console encoding errors
    print(json.dumps(records[0], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
