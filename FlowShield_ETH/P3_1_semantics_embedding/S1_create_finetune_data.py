import pandas as pd
import json
from pathlib import Path

day_names = [
"first","second","third","fourth","fifth",'sixth',"seventh",
    "eighth","ninth","tenth","eleventh","twelveth","thirteenth","fourteenth"
]


for day in day_names:
    print(f'processing {day} day')
    df_path=f"/workspace/0818_Day1/Dataset/fund_flow/{day}_day_tx_flow_txt.csv"



    re_path = Path(f"/workspace/0818_Day1/Dataset/finetune_data/{day}_day")

    re_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(df_path)  
    df_processed = df

  
    df_processed['tx_label'] = df_processed['tx_label'].apply(lambda x: 1 if x in [1, 2] else 0)

    from sklearn.model_selection import train_test_split
    
    if len(df_processed)<50000:
        train_df, val_df = train_test_split(df_processed, train_size=5000, stratify=df_processed["tx_label"], random_state=42)
    else:
        train_df, val_df = train_test_split(df_processed, train_size=5000, test_size=45000, stratify=df_processed["tx_label"], random_state=42)

    print(len(train_df))
    print(len(val_df))
    
    val_entries = []
    for _, row in val_df.iterrows():
        entry = {
            "instruction": "Classify this transaction narrative.",
            "input": row["narrative"],
            "output": str(row["tx_label"])
        }
        val_entries.append(entry)

    with open(f"/workspace/0818_Day1/Dataset/finetune_data/{day}_day/val_data.json", "w") as f_val:
        json.dump(val_entries, f_val, indent=2)
    with open(f"/workspace/LLaMA-Factory/data/{day}_day_val_data.json", "w") as f_val:
        json.dump(val_entries, f_val, indent=2)



    val_entries_with_index = []
    for _, row in val_df.iterrows():
        entry = {
            "index": int(row["index"]),
            "instruction": "Classify this transaction narrative.",
            "input": row["narrative"],
            "output": str(row["tx_label"])
        }
        val_entries_with_index.append(entry)

    with open(f"/workspace/0818_Day1/Dataset/finetune_data/{day}_day/val_data_with_index.json", "w") as f_val_with_index_1:
        json.dump(val_entries_with_index, f_val_with_index_1, indent=2)



    
    train_entries = []
    for _, row in train_df.iterrows():
        entry = {
            "instruction": "Classify this transaction narrative.",
            "input": row["narrative"],
            "output": str(row["tx_label"])
        }
        train_entries.append(entry)

    with open(f"/workspace/0818_Day1/Dataset/finetune_data/{day}_day/train_data.json", "w") as f_train:
        json.dump(train_entries, f_train, indent=2)
    with open(f"/workspace/LLaMA-Factory/data/{day}_day_train_data.json", "w") as f_train:
        json.dump(train_entries, f_train, indent=2)

    
    train_entries_indexed = []
    for _, row in train_df.iterrows():
        entry = {
            "index": int(row["index"]),
            "instruction": "Classify this transaction narrative.",
            "input": row["narrative"],
            "output": str(row["tx_label"])
        }
        train_entries_indexed.append(entry)

    with open(f"/workspace/0818_Day1/Dataset/finetune_data/{day}_day/train_data_with_index.json", "w") as f_train_indexed:
        json.dump(train_entries_indexed, f_train_indexed, indent=2)
