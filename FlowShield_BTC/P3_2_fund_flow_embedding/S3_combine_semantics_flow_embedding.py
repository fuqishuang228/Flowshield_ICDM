import pandas as pd

day_names = [ "first"
]


for day in day_names:
    try:

        txt_path = f"/workspace/Bybit_btc/tx_txt_embeddings_train_2025-11-13-00-35-30.csv"
        str_path = f"/workspace/Graph_data/Bybit_btc/Embedding/edge_embeddings_best_run.csv"
        out_path = f"/workspace/Bybit_btc/semantics_flow_embedding_train_2025-11-13-00-35-30.parquet"

        


        txt_eb = pd.read_csv(txt_path)
        str_eb = pd.read_csv(str_path)

        str_eb = str_eb.rename(columns={"edge_id": "index"})

        merged = pd.merge(txt_eb, str_eb, on="index", how="inner")


        merged.to_parquet(out_path)

        in_path  =out_path
        out_2_path = f"/workspace/Bybit_btc/semantics_flow_embedding_with_label_train_2025-11-13-00-35-30.parquet"
        info_path= f"/workspace/Tx_Bybit_btc_with_semantic_txt.csv"


        info_df = pd.read_csv(info_path)[["index", "label"]]
        in_df   = pd.read_parquet(in_path)
        print(len(in_df))

        merged = pd.merge(in_df, info_df, on="index", how="left")
        print(len(merged))


        merged.to_parquet(out_2_path)


    except Exception as e:
        print(f"❌ {day}_day: {e}\n")
