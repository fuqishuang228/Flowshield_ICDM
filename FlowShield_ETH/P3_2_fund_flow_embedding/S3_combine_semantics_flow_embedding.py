import pandas as pd

day_names = [ "first","second","third","fourth","fifth",'sixth',"seventh",
    "eighth","ninth","tenth","eleventh","twelveth","thirteenth","fourteenth"
]


for day in day_names:
    try:

        txt_path = f"/workspace/0818_Day1/Dataset/Txt_Embedding/{day}_day_txt_embeddings.csv"
        str_path = f"/workspace/0818_Day1/Dataset/Graph_data/{day}_day_Embedding/edge_embeddings_best_run.csv"
        out_path = f"/workspace/0818_Day1/Dataset/semantics_flow_embedding/{day}_day_semantics_flow_embedding.parquet"


        txt_eb = pd.read_csv(txt_path)
        str_eb = pd.read_csv(str_path)

        str_eb = str_eb.rename(columns={"edge_id": "index"})

        merged = pd.merge(txt_eb, str_eb, on="index", how="inner")


        merged.to_parquet(out_path)

        in_path  = f"/workspace/0818_Day1/Dataset/semantics_flow_embedding/{day}_day_semantics_flow_embedding.parquet"
        out_2_path = f"/workspace/0818_Day1/Dataset/semantics_flow_embedding/{day}_day_semantics_flow_embedding_with_label.parquet"
        info_path= f"/workspace/0818_Day1/Dataset/Each_Day_Tx/{day}_day_Tx_after_flow.parquet"


        info_df = pd.read_parquet(info_path)[["index", "tx_label"]]
        in_df   = pd.read_parquet(in_path)
        print(len(in_df))

        merged = pd.merge(in_df, info_df, on="index", how="left")
        print(len(merged))


        merged.to_parquet(out_2_path)


    except Exception as e:
        print(f"❌ {day}_day: {e}\n")
