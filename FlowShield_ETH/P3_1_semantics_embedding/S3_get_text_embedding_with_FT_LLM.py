import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import csv
import os

# === 路径配置 ===
base_model_path = "/workspace/Llama3-8B/LLM-Research/Meta-Llama-3-8B-Instruct"


jobs = [
    # {
    #     "name": "day1",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-23-54-54",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/first_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/first_day_txt_embeddings.csv",
    # },
    {
        "name": "day2",
        "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-22-00-10-49",
        "input": "/workspace/0818_Day1/Dataset/fund_flow/second_day_tx_flow_txt.csv",
        "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/second_day_txt_embeddings.csv",
    }
    # {
    #     "name": "day3",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-05-08-44",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/third_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/third_day_txt_embeddings.csv",
    # },
    #     {
    #     "name": "day4",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-09-45-16",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/fourth_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/fourth_day_txt_embeddings.csv",
    # },
    #     {
    #     "name": "day5",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-09-51-16",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/fifth_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/fifth_day_txt_embeddings.csv",
    # },
    #     {
    #     "name": "day6",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-09-53-46",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/sixth_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/sixth_day_txt_embeddings.csv",
    # },
    #     {
    #     "name": "day7",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-05-31-17",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/seventh_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/seventh_day_txt_embeddings.csv",
    # },
    #     {
    #     "name": "day8",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-05-36-39",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/eighth_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/eighth_day_txt_embeddings.csv",
    # },
    #     {
    #     "name": "day9",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-05-44-17",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/ninth_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/ninth_day_txt_embeddings.csv",
    # },
    #     {
    #     "name": "day10",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-05-51-16-50-19",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/tenth_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/tenth_day_txt_embeddings.csv",
    # },
    #     {
    #     "name": "day11",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-05-52-46",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/eleventh_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/eleventh_day_txt_embeddings.csv",
    # },
    #     {
    #     "name": "day12",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-05-56-37",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/twelveth_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/twelveth_day_txt_embeddings.csv",
    # },
    #     {
    #     "name": "day13",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-05-59-17",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/thirteenth_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/thirteenth_day_txt_embeddings.csv",
    # },
    #     {
    #     "name": "day14",
    #     "lora": "/workspace/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-08-21-06-10-16",
    #     "input": "/workspace/0818_Day1/Dataset/fund_flow/fourteenth_day_tx_flow_txt.csv",
    #     "output": "/workspace/0818_Day1/Dataset/Txt_Embedding/fourteenth_day_txt_embeddings.csv",
    # },
    
]


for job in jobs:
    print(f"=== Running {job['name']} ===")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa"
    )


    model = PeftModel.from_pretrained(model, job["lora"])
    model.eval()


    df = pd.read_csv(job["input"])


    with open(job["output"], mode="w", newline="") as f_out:
        writer = None

        for i, row in tqdm(df.iterrows(), total=len(df)):
            text = row["narrative"]
            index = int(row["index"])

            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nClassify this transaction narrative.\n{text}<|eot_id|>"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # 最后一层
                embedding = hidden_states.mean(dim=1).squeeze().to(torch.float32).cpu().numpy()


 
            embedding_dict = {"index": index, **{f"dim_{i}": val for i, val in enumerate(embedding)}}
            
            if writer is None:
                writer = csv.DictWriter(f_out, fieldnames=embedding_dict.keys())
                writer.writeheader()
            
            writer.writerow(embedding_dict)

    print(f"✅ Saved embeddings to: {job['output']}")
