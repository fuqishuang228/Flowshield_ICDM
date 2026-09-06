import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import csv
import os

# === 路径配置 ===
base_model_path = "/workspace/Meta-Llama-3-8B-Instruct"


jobs = [

    {
        "name": " ",
        "lora": "/workspace/LLaMA-Factory/LLaMA-Factory/saves/Llama-3-8B-Instruct/lora/train_2025-11-13-00-35-30",
        "input": "/workspace/Bybit_btc/tx_flow_txt.csv",
        "output": "/workspace/Bybit_btc/tx_txt_embeddings_train_2025-11-13-00-35-30.csv",
    }
   
    
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
