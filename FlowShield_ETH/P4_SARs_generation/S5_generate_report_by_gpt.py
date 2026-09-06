import json
import os
from pathlib import Path
from tqdm import tqdm
import openai


client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
PROMPT_JSON = Path("/workspace/Llama4AML_250721/explainability/llama_prompts/money_laundering_prompts.json")
OUTPUT_JSON = Path("/workspace/Llama4AML_250721/explainability/llama_prompts/gpt4_outputs.json")

MODEL = "gpt-4o"  


with open(PROMPT_JSON, "r") as f:
    prompt_data = json.load(f)




results = []
for item in tqdm(prompt_data):
    try:
        prompt = item["prompt"]
        file_name = item["file"]

        messages = [
            {"role": "system", "content": "You are a blockchain forensic analyst specializing in detecting money laundering paths."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )

        result = {
            "file": file_name,
          
            "response": response.choices[0].message.content
        }
        results.append(result)

    except Exception as e:
        print(f"[!] Error on {item['file']}: {e}")

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

