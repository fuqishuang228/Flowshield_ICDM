import pandas as pd
from pathlib import Path
from collections import deque
from multiprocessing import Pool, cpu_count
import os

# ============ 配置路径 ============
SEED_FILE = '/workspace/Llama4AML_250721/model_results/days_results/first_day/first_day_monl_info_detect.csv'
FLOW_DIR = Path('/workspace/Llama4AML_250721/dataset/narrative_flow_tS')
OUTPUT_DIR = Path('/workspace/Llama4AML_250721/explainability/sankey_parallel_updated')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SKIPPED_LOG = OUTPUT_DIR / "skipped_seeds_updated.txt"  # 新增日志文件路径

# ============ 函数：追踪递归下游 ============
# ============ 函数：追踪递归下游 ============

def process_downstream(seed_tuple, max_depth=5):
    visited = set()
    queue = deque([(seed_tuple, 0)])  # 记录当前深度
    tx_downstream = pd.DataFrame()
    seed_index_list = pd.read_csv(SEED_FILE)['index'].values  # 提前加载，加速

    while queue:
        (timeStamp, index, tx_hash), depth = queue.popleft()
        tx_key = (timeStamp, index, tx_hash)

        if tx_key in visited or depth > max_depth:
            continue
        visited.add(tx_key)

        file_path = FLOW_DIR / f"{timeStamp}_{index}_{tx_hash}_downstream_tx.csv"
        if not file_path.exists():
            continue

        try:
            downstream_df = pd.read_csv(file_path)
        except Exception:
            continue

        if not downstream_df.empty:
            # 保留 SEED_FILE 中存在的 index 行
            downstream_df = downstream_df[downstream_df['index'].isin(seed_index_list)]

            if len(downstream_df) > 50:
                with open(SKIPPED_LOG, "a") as f:
                    f.write(f"{index}\n")
                print(f"⚠️ 子交易 {index} 达到行数限制（{len(downstream_df)} > 50），停止递归，但保留当前层")
                tx_downstream = pd.concat([tx_downstream, downstream_df], ignore_index=True)
                continue

            tx_downstream = pd.concat([tx_downstream, downstream_df], ignore_index=True)
            for _, row in downstream_df.iterrows():
                next_tuple = (row['timeStamp'], row['index'], row['hash'])
                if next_tuple not in visited:
                    queue.append((next_tuple, depth + 1))  # 深度 +1

    tx_downstream = tx_downstream.drop_duplicates(subset=['index']).reset_index(drop=True)
    return visited, tx_downstream


# def process_downstream(seed_tuple):
#     visited = set()
#     queue = deque([seed_tuple])
#     tx_downstream = pd.DataFrame()

#     while queue:
#         timeStamp, index, tx_hash = queue.popleft()
#         tx_key = (timeStamp, index, tx_hash)

#         if tx_key in visited:
#             continue
#         visited.add(tx_key)

#         file_path = FLOW_DIR / f"{timeStamp}_{index}_{tx_hash}_downstream_tx.csv"
#         if not file_path.exists():
#             continue

#         try:
#             downstream_df = pd.read_csv(file_path)
#         except Exception:
#             continue

#          # ✅ 如果超过限制，直接返回空 DataFrame，并记录 seed_index
#         if len(downstream_df) > 400:
#             with open(SKIPPED_LOG, "a") as f:
#                 f.write(f"{index}\n")
#             print(f"⚠️ Seed {index} 被跳过（下游文件行数 {len(downstream_df)} > 80）")
#             return visited, pd.DataFrame()

#         if not downstream_df.empty:
#             tx_downstream = pd.concat([tx_downstream, downstream_df], ignore_index=True)
#             for _, row in downstream_df.iterrows():
#                 next_tuple = (row['timeStamp'], row['index'], row['hash'])
#                 if next_tuple not in visited:
#                     queue.append(next_tuple)

#     tx_downstream = tx_downstream.drop_duplicates(subset=['index']).reset_index(drop=True)
#     return visited, tx_downstream


# ============ 函数：扩展 Sankey 边 ============
def augment_sankey_transactions(tx_downstream, full_df):
    sankey_df = full_df[full_df['index'].isin(tx_downstream['index'])].copy()

    swap_df = sankey_df[sankey_df['transaction_type'] == 'token_swap'].copy()
    reversed_swaps = swap_df.copy()
    reversed_swaps['from'] = swap_df['to']
    reversed_swaps['to'] = swap_df['from']
    reversed_swaps['value'] = swap_df['swap_amount']
    reversed_swaps['tokenName'] = swap_df['swap_token_name']
    reversed_swaps['tokenSymbol'] = swap_df['swap_token_symbol']
    reversed_swaps['tokenDecimal'] = swap_df['swap_token_decimal']
    reversed_swaps['contractAddress'] = swap_df['swap_token_address']

    cross_df = sankey_df[sankey_df['transaction_type'] == 'cross_chain'].copy()
    cross_df['to'] = cross_df['cross_to_true']

    indirect_df = sankey_df[sankey_df['transaction_type'] == 'indirect_transfer'].copy()
    indirect_df['to'] = indirect_df['transfer_to_true']

    sankey_df = pd.concat([sankey_df, reversed_swaps, cross_df, indirect_df], ignore_index=True)
    sankey_df = sankey_df.sort_values(by='index').reset_index(drop=True)
    return sankey_df


# ============ 单个种子交易处理逻辑 ============
def process_one_seed(seed_row_dict):
    row = pd.Series(seed_row_dict)
    seed_index = row['index']
    seed_tuple = (row['timeStamp'], row['index'], row['hash'])

    try:
        visited, tx_downstream = process_downstream(seed_tuple)

        seed_tx_df = pd.DataFrame([row])[[
            'hash', 'from', 'to', 'value', 'timeStamp', 'blockNumber',
            'tokenName', 'tokenSymbol', 'tokenDecimal', 'contractAddress',
            'isError', 'gasPrice', 'gasUsed', 'gasLimit', 'gasFee',
            'tx_label', 'from_label', 'to_label', 'index'
        ]]
        tx_downstream = pd.concat([tx_downstream, seed_tx_df], ignore_index=True)
        tx_downstream = tx_downstream.drop_duplicates(subset=['index']).sort_values(by='index')

        # 加载 full df 的子集（避免重复读）
        full_df = pd.read_csv(SEED_FILE, usecols=lambda x: x != 'Unnamed: 0')

        sankey_df = augment_sankey_transactions(tx_downstream, full_df)
        sankey_outfile = OUTPUT_DIR / f"{seed_index}_sankey_augmented.csv"
        sankey_df.to_csv(sankey_outfile, index=False)

        print(f"✅ Seed {seed_index} 完成，共 {len(sankey_df)} 条")
        return seed_index
    except Exception as e:
        print(f"❌ Seed {seed_index} 失败：{e}")
        return None


# ============ 主函数 ============
def main():
    print("🚀 多进程启动")
    df = pd.read_csv(SEED_FILE)
    #把df按照index列升序
    df = df.sort_values(by='index').reset_index(drop=True)
    df=df[df['value']>1e-2]
    # df=df[df['hash'] == '0xbf80907830e46317da2c1708a13a9f016e242f8a6db6e6b0706ea5f2328cb001']
    
    seed_rows = df.to_dict(orient='records')

    # with Pool(processes=min(cpu_count(), 1)) as pool:  # 建议不超过 CPU 数 - 1
    #     results = pool.map(process_one_seed, seed_rows)
    for i, row in enumerate(seed_rows):  
        print
        process_one_seed(row)  

    # print("🏁 所有任务完成")  
    print("🏁 所有任务完成")


if __name__ == "__main__":
    main()


# import pandas as pd
# from pathlib import Path
# from collections import deque
# from multiprocessing import cpu_count, get_context, TimeoutError
# import os

# # ============ 配置路径 ============
# SEED_FILE = '/workspace/Llama4AML_250721/model_results/days_results/first_day/first_day_monl_info_detect.csv'
# FLOW_DIR = Path('/workspace/Llama4AML_250721/dataset/narrative_flow_tS')
# OUTPUT_DIR = Path('/workspace/Llama4AML_250721/explainability/sankey_parallel')
# LOG_DIR = Path('/workspace/Llama4AML_250721/explainability/log')
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# LOG_DIR.mkdir(parents=True, exist_ok=True)

# TIMEOUT_SECONDS = 180  # 超时时间

# # ============ 函数：追踪递归下游 ============
# def process_downstream(seed_tuple):
#     from collections import deque
#     visited = set()
#     queue = deque([seed_tuple])
#     tx_downstream = pd.DataFrame()

#     while queue:
#         timeStamp, index, tx_hash = queue.popleft()
#         tx_key = (timeStamp, index, tx_hash)

#         if tx_key in visited:
#             continue
#         visited.add(tx_key)

#         file_path = FLOW_DIR / f"{timeStamp}_{index}_{tx_hash}_downstream_tx.csv"
#         if not file_path.exists():
#             continue

#         try:
#             downstream_df = pd.read_csv(file_path)
#         except Exception:
#             continue

#         if not downstream_df.empty:
#             tx_downstream = pd.concat([tx_downstream, downstream_df], ignore_index=True)
#             for _, row in downstream_df.iterrows():
#                 next_tuple = (row['timeStamp'], row['index'], row['hash'])
#                 if next_tuple not in visited:
#                     queue.append(next_tuple)

#     tx_downstream = tx_downstream.drop_duplicates(subset=['index']).reset_index(drop=True)
#     return visited, tx_downstream

# # ============ 函数：扩展 Sankey 边 ============
# def augment_sankey_transactions(tx_downstream, full_df):
#     sankey_df = full_df[full_df['index'].isin(tx_downstream['index'])].copy()

#     swap_df = sankey_df[sankey_df['transaction_type'] == 'token_swap'].copy()
#     reversed_swaps = swap_df.copy()
#     reversed_swaps['from'] = swap_df['to']
#     reversed_swaps['to'] = swap_df['from']
#     reversed_swaps['value'] = swap_df['swap_amount']
#     reversed_swaps['tokenName'] = swap_df['swap_token_name']
#     reversed_swaps['tokenSymbol'] = swap_df['swap_token_symbol']
#     reversed_swaps['tokenDecimal'] = swap_df['swap_token_decimal']
#     reversed_swaps['contractAddress'] = swap_df['swap_token_address']

#     cross_df = sankey_df[sankey_df['transaction_type'] == 'cross_chain'].copy()
#     cross_df['to'] = cross_df['cross_to_true']

#     indirect_df = sankey_df[sankey_df['transaction_type'] == 'indirect_transfer'].copy()
#     indirect_df['to'] = indirect_df['transfer_to_true']

#     sankey_df = pd.concat([sankey_df, reversed_swaps, cross_df, indirect_df], ignore_index=True)
#     sankey_df = sankey_df.sort_values(by='index').reset_index(drop=True)
#     return sankey_df

# # ============ 子进程任务 ============
# def process_one_seed(seed_row_dict):
#     row = pd.Series(seed_row_dict)
#     seed_index = row['index']
#     seed_tuple = (row['timeStamp'], row['index'], row['hash'])

#     try:
#         visited, tx_downstream = process_downstream(seed_tuple)

#         seed_tx_df = pd.DataFrame([row])[[  # 补入种子
#             'hash', 'from', 'to', 'value', 'timeStamp', 'blockNumber',
#             'tokenName', 'tokenSymbol', 'tokenDecimal', 'contractAddress',
#             'isError', 'gasPrice', 'gasUsed', 'gasLimit', 'gasFee',
#             'tx_label', 'from_label', 'to_label', 'index'
#         ]]
#         tx_downstream = pd.concat([tx_downstream, seed_tx_df], ignore_index=True)
#         tx_downstream = tx_downstream.drop_duplicates(subset=['index']).sort_values(by='index')

#         full_df = pd.read_csv(SEED_FILE, usecols=lambda x: x != 'Unnamed: 0')
#         sankey_df = augment_sankey_transactions(tx_downstream, full_df)
#         sankey_outfile = OUTPUT_DIR / f"{seed_index}_sankey_augmented.csv"
#         sankey_df.to_csv(sankey_outfile, index=False)

#         with open(LOG_DIR / "finished.txt", "a") as f:
#             f.write(f"{seed_index}\n")

#         print(f"✅ Seed {seed_index} 完成，共 {len(sankey_df)} 条")
#     except Exception as e:
#         with open(LOG_DIR / "failed.txt", "a") as f:
#             f.write(f"{seed_index}\n")
#         print(f"❌ Seed {seed_index} 失败：{e}")

# # ============ 加超时的封装器 ============
# def run_with_timeout(seed_row_dict, timeout=TIMEOUT_SECONDS):
#     ctx = get_context("spawn")
#     with ctx.Pool(1) as pool:
#         result = pool.apply_async(process_one_seed, (seed_row_dict,))
#         try:
#             result.get(timeout=timeout)
#         except TimeoutError:
#             seed_index = seed_row_dict['index']
#             with open(LOG_DIR / "timeout_failed.txt", "a") as f:
#                 f.write(f"{seed_index}\n")
#             print(f"⏱️ Seed {seed_index} 超时跳过")

# # ============ 主函数 ============
# def main():
#     print("🚀 启动多进程 + 超时控制")
#     df = pd.read_csv(SEED_FILE).sort_values(by='index').reset_index(drop=True)
#     seed_rows = df.to_dict(orient='records')

#     for seed_row in seed_rows:
#         seed_index = seed_row['index']
#         print(f"🛠️ 正在处理 Seed {seed_index}")
#         run_with_timeout(seed_row)

#     print("🏁 所有交易处理完毕")

# if __name__ == "__main__":
#     main()

