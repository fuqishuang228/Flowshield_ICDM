import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path


def classify_transfer_type(seed_tx, parallel_tx, downstream_tx, tx_sentence_df):
    seed_index = seed_tx['index']
    seed_to = seed_tx['to']
    seed_from = seed_tx['from']
    seed_value = seed_tx['value']
    token = seed_tx['tokenSymbol']
    contract = seed_tx['contractAddress'][:10]
    seed_time = seed_tx['timeStamp']
    seed_behavior = tx_sentence_df.loc[tx_sentence_df['index'] == seed_index, 'behavior_sentence'].values[0]
    if seed_tx['transaction_type'] == 'indirect_transfer':
        seed_to = seed_tx['transfer_to_true']
    elif seed_tx['transaction_type'] == 'token_swap':
        seed_from= seed_tx['to'] 
        seed_to = seed_tx['from']
        contract=seed_tx['swap_token_address'][:10] 
        seed_value=seed_tx['swap_amount']
        token=seed_tx['swap_token_symbol']  


    out_sum = downstream_tx['value'].astype(float).sum()
    time_diff_max = (downstream_tx['timeStamp'].max() - seed_time) / 3600 if not downstream_tx.empty else None
    if seed_tx['transaction_type'] == 'cross_chain':
        blockchain= seed_tx.get('cross_to_blockchain', 'unknown')
        to_addr=seed_tx.get('cross_to_true', 'unknown')
        cross_asset= seed_tx.get('cross_asset', 'unknown')
        seed_behavior = f"{seed_behavior}\n cross_chain_transfer: The recipient {seed_to} transferred the {seed_value} {token} ({contract}) to {to_addr} address on {blockchain} blockchain using {cross_asset} token."
    
    if downstream_tx.empty:
        return "no_forward", (
            f"{seed_behavior}\n "
            f"no_forward: The recipient {seed_to} did not transfer out the {seed_value} {token} ({contract}) received from {seed_from} within 24 hours."
        )


    if 0.8 * seed_value <= out_sum <= 1.2 * seed_value:
        if len(downstream_tx) == 1:
            tx = downstream_tx.iloc[0]
            hours = (tx['timeStamp'] - seed_time) / 3600
            return "direct_transfer", (
                f"{seed_behavior}\n "
                f"direct_transfer: The recipient {seed_to} transferred the {seed_value} {token} ({contract}) received from {seed_from} "
                f"in a single transaction to {tx['to']}, sending {tx['value']} {token} ({contract}) within {hours:.2f} hours."
            )
        else:
            return "entire_disperse", (
                f"{seed_behavior}\n "
                f"entire_disperse: The recipient {seed_to} dispersed the {seed_value} {token} ({contract}) received from {seed_from} "
                f"into {len(downstream_tx)} transactions sent to {downstream_tx['to'].nunique()} addresses, "
                f"totaling {out_sum} {token} ({contract}) within {time_diff_max:.2f} hours."
            )

 
    if len(parallel_tx) > 1 and out_sum >= 1.2*seed_value:
        return "aggregate", (
            f"{seed_behavior}\n "
            f"aggregate: The recipient {seed_to} aggregated the {seed_value} {token} ({contract}) from {seed_from} with "
            f"{len(parallel_tx) - 1} other incoming transactions, and transferred them in {len(downstream_tx)} transactions "
            f"to {downstream_tx['to'].nunique()} addresses, totaling {out_sum} {token} ({contract}) within {time_diff_max:.2f} hours."
        )


    if out_sum < 0.8 * seed_value:
        return "partial_disperse", (
            f"{seed_behavior}\n "
            f"partial_disperse: The recipient {seed_to} partially dispersed the {seed_value} {token} ({contract}) from {seed_from} "
            f"in {len(downstream_tx)} transactions to {downstream_tx['to'].nunique()} addresses, "
            f"totaling {out_sum} {token} ({contract}) within {time_diff_max:.2f} hours."
        )

    # Case: unknown
    return "unknown", (
        f"{seed_behavior}\n "
        f"unknown: The fund movement behavior does not clearly fit into known patterns."
    )


def slice_to_threshold90(sorted_df, value_col, threshold):

    if sorted_df.empty:
        return sorted_df
    cum = sorted_df[value_col].cumsum().values
    pos = np.searchsorted(cum, threshold, side='left')
    if pos < len(sorted_df):
        return sorted_df.iloc[:pos+1]
    else:
        return sorted_df 


def normalize_seed_row(seed_row):

    tx_type = seed_row['transaction_type']
    out = {
        'index': seed_row['index'],
        'hash': seed_row['hash'],
        'tx_label': seed_row.get('tx_label', ''),
        'transaction_type': tx_type,
        'time': int(seed_row['timeStamp']),
    }

    if tx_type == 'indirect_transfer':
        
        out['to_addr'] = str(seed_row['transfer_to_true']).lower()
        out['from_addr'] = str(seed_row['from']).lower()
        out['token_addr'] = str(seed_row['contractAddress'])
        out['token_symbol'] = seed_row.get('tokenSymbol', '')
        out['value'] = float(seed_row['value'])
    elif tx_type == 'token_swap':
        
        out['to_addr'] = str(seed_row['from']).lower()   
        out['from_addr'] = str(seed_row['to']).lower()
        out['token_addr'] = str(seed_row['swap_token_address'])
        out['token_symbol'] = seed_row.get('swap_token_symbol', '')
        out['value'] = float(seed_row['swap_amount'])
    else:

        out['to_addr'] = str(seed_row['to']).lower()
        out['from_addr'] = str(seed_row['from']).lower()
        out['token_addr'] = str(seed_row['contractAddress'])
        out['token_symbol'] = seed_row.get('tokenSymbol', '')
        out['value'] = float(seed_row['value'])

    return out

def pick_downstream_for_seed(tx_df, seed_norm, window_seconds, send_ratio_thr, consumed, keep_all_if_less_90=True):

    seed_to = seed_norm['to_addr']
    seed_time = seed_norm['time']
    seed_value = seed_norm['value']
    seed_token = seed_norm['token_addr']
    time_upper = seed_time + window_seconds
    send_value_thr = send_ratio_thr * seed_value

    ds = tx_df[
        (tx_df['from'].str.lower() == seed_to) &
        (tx_df['timeStamp'].between(seed_time, time_upper)) &
        (tx_df['contractAddress'] == seed_token) &
        (tx_df['value'] >= send_value_thr) &
        (~tx_df['index'].isin(consumed))
    ].sort_values('timeStamp').copy()

    if ds.empty:
        return ds

    threshold = 0.9 * seed_value
    sliced = slice_to_threshold90(ds, 'value', threshold)


    if not keep_all_if_less_90 and sliced['value'].sum() < threshold:
        return ds.iloc[[]]  # 空


    consumed.update(sliced['index'].tolist())
    return sliced



def pick_upstream_tx(seed_row, all_tx_df, tolerance_ratio=0.9):
    from_addr = seed_row['from']
    ts = seed_row['timeStamp']
    token = seed_row['tokenName']
    target_value = seed_row['value']


    direct_df = all_tx_df[
        (all_tx_df['transaction_type'].isin(['direct_eth_transfer', 'direct_token_transfer'])) &
        (all_tx_df['to'] == from_addr) &
        (all_tx_df['tokenName'] == token) &
        (all_tx_df['timeStamp'] < ts)
    ].copy()
    direct_df['source_type'] = 'direct'


    swap_df = all_tx_df[
        (all_tx_df['transaction_type'] == 'token_swap') &
        (all_tx_df['from'] == from_addr) &
        (all_tx_df['swap_token_name'] == token) &
        (all_tx_df['timeStamp'] < ts)
    ].copy()
    if not swap_df.empty:
        swap_df['value'] = swap_df['swap_amount']
        swap_df['tokenName'] = swap_df['swap_token_name']
        swap_df['to'] = swap_df['from']
        swap_df['source_type'] = 'swap'

   
    indirect_df = all_tx_df[
        (all_tx_df['transaction_type'] == 'indirect_transfer') &
        (all_tx_df['transfer_to_true'] == from_addr) &
        (all_tx_df['tokenName'] == token) &
        (all_tx_df['timeStamp'] < ts)
    ].copy()
    indirect_df['source_type'] = 'indirect'


    source_df = pd.concat([direct_df, swap_df, indirect_df], ignore_index=True)
    source_df = source_df.sort_values(by='timeStamp', ascending=False).reset_index(drop=True)

    selected_rows = []
    cumulative = 0.0
    min_required = target_value * tolerance_ratio
    for _, tx in source_df.iterrows():
        if tx['value'] > 0.1 * target_value:
            cumulative += tx['value']
            selected_rows.append(tx)
        if cumulative >= min_required:
            break

    return pd.DataFrame(selected_rows)



def pick_parallel_for_seed(tx_df, seed_norm, window_seconds, recv_ratio_thr):
    seed_to = seed_norm['to_addr']
    seed_time = seed_norm['time']
    seed_value = seed_norm['value']
    seed_token = seed_norm['token_addr']
    time_upper = seed_time + window_seconds
    recv_value_thr = recv_ratio_thr * seed_value

    us = tx_df[
        (tx_df['to'].str.lower() == seed_to) &
        (tx_df['timeStamp'].between(seed_time, time_upper)) &
        (tx_df['contractAddress'] == seed_token) &
        ((tx_df['value'] - seed_value).abs() < 0.05 * seed_value)  
    ].copy()
    return us


def process_group(args):

    (to_addr, seed_rows, tx_df, tx_sentence_df, window_seconds,
     send_ratio_thr, recv_ratio_thr, out_dir,tx_df_with_txtype) = args

    consumed = set()
    outputs = []


    seed_rows = sorted(seed_rows, key=lambda r: int(r[1]['timeStamp']))


    for i, seed_row in seed_rows:
        seed_norm = normalize_seed_row(seed_row)

        downstream_tx = pick_downstream_for_seed(
            tx_df, seed_norm, window_seconds, send_ratio_thr, consumed,
            keep_all_if_less_90=True
        )
        parallel_tx = pick_parallel_for_seed(
            tx_df, seed_norm, window_seconds, recv_ratio_thr
        )


        transfer_type, narrative = classify_transfer_type(
            seed_row, parallel_tx, downstream_tx, tx_sentence_df
        )


        upstream_tx = pick_upstream_tx(seed_row, tx_df_with_txtype)

        seed_time = seed_norm['time']
        seed_index = seed_norm['index']
        seed_hash = seed_row['hash']

        result_df = pd.DataFrame({
            'index': [seed_index],
            'tx_type': [narrative.split(':')[0]],
            'transfer_type': [transfer_type],
            'narrative': [narrative]
        })
        result_df.to_csv(f"{out_dir}/{seed_time}_{seed_index}_{seed_hash}_narrative_flow.csv", index=False)
        downstream_tx.to_csv(f"{out_dir}/{seed_time}_{seed_index}_{seed_hash}_downstream_tx.csv", index=False)
        parallel_tx.to_csv(f"{out_dir}/{seed_time}_{seed_index}_{seed_hash}_parallel_tx.csv", index=False)
        upstream_tx.to_csv(f"{out_dir}/{seed_time}_{seed_index}_{seed_hash}_upstream_tx.csv", index=False)

        outputs.append({
            'index': seed_index,
            'timeStamp': seed_row['timeStamp'],
            'hash': seed_hash,
            'tx_label': seed_row.get('tx_label', ''),
            'narrative': narrative
        })

    return pd.DataFrame(outputs)
def extract_actual_to(row):
    tx_type = row['transaction_type']
    if tx_type == 'indirect_transfer':
        return row['transfer_to_true']
    elif tx_type == 'token_swap':
        return row['from']
    elif tx_type == 'cross_chain':
        return row['cross_to_true']
    else:
        return row['to']



tx_path='/workspace/0818_Day1/Dataset/Bg_data/Bg_data_first_day_Tx.parquet'
tx_sentence_path='/workspace/0818_Day1/Dataset/semantic_awareness/first_day_tx_with_semantic_txt.csv'


cols_needed = ['index', 'hash', 'from', 'to', 'value', 'timeStamp', 'contractAddress', 'tokenSymbol',
               'transaction_type', 'transfer_to_true', 'swap_amount', 'swap_token_address', 'tx_label','cross_to_blockchain', 'cross_to_true', 'cross_asset']

dtypes = {
    'index': 'int32',
    'value': 'float64',
    'timeStamp': 'int32',
    'from': 'category',
    'to': 'category',
    'contractAddress': 'category',
    'transaction_type': 'category',
    'swap_amount': 'float64'
}


print('读取交易数据')
tx_df = pd.read_parquet(tx_path)

tx_df['value'] = pd.to_numeric(tx_df['value'], errors='coerce')
tx_df['timeStamp'] = pd.to_numeric(tx_df['timeStamp'], errors='coerce')

tx_df['timeStamp'] = tx_df['timeStamp'].astype(int)
tx_df['from'] = tx_df['from'].astype(str).str.lower()
tx_df['to'] = tx_df['to'].astype(str).str.lower()
tx_df['contractAddress'] = tx_df['contractAddress'].astype(str)

print('读取交易语义数据')

tx_sentence_df = pd.read_csv(tx_sentence_path)

print('读取种子交易数据')

seed_path='/workspace/0818_Day1/Dataset/semantic_awareness/first_day_tx_with_semantic_type.csv'

seed_txs = pd.read_csv(seed_path)



seed_txs['value'] = pd.to_numeric(seed_txs['value'], errors='coerce')
seed_txs['timeStamp'] = pd.to_numeric(seed_txs['timeStamp'], errors='coerce')

seed_txs['timeStamp'] = seed_txs['timeStamp'].astype(int)
seed_txs['from'] = seed_txs['from'].astype(str).str.lower()
seed_txs['to'] = seed_txs['to'].astype(str).str.lower()
seed_txs['contractAddress'] = seed_txs['contractAddress'].astype(str)

seed_txs['_seed_to_actual'] = seed_txs.apply(extract_actual_to, axis=1)







tx_df_with_txtype=seed_txs

groups = []

flow_out_path = Path('/workspace/0818_Day1/Dataset/fund_flow/first_day')

flow_out_path.mkdir(parents=True, exist_ok=True)

for to_addr, grp in seed_txs.groupby('_seed_to_actual'):
    groups.append((
        to_addr,
        list(grp.iterrows()), 
        tx_df,
        tx_sentence_df,
        24*60*60,              
        0.02,                   
        0.02,                   
        flow_out_path, 
        tx_df_with_txtype
    ))


procs = 16
with Pool(processes=procs) as pool:
    dfs = list(tqdm(pool.imap_unordered(process_group, groups), total=len(groups)))

all_narratives_df = pd.concat([df for df in dfs if df is not None and not df.empty], ignore_index=True)
all_narratives_df.to_csv("/workspace/0818_Day1/Dataset/fund_flow/first_day_tx_flow_txt.csv", index=False)

