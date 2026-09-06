import pandas as pd

from tqdm import tqdm
from pathlib import Path



tqdm.pandas()


def convert_row(row):
    tx_type = row['transaction_type']
    hash_id = row['txhash']
    from_addr = row['fromaddress']
    to_addr = row['toaddress']
    timestamp = row['date_time']
    value = row['amount']

    

    if tx_type in ['direct_btc_transfer']:
        return f"Direct_transfer: In transaction {hash_id}, address {from_addr} sent {value} BTC to address {to_addr} at time {timestamp}."

input_path = f"/workspace/Tx_info_Bitcoin.csv"
output_path = f"/workspace/Tx_Bybit_btc_with_semantic_txt.csv"


df = pd.read_csv(input_path)
df['transaction_type']='direct_btc_transfer'  # BTC transaction only has direct transfer type, ETH transaction has more types
df.fillna('', inplace=True)

df['behavior_sentence'] = df.progress_apply(convert_row, axis=1)

keep_cols = ['index', 'txhash', 'fromaddress', 'toaddress', 'amount', 'date_time', 'label', 'transaction_type', 'behavior_sentence']
keep_cols = [col for col in keep_cols if col in df.columns]

df[keep_cols].to_csv(output_path, index=False)