import pandas as pd

from tqdm import tqdm
from pathlib import Path



tqdm.pandas()


def convert_row(row):
    tx_type = row['transaction_type']
    hash_id = row['hash']
    from_addr = row['from']
    to_addr = row['to']
    timestamp = row['timeStamp']
    value = row['value']

    token = row['tokenSymbol']
    contract = row['contractAddress'][:10]
    gasFee = row['gasFee']
    

    if tx_type in ['direct_eth_transfer', 'direct_token_transfer']:
        return f"Direct_transfer: In transaction {hash_id}, address {from_addr} sent {value} {token} ({contract}) to address {to_addr} at time {timestamp}, with gas fee {gasFee}."

    elif tx_type == 'token_swap':
        swap_amount = row['swap_amount']
        swap_token_symbol = row['swap_token_symbol']
        swap_token_address = row['swap_token_address'][:10]
        return f"Token_swap: In transaction {hash_id}, address {from_addr} used address {to_addr} to swap {value} {token} ({contract}) into {swap_amount} {swap_token_symbol} ({swap_token_address}) at time {timestamp}, with gas fee {gasFee}."

    elif tx_type == 'indirect_transfer':
        transfer_to_true = row['transfer_to_true']
        return f"Indirect_transfer: In transaction {hash_id}, address {from_addr} used address {to_addr} to transfer {value} {token} ({contract}) to address {transfer_to_true} at time {timestamp}, with gas fee {gasFee}."

    elif tx_type == 'cross_chain':

        cross_to_true = row['cross_to_true']
        cross_chain=row['cross_to_blockchain']
        return f"Cross_chain_transfer: In transaction {hash_id}, address {from_addr} used address {to_addr} to cross-chain transfer {value} {token} ({contract}) to address {cross_to_true} on {cross_chain} blockchain at time {timestamp}, with gas fee {gasFee}."

    else:
        return f"Unknown: In transaction {hash_id}, address {from_addr} sent {value} {token} ({contract}) to address {to_addr} at time {timestamp}, with gas fee {gasFee}."


day_names = {
    1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
    6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth',
    11: 'eleventh', 12: 'twelveth', 13: 'thirteenth', 14: 'fourteenth'
}

base_dir = Path("/workspace/0818_Day1/Dataset/semantic_awareness")

for day in range(1, 15):

    day_str = day_names[day]
    input_path = base_dir / f"{day_str}_day_tx_with_semantic_type.csv"
    output_path = base_dir / f"{day_str}_day_tx_with_semantic_txt.csv"


    df = pd.read_csv(input_path)
    df.fillna('', inplace=True)

    df['behavior_sentence'] = df.progress_apply(convert_row, axis=1)

    keep_cols = ['index', 'hash', 'from', 'to', 'value', 'timeStamp', 'tx_label', 'transaction_type', 'behavior_sentence']
    keep_cols = [col for col in keep_cols if col in df.columns]

    df[keep_cols].to_csv(output_path, index=False)

