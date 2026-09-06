import os
import json
import requests
import pandas as pd
from web3 import Web3
from eth_utils import keccak, to_hex
from eth_abi import decode
from tqdm import tqdm
from collections import defaultdict
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from pathlib import Path


INFURA_PROJECT_ID = os.environ["INFURA_PROJECT_ID"]
ETHERSCAN_API_KEY = os.environ["ETHERSCAN_API_KEY"]
w3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{INFURA_PROJECT_ID}"))


default_topic_map = {
    '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef': 'Transfer(address,address,uint256)',
    '0x8c5be1e5ebec7d5bd14f714f6cf1796f20b57c3c5c5ed3e9e1e82e6ec5b66f9b': 'Approval(address,address,uint256)',
    '0x7c7bb9f0b469c21da4666496577565b8e0f6a5da9834e8d15b12603b260ca6c6': 'WithdrawRequested(address,address,uint96,uint40)',
}

abi_cache = {}
ABI_CACHE_FILE = "../Dataset/abi_cache.json"



crosschain_addresses = {
    '0x3624525075b88b24ecc29ce226b0cec1ffcb6976',  
    '0xd37bbe5744d730a1d98d8dc97c42f0ca46ad7146',  
}




if os.path.exists(ABI_CACHE_FILE):
    with open(ABI_CACHE_FILE, 'r') as f:
        abi_cache = json.load(f)

def get_contract_abi_cached(address):
    if address in abi_cache:
        return abi_cache[address]
    
    abi = get_contract_abi(address)
    if abi:
        try:
            json.loads(abi)  # 确保格式合法
            abi_cache[address] = abi
            with open(ABI_CACHE_FILE, 'w') as f:
                json.dump(abi_cache, f)
        except Exception:
            pass
    return abi



def is_eth_transfer(tx):
  
    return  (tx['contractAddress'] == "ETH" and float(tx['value']) > 0)

def is_token_transfer(tx):
   
    return  (  tx['contractAddress'] !=  "ETH" and  float(tx['value']) > 0)

def is_contract_call(tx):

    return (tx['to'] == tx.get('contractAddress', '') and float(tx.get('value', 0)) == 0)

def parse_thorchain_memo(memo_str):
    try:
        parts = memo_str.split(":")
        blockchain='Unknown'
        if len(parts) < 3:
            return None
        asset = parts[1]
       
        if '.' in asset:
            asset_parts = asset.split('.')
            if len(asset_parts) > 1:
                blockchain = asset_parts[0]
                asset=asset_parts[1]
        else:
            blockchain=asset
            if asset=='b':
                blockchain='BTC'
        
        to_address = parts[2]
        return {"asset": asset, "to_address": to_address, "blockchain": blockchain}
    except Exception:
        return None

    
def get_contract_abi(address):
    url = "https://api.etherscan.io/api"
    params = {
        "module": "contract",
        "action": "getabi",
        "address": address,
        "apikey": ETHERSCAN_API_KEY
    }
    response = requests.get(url, params=params)
    result = response.json()
    return result['result'] if result['status'] == '1' else None

def get_event_abi_by_topic(topic0, abi):
    for item in abi:
        if item.get("type") == "event":
            name = item["name"]
            types = ",".join([inp["type"] for inp in item["inputs"]])
            sig = f"{name}({types})"
            computed_topic = to_hex(keccak(text=sig)).lower()
            if computed_topic == topic0:
                return item
    return None

def decode_event(log, event_abi):
    decoded = {}
    indexed_inputs = [i for i in event_abi["inputs"] if i["indexed"]]
    non_indexed_inputs = [i for i in event_abi["inputs"] if not i["indexed"]]

    for idx, param in enumerate(indexed_inputs):
        topic_bytes = log["topics"][idx + 1]
        if param["type"] == "address":
            decoded[param["name"]] = Web3.to_checksum_address(topic_bytes[-20:].hex())
        else:
            decoded[param["name"]] = int.from_bytes(topic_bytes, byteorder="big")

    if non_indexed_inputs:
        types = [i["type"] for i in non_indexed_inputs]
        bytes_data = log["data"]
        if isinstance(bytes_data, bytes):
            bytes_data = bytes_data.hex()
        try:
            values = decode(types, bytes.fromhex(bytes_data[2:] if bytes_data.startswith("0x") else bytes_data))
            for i, param in enumerate(non_indexed_inputs):
                val = values[i]
                if param["type"].startswith("uint"):
                    val = val / 1e18
                decoded[param["name"]] = val
        except Exception as e:
            decoded["decode_error"] = str(e)
    return decoded

def analyze_tx_logs(tx_hash):
    time.sleep(0.2)  
    receipt = w3.eth.get_transaction_receipt(tx_hash)
    results = []

    if len(receipt.logs) <=20 :
        for log in receipt.logs:
            if not log.topics or len(log.topics) == 0:
                continue
            address = Web3.to_checksum_address(log.address)
            topic0 = to_hex(log.topics[0]).lower()
            event_abi = None
            abi_json = get_contract_abi_cached(address)
            if abi_json:
                abi = json.loads(abi_json)
                event_abi = get_event_abi_by_topic(topic0, abi)
            if event_abi:
                event_name = event_abi["name"]
                decoded = decode_event(log, event_abi)
            else:
                event_name = default_topic_map.get(topic0, "UnknownEvent")
                if event_name == "Transfer(address,address,uint256)":
                    try:
                        value = int(log.data.hex(), 16) / 1e18 if isinstance(log.data, bytes) else int(log.data, 16) / 1e18
                        decoded = {
                            "from": Web3.to_checksum_address(log.topics[1][-20:].hex()),
                            "to": Web3.to_checksum_address(log.topics[2][-20:].hex()),
                            "value": value
                        }
                    except Exception as e:
                        decoded = {"fallback_decode_error": str(e)}
                else:
                    decoded = {"warning": "No ABI found"}

            results.append({
                "contract": address,
                "event": event_name,
                **decoded
            })
    return results



if __name__ == "__main__":

    base_dir = Path("/workspace/0818_Day1/Dataset")
    each_day_dir = base_dir / "Each_Day_Tx"
    bg_dir = base_dir / "Bg_data"
    output_dir = base_dir / "semantic_awareness"
    output_dir.mkdir(exist_ok=True)


    day_names = {1:'first',
        2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh',
        8: 'eighth', 9: 'ninth', 10: 'tenth', 11: 'eleventh', 12: 'twelfth',
        13: 'thirteenth', 14: 'fourteenth'
    }
    for day in range(1,15):
        day_str = day_names[day]


       
        sample_path = each_day_dir / f"{day_str}_day_Tx.parquet"
        bg_path = bg_dir / f"Bg_data_{day_str}_day_Tx.parquet"
        output_path = output_dir / f"{day_str}_day_tx_with_semantic_type.csv"
        sample_tx_df = pd.read_parquet(sample_path)
        all_tx_df = pd.read_parquet(bg_path)
        tx_groups = all_tx_df.groupby('hash')

        sample_tx_df['transfer_to_true'] = ''
        sample_tx_df['transfer_from_true'] = ''
        sample_tx_df['cross_to_true'] = ''
        sample_tx_df['cross_asset'] = ''
        sample_tx_df['swap_token_address'] = ''
        sample_tx_df['swap_token_name'] = ''
        sample_tx_df['swap_token_symbol'] = ''
        sample_tx_df['swap_token_decimal'] = ''
        sample_tx_df['swap_amount'] = ''
        sample_tx_df['transaction_type']   = ''
       
        sample_tx_df['cross_to_blockchain'] = ''

        i=1
        for index, tx in sample_tx_df.iterrows():

            print(f"Processing transaction {i}/{len(sample_tx_df)}")

            tx_hash = tx['hash']
            i+=1

            group_txs = tx_groups.get_group(tx_hash) if tx_hash in tx_groups.groups else pd.DataFrame([tx])
            
            if tx['to'] in crosschain_addresses:
                sample_tx_df.loc[tx.name, 'transaction_type'] = 'cross_chain'
                if (tx['to']=='0x3624525075b88b24ecc29ce226b0cec1ffcb6976') or (tx['to']=='0xd37bbe5744d730a1d98d8dc97c42f0ca46ad7146'):
                    events= analyze_tx_logs(tx_hash)
                    for event in events:
                        if "Deposit" in event['event'] and 'memo' in event and isinstance(event['memo'], str):
                            parsed = parse_thorchain_memo(event['memo'])
                            if parsed:
                                sample_tx_df.loc[tx.name, 'cross_to_true'] = parsed['to_address']
                                sample_tx_df.loc[tx.name, 'cross_asset'] = parsed['asset']
                                sample_tx_df.loc[tx.name, 'cross_to_blockchain'] = parsed['blockchain']
                               
                continue
 
            if len(group_txs)==1 and is_eth_transfer(tx):
                sample_tx_df.loc[tx.name, 'transfer_to_true'] = tx['to']
                sample_tx_df.loc[tx.name, 'ETH_Transfer'] = '1'
                sample_tx_df.loc[tx.name, 'transaction_type'] = 'direct_eth_transfer'

                continue


            if len(group_txs)==1 and is_token_transfer(tx):
                sample_tx_df.loc[tx.name, 'transfer_to_true'] = tx['to']
                sample_tx_df.loc[tx.name, 'transaction_type'] = 'direct_token_transfer'
      
                continue
                   

            
            if len(group_txs) > 1:
                from_address=tx['from']
                to_address=tx['to']
                contract_address=tx['contractAddress']
    
                incoming = group_txs[group_txs['to'] == from_address]
                outcoming = group_txs[(group_txs['from'] == to_address)]
         
                if len(incoming)>0:
                    
                    incoming_contracts = incoming['contractAddress'].unique().tolist()
                    
                    incoming_contracts = [addr for addr in incoming_contracts if pd.notna(addr) and addr != contract_address]
                    
                    if len(incoming_contracts) > 0:
                       
                        incoming_row = incoming[incoming['contractAddress'] == incoming_contracts[0]]
                       
                        if len(incoming_row) > 0:
                            incoming_row = incoming_row.iloc[0]
                        sample_tx_df.loc[tx.name, 'transaction_type'] = 'token_swap'
                        sample_tx_df.loc[tx.name, 'swap_token_address'] = incoming_contracts[0]
                        sample_tx_df.loc[tx.name, 'swap_token_name'] = incoming_row.get('tokenName', '')
                        sample_tx_df.loc[tx.name, 'swap_token_symbol'] = incoming_row.get('tokenSymbol', '')
                        sample_tx_df.loc[tx.name, 'swap_token_decimal'] =incoming_row.get('tokenDecimal', '')
                        sample_tx_df.loc[tx.name, 'swap_amount'] = incoming_row['value']
                        
                        continue

                if len(outcoming)>0:
                    
                    outcoming_contracts = outcoming['contractAddress'].unique().tolist()

                    if contract_address in outcoming_contracts:
                        same_asset_txs = outcoming[outcoming['contractAddress'] == contract_address]
                        sample_tx_df.loc[tx.name, 'transaction_type'] = 'indirect_transfer'
                        sample_tx_df.loc[tx.name, 'transfer_to_true'] = same_asset_txs['to'].iloc[0]
                        
                        continue
                
                
                if contract_address == "ETH":
                    sample_tx_df.loc[tx.name, 'transaction_type'] = 'direct_eth_transfer'
                    sample_tx_df.loc[tx.name, 'transfer_to_true'] = tx['to']
                    
                    continue
                else:
                    sample_tx_df.loc[tx.name, 'transaction_type'] = 'direct_token_transfer'
                    sample_tx_df.loc[tx.name, 'transfer_to_true'] = tx['to']
                    
                    continue    
                    

        numeric_cols = ['swap_amount','swap_token_decimal']
        sample_tx_df[numeric_cols] = sample_tx_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        sample_tx_df.to_csv(output_path, index=False)
   
