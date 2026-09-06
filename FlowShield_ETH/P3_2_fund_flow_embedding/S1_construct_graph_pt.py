import pandas as pd
import torch
from torch_geometric.data import Data
import json
from torch_geometric.utils import degree
from sklearn.preprocessing import StandardScaler


def build_edge_classification_graph(csv_path, output_path,addr2id_path):
  
    df = pd.read_parquet(csv_path)


    address_set = set(df["from"]).union(df["to"])
    address_list = sorted(address_set)
    addr2id = {addr: i for i, addr in enumerate(address_list)}
    num_nodes = len(addr2id)


    src = df["from"].map(addr2id).tolist()
    dst = df["to"].map(addr2id).tolist()
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    edge_label = torch.tensor(df["tx_label"].values, dtype=torch.long)

    src_tensor = torch.tensor(src, dtype=torch.long)
    dst_tensor = torch.tensor(dst, dtype=torch.long)

    out_degree = degree(src_tensor, num_nodes=num_nodes).view(-1, 1)
    in_degree = degree(dst_tensor, num_nodes=num_nodes).view(-1, 1)


    deg_feat = torch.cat([in_degree, out_degree], dim=1)


    from_out_deg = deg_feat[src_tensor][:, 1].view(-1, 1)  
    to_in_deg    = deg_feat[dst_tensor][:, 0].view(-1, 1)  

    value = torch.tensor(df["value"].values, dtype=torch.float)
    value = (value - value.mean()) / (value.std() + 1e-6)
    value = value.view(-1, 1)
    timestamp = torch.tensor(df["timeStamp"].values, dtype=torch.float).view(-1, 1)
    timestamp = (timestamp - timestamp.mean()) / (timestamp.std() + 1e-6)  # 可选标准化


    edge_attr = torch.cat([value, from_out_deg,to_in_deg], dim=1)

    x = torch.tensor(deg_feat, dtype=torch.float)


    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label
    )
    data.edge_id = torch.tensor(df["index"].values, dtype=torch.long)


    torch.save(data, output_path)
    with open(addr2id_path, "w") as f:
        json.dump(addr2id, f)

    print(f"✅ Graph saved to {output_path}")
    print(f"Nodes: {num_nodes} | Edges: {edge_index.shape[1]}")


day_names = [ "first","second","third","fourth","fifth",'sixth',"seventh",
    "eighth","ninth","tenth","eleventh","twelveth","thirteenth","fourteenth"
]

for day in day_names:

    day_tx_flow_path=f"/workspace/0818_Day1/Dataset/Each_Day_Tx/{day}_day_Tx_after_flow.parquet"
    day_tx_graph_path=f"/workspace/0818_Day1/Dataset/Graph_data/{day}_day_Tx_after_flow.pt"
    addr2id_path=f"/workspace/0818_Day1/Dataset/Graph_data/{day}_day_address2id.json"


    build_edge_classification_graph(day_tx_flow_path,day_tx_graph_path,addr2id_path)