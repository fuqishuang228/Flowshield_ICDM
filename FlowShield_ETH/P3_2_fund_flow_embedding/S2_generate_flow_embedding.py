import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import numpy as np
from torch_geometric.utils import degree
import os



class EdgeClassifier(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=16):
        super().__init__()
        self.gcn1 = GCNConv(node_feat_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, edge_index, edge_attr, edge_list, return_embedding=False):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        src = edge_index[0, edge_list]
        dst = edge_index[1, edge_list]
        edge_input = torch.cat([x[src], x[dst]], dim=1)
        if return_embedding:
            return edge_input
        else:
            return self.edge_mlp(edge_input)

day_names =[ "first","second","third","fourth","fifth",'sixth',"seventh",
    "eighth","ninth","tenth","eleventh","twelveth","thirteenth","fourteenth"
]
for day in day_names:

    graph_path = f"/workspace/0818_Day1/Dataset/Graph_data/{day}_day_Tx_after_flow.pt"
    save_dir = f"/workspace/0818_Day1/Dataset/Graph_data/{day}_day_Embedding/"
    os.makedirs(save_dir, exist_ok=True)



    global_best_f1 = -1         
    global_best_state = None

    metrics_all = []

    for run in range(10):
        print(f"\n🚀 Run {run + 1}/10")
        data = torch.load(graph_path)
        data.edge_label[data.edge_label == 2] = 1

        y = data.edge_label.cpu().numpy()
        edge_indices = np.arange(len(y))
        train_idx, temp_idx, y_train, y_temp = train_test_split(edge_indices, y, test_size=0.4, stratify=y, random_state=42 + run)
        val_idx, test_idx, _, _ = train_test_split(temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42 + run)

        class_sample_count = np.bincount(y)
        weight = 1.0 / (class_sample_count + 1e-6)
        weight = torch.tensor(weight, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EdgeClassifier(data.x.size(1), data.edge_attr.size(1)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss(weight=weight.to(device))
        data = data.to(device)

        def train():
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, train_idx)
            loss = criterion(out, data.edge_label[train_idx])
            loss.backward()
            optimizer.step()
            return loss.item()

        def evaluate(split_idx):
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.edge_attr, split_idx)
                pred = out.argmax(dim=1).cpu()
                true = data.edge_label[split_idx].cpu()
                report = classification_report(true, pred, digits=4, output_dict=True)
                f1 = report["weighted avg"]["f1-score"]
            return f1, pred, true, report

        best_f1 = 0
        patience = 300
        counter = 0
        best_path = os.path.join(save_dir, f"best_model_run{run + 1}.pt")

        for epoch in range(1, 1000):
            loss = train()
            val_f1, _, _, _ = evaluate(val_idx)
            if val_f1 > best_f1:
                best_f1 = val_f1
                counter = 0
                torch.save(model.state_dict(), best_path)
            else:
                counter += 1
                if counter >= patience:
                    break

        model.load_state_dict(torch.load(best_path))
        test_f1, pred, true, report = evaluate(test_idx)
        cm = confusion_matrix(true, pred)


        metrics_all.append({
        "run": run + 1,
        "f1_0": report["0"]["f1-score"],
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        
        "f1_1": report["1"]["f1-score"],
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],

        "support_0": report["0"]["support"],
        "support_1": report["1"]["support"],
        "confusion_matrix": cm.tolist()
    })



        edge_ids = data.edge_id[test_idx].cpu().numpy()
        results_df = pd.DataFrame({
            "edge_id": edge_ids,
            "true_label": true.numpy(),
            "pred_label": pred.numpy()
        })
        results_df.to_csv(os.path.join(save_dir, f"prediction_results_run{run + 1}.csv"), index=False)

        with torch.no_grad():
            edge_embeddings = model(data.x, data.edge_index, data.edge_attr, torch.arange(data.edge_index.shape[1], device=device), return_embedding=True)
        edge_emb_np = edge_embeddings.cpu().numpy()
        edge_ids_all = data.edge_id.cpu().numpy()
        edge_emb_df = pd.DataFrame(edge_emb_np)
        edge_emb_df.insert(0, "edge_id", edge_ids_all)
        edge_emb_df.to_csv(os.path.join(save_dir, f"edge_embeddings_run{run + 1}.csv"), index=False)

        if report["1"]["f1-score"] == max(m["f1_1"] for m in metrics_all):
            best_embedding_path = os.path.join(save_dir, "edge_embeddings_best_run.csv")
            edge_emb_df.to_csv(best_embedding_path, index=False)
            print(f"✅ Saved best edge embedding from Run {run + 1} with F1 = {test_f1:.4f}")
        
    
        run_f1_target = report["1"]["f1-score"]   
        if run_f1_target > global_best_f1:
            global_best_f1   = run_f1_target
            global_best_state = model.state_dict().copy()   
            torch.save(global_best_state, os.path.join(save_dir, "best_edge_classifier.pt"))
            print(f"🌟 New global best model saved (Run {run+1}, F1_1={run_f1_target:.4f})")


    results_summary = pd.DataFrame(metrics_all)


    mean_metrics = results_summary[[
        "f1_0", "precision_0", "recall_0",
        "f1_1", "precision_1", "recall_1"
    ]].mean()

    std_metrics = results_summary[[
        "f1_0", "precision_0", "recall_0",
        "f1_1", "precision_1", "recall_1"
    ]].std()

    summary_df = pd.DataFrame({
        "mean": mean_metrics,
        "std": std_metrics
    })


    summary_df.to_csv(os.path.join(save_dir, "summary_metrics_by_class.csv"))
    results_summary.to_csv(os.path.join(save_dir, "detailed_metrics_per_run.csv"), index=False)


    print(f"\n🔎  Loading best model (F1* = {global_best_f1:.4f}) for full‑graph inference …")
    best_model = EdgeClassifier(data.x.size(1), data.edge_attr.size(1)).to(device)
    best_model.load_state_dict(torch.load(os.path.join(save_dir, "best_edge_classifier.pt")))
    best_model.eval()

    all_idx = torch.arange(data.edge_index.size(1), device=device)
    with torch.no_grad():
        logits_all = best_model(data.x, data.edge_index, data.edge_attr, all_idx)
        probs_all  = F.softmax(logits_all, dim=1).cpu().numpy()
        preds_all  = logits_all.argmax(dim=1).cpu().numpy()


    full_df = pd.DataFrame({
        "edge_id": data.edge_id.cpu().numpy(),
        "true_label": data.edge_label.cpu().numpy(),
        "pred_label": preds_all,
        "prob_0": probs_all[:,0],
        "prob_1": probs_all[:,1]
    })

    full_path = os.path.join(save_dir, "all_edges_prediction.csv")
    full_df.to_csv(full_path, index=False)



