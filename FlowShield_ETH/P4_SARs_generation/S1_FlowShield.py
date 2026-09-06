# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from typing import List
from tqdm import tqdm


day='twelveth'
DATA_CSV   = f"/workspace/0818_Day1/Dataset/semantics_flow_embedding/{day}_day_semantics_flow_embedding_with_label.parquet"
NUM_RUNS   = 5
EPOCHS     = 51
BATCH_TRAIN = 128
BATCH_EVAL  = 128
PROJ_DIM   = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiCrossFusion(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_s2t = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.attn_t2s = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm     = nn.LayerNorm(dim * 2)

    def forward(self, s_vec: torch.Tensor, t_vec: torch.Tensor):
        fusion_s, _ = self.attn_s2t(s_vec.unsqueeze(1), t_vec.unsqueeze(1), t_vec.unsqueeze(1))
        fusion_t, _ = self.attn_t2s(t_vec.unsqueeze(1), s_vec.unsqueeze(1), s_vec.unsqueeze(1))
        out = torch.cat([fusion_s.squeeze(1), fusion_t.squeeze(1)], dim=-1)
        return self.norm(out)

class LLMGNNBiCrossClassifier(nn.Module):
    def __init__(self, llm_dim=4096, struct_dim=32,
                 proj_dim=256, hidden_dim=64,
                 num_classes=2, heads=4):
        super().__init__()
        self.llm_proj    = nn.Linear(llm_dim, proj_dim)
        self.struct_proj = nn.Linear(struct_dim, proj_dim)
        self.cross       = BiCrossFusion(proj_dim, heads)
        self.classifier  = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, llm_vec: torch.Tensor, struct_vec: torch.Tensor):
        t = self.llm_proj(llm_vec)
        s = self.struct_proj(struct_vec)
        fused = self.cross(s, t)
        return self.classifier(fused)

# ---------------- Dataset ----------------
class AMLDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.llm_tensor   = torch.tensor(df[[f"dim_{i}" for i in range(4096)]].values, dtype=torch.float32).to(DEVICE, non_blocking=True) 
        self.struct_tensor= torch.tensor(df[[str(i) for i in range(32)]].values, dtype=torch.float32).to(DEVICE, non_blocking=True) 
        labels = df["tx_label"].values
        labels[labels > 0] = 1
        self.label_tensor = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.label_tensor)

    def __getitem__(self, i):
        return self.llm_tensor[i], self.struct_tensor[i], self.label_tensor[i]

# ---------------- 主流程 ----------------
def main():

    df = pd.read_parquet(DATA_CSV)
    df["index"] = df["index"].astype(int)

    dataset = AMLDataset(df)
    indices = list(range(len(dataset)))

    runs_metrics = []
    best_global_f1 = -1
    best_model_state = None

    for run in range(NUM_RUNS):
        print(f"\n========== Run {run+1}/{NUM_RUNS} ==========")
        train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=dataset.label_tensor, random_state=42+run)
        train_idx, val_idx  = train_test_split(train_idx, test_size=0.25, stratify=dataset.label_tensor[train_idx], random_state=100+run)

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=BATCH_TRAIN, shuffle=True)
        val_loader   = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=BATCH_EVAL)
        test_loader  = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=BATCH_EVAL)

        model = LLMGNNBiCrossClassifier().to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = -1
        no_improve = 0
        for epoch in range(1, EPOCHS+1):
            model.train()
            for llm_b, struct_b, label_b in train_loader:
                llm_b, struct_b, label_b = llm_b.to(DEVICE), struct_b.to(DEVICE), label_b.to(DEVICE)
                optim.zero_grad()
                logits = model(llm_b, struct_b)
                loss = criterion(logits, label_b)
                loss.backward()
                optim.step()

            model.eval()
            v_pred, v_true = [], []
            with torch.no_grad():
                for llm_b, struct_b, label_b in val_loader:
                    logits = model(llm_b.to(DEVICE), struct_b.to(DEVICE))
                    v_pred.extend(torch.argmax(logits, 1).cpu().tolist())
                    v_true.extend(label_b.tolist())

            f1 = precision_recall_fscore_support(v_true, v_pred, labels=[0, 1], zero_division=0)[2].mean()
            print(f"[Run {run+1}] Epoch {epoch}/{EPOCHS} | Val F1={f1:.4f}")
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_model_this_run = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 10:
                    print(f"🔔 Early Stopping at epoch {epoch}")
                    break


        model.load_state_dict(best_model_this_run)
        model.eval()
        t_pred, t_true = [], []
        with torch.no_grad():
            for llm_b, struct_b, label_b in test_loader:
                logits = model(llm_b.to(DEVICE), struct_b.to(DEVICE))
                t_pred.extend(torch.argmax(logits, 1).cpu().tolist())
                t_true.extend(label_b.tolist())

        p, r, f, _ = precision_recall_fscore_support(t_true, t_pred, labels=[0,1], zero_division=0)
        runs_metrics.append({"precision_0":p[0],"recall_0":r[0],"f1_0":f[0],"precision_1":p[1],"recall_1":r[1],"f1_1":f[1]})
        if best_val_f1 > best_global_f1:
            best_global_f1 = best_val_f1
            best_model_state = best_model_this_run

    torch.save(best_model_state, f"/workspace/0818_Day1/model/model_results/{day}_Flowshield_best_model.pt")
    print("\n====== Average over runs (Test) ======")
    df = pd.DataFrame(runs_metrics)
    avg_results = {}
    for c in [0, 1]:
        for m in ["precision", "recall", "f1"]:
            vals = df[f"{m}_{c}"].values
            mean_val, std_val = vals.mean(), vals.std()
            print(f"Class {c} {m}: {mean_val:.4f} ± {std_val:.4f}")
            avg_results[f"{m}_{c}_mean"] = mean_val
            avg_results[f"{m}_{c}_std"] = std_val


    out_csv = f"/workspace/0818_Day1/model/model_results/{day}_Flowshield_metrics.csv"
    df.to_csv(out_csv, index=False)

    avg_df = pd.DataFrame([avg_results])
    avg_df.to_csv(out_csv, mode="a", index=False)


if __name__ == '__main__':
    main()
