import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd

batch_size = 8
num_passes = 100
num_warmup_runs = 10
head_dim_list = [16, 32, 64, 128]
seq_len_list = [256, 1024, 4096, 8192, 16384]

class pytorch_attention(nn.Module):
    def __init__(self, head_dim, seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.Q = nn.Parameter(torch.randn(head_dim, head_dim))
        self.K = nn.Parameter(torch.randn(head_dim, head_dim))
        self.V = nn.Parameter(torch.randn(head_dim, head_dim))

    def forward(self, x):
        q = x @ self.Q
        k = x @ self.K
        v = x @ self.V
        attn_scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = attn_weights @ v
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Collect results in long-form list
rows = []

for hd in head_dim_list:
    for sq in seq_len_list:

        model = pytorch_attention(hd, sq)
        x = torch.randn(batch_size, sq, hd)
        model.to(device)
        x = x.to(device)

        # warmup
        for _ in range(num_warmup_runs):
            _ = model.forward(x)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(num_passes):
            _ = model.forward(x)

        torch.cuda.synchronize()
        end = time.time()

        total_time = end - start

        rows.append({
            "head_dim": hd,
            "seq_len": sq,
            "time_sec": total_time
        })

# 2) Convert to DataFrame
df = pd.DataFrame(rows)

# 3) Pivot to matrix/table format
pivot_df = df.pivot(index="head_dim", columns="seq_len", values="time_sec")

# 4) Save matrix-style CSV
pivot_df.to_csv("attention_matrix_profile_time.csv")

print("\nSaved CSV as attention_matrix_profile_time.csv")
print(pivot_df)
