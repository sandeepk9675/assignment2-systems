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
rows_fwd = []

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
        start_fwd = time.time()

        for _ in range(num_passes):
            _ = model.forward(x)

        torch.cuda.synchronize()
        end_fwd = time.time()

        total_time = end_fwd - start_fwd

        rows_fwd.append({
            "head_dim": hd,
            "seq_len": sq,
            "time_sec": total_time
        })

# 2) Convert to DataFrame
df = pd.DataFrame( rows_fwd)

# 3) Pivot to matrix/table format
pivot_df = df.pivot(index="head_dim", columns="seq_len", values="time_sec")

# 4) Save matrix-style CSV
pivot_df.to_csv("attention_matri_forward_profile_time.csv")

print("\nSaved CSV as attention_matrix_profile_time.csv")
print(pivot_df)

print('-'*80)


# backward pass profiling
rows_bwd = []

for hd in head_dim_list:
    for sq in seq_len_list:

        model = pytorch_attention(hd, sq)
        x = torch.randn(batch_size, sq, hd)
        model.to(device)
        x = x.to(device)

        # warmup
        for _ in range(num_warmup_runs):
            out = model.forward(x)
            loss = out.sum()
            loss.backward()

        torch.cuda.synchronize()
        
        # Measure memory before backward pass
        out = model.forward(x)
        loss = out.sum()
        torch.cuda.synchronize()
        mem_before_backward = torch.cuda.memory_allocated() / 1e9
        
        start_bwd = time.time()

        for _ in range(num_passes):
            out = model.forward(x)
            loss = out.sum()
            loss.backward()
            model.zero_grad()

        torch.cuda.synchronize()
        end_bwd = time.time()

        total_time = end_bwd - start_bwd

        rows_bwd.append({
            "head_dim": hd,
            "seq_len": sq,
            "time_sec": total_time,
            "memory_before_backward_gb": mem_before_backward
        })

# 2) Convert to DataFrame
df_bwd = pd.DataFrame(rows_bwd)

# 3) Create separate pivot tables for time and memory
pivot_df_bwd_time = df_bwd.pivot(index="head_dim", columns="seq_len", values="time_sec")
pivot_df_bwd_memory = df_bwd.pivot(index="head_dim", columns="seq_len", values="memory_before_backward_gb")

# 4) Save both CSV files
pivot_df_bwd_time.to_csv("attention_matrix_backward_profile_time.csv")
pivot_df_bwd_memory.to_csv("attention_matrix_backward_memory_before_backward.csv")

print("\nBackward pass timing:")
print(pivot_df_bwd_time)
print("\nMemory before backward (GB):")
print(pivot_df_bwd_memory)