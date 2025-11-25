from os import times
from cs336_basics import BasicsTransformerLM
import argparse
import time
import torch
import pandas as pd
import os


model_configs = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32
    },
}
def benchmark_model(model, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 10000
    context_length = 256
    d_model = model_configs[args.model_size]["d_model"]
    num_layers = model_configs[args.model_size]["num_layers"]
    num_heads = model_configs[args.model_size]["num_heads"]
    d_ff = model_configs[args.model_size]["d_ff"]
    rope_theta = 10000

    model = BasicsTransformerLM(vocab_size, context_length, d_model, \
                num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, \
                rope_theta=rope_theta)

    input_data = torch.randint(0, 10000, (4, context_length)).to(device)
    
    # number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = model.to(device)

    if args.warmup:
        for _ in range(5):
            model(input_data)

    fwd_times = []
    num_trials = 10
    for trials in range(num_trials):
        datatype = getattr(torch, args.dtype)
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.cuda.amp.autocast(dtype=datatype):
            model(input_data)
            torch.cuda.synchronize()
        end_time = time.time()
        fwd_times.append(end_time - start_time)
        model.zero_grad()
    avg_time_fwd = sum(fwd_times) / num_trials
    variance_fwd = torch.var(torch.tensor(fwd_times)).item()

    if args.warmup:
        for _ in range(5):
            model = model.to(device)
            model(input_data)
    

    bwd_times = []
    for _ in range(num_trials):
        model.train()
        datatype = getattr(torch, args.dtype)
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.cuda.amp.autocast(dtype=datatype):
            output = model(input_data)
            loss = output.sum()
            model.zero_grad()
            loss.backward()
            torch.cuda.synchronize()
        end_time = time.time()
        bwd_times.append(end_time - start_time)
    avg_time_bwd = sum(bwd_times) / num_trials
    variance_bwd = torch.var(torch.tensor(bwd_times)).item()
    # cleanup

    return avg_time_fwd, variance_fwd, avg_time_bwd, variance_bwd, num_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM")
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="Size of the model to benchmark"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Whether to perform warmup runs before benchmarking"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data type for mixed precision (float16, bfloat16, float32)"
    )

    args = parser.parse_args()
    path_to_save = f"benchmark_with_warmup_{args.dtype}.csv" if args.warmup else f"benchmark_without_warmup_{args.dtype}.csv"

    if os.path.exists(path_to_save):
        df = pd.read_csv(path_to_save)
    else:
        df = pd.DataFrame(columns=["Model Size", "Forward Avg Time",\
            "Forward Variance", "Backward Avg Time", \
            "Backward Variance", "Num Parameters"])
    size = args.model_size
    avg_time_fwd, variance_fwd, avg_time_bwd, variance_bwd, num_parameters = benchmark_model(BasicsTransformerLM, args)
    df = df._append({"Model Size": size,
                    "Forward Avg Time": avg_time_fwd,
                    "Forward Variance": variance_fwd,
                    "Backward Avg Time": avg_time_bwd,
                    "Backward Variance": variance_bwd,
                    "Num Parameters": num_parameters / 1e6}, ignore_index=True)
    print(f"Model Size: {size}, Forward Average Time: {avg_time_fwd:.6f}s, Forward Variance: {variance_fwd:.6f}s, \n  Backward Average Time: {avg_time_bwd:.6f}s, Backward Variance: {variance_bwd:.6f}s, Number of Parameters: {num_parameters/1e6:.3f}M")
    print("-"*80)
    df.to_csv(path_to_save, index=False)