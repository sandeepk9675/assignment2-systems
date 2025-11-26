from os import times
from cs336_basics import BasicsTransformerLM
import argparse
import time
import torch
import pandas as pd
import os
import argparse
from tqdm import tqdm

from torch.optim import AdamW

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

def profile_memory(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = model_configs[args.model_size]

    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=args.ctx_len,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000
    ).to(device)

    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # lets do some warmup runs
    for _ in range(5):
        out = model(torch.randint(0, 10000, (4, args.ctx_len), device=device))

    torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    if args.mode != "forward":
        optimizer = AdamW(model.parameters(), lr=1e-4)

    input_data = torch.randint(0, 10000, (4, args.ctx_len), device=device)

    dtype = getattr(torch, args.dtype)
    use_autocast = device == "cuda" and args.dtype in ["float16", "bfloat16"]

    # Set mode once
    if args.mode == "forward":
        model.eval()
        context_manager = torch.inference_mode()
    else:
        model.train()
        context_manager = torch.enable_grad()

    for _ in tqdm(range(args.num_trials), desc="Running trials"):
        with context_manager:
            if use_autocast:
                with torch.cuda.amp.autocast(dtype=dtype):
                    out = model(input_data)
            else:
                out = model(input_data)

            if args.mode == "forward":
                continue

            loss = out.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    torch.cuda.memory._dump_snapshot(
        f"memory_profile_{args.mode}_{args.ctx_len}_{args.dtype}.pickle"
    )
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory profiling")

    parser.add_argument("--model_size", type=str, default="2.7B",
                        choices=["small", "medium", "large","xl", "2.7B"])

    parser.add_argument("--ctx_len", type=int, default=128,
                        choices=[128, 256, 512],
                        help="Context length")

    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "bfloat16", "float32"],
                        help="Computation dtype")

    parser.add_argument("--mode", type=str, default="forward",
                        choices=["forward", "full_training"],
                        help="forward = inference only, fulltraining = fwd+bwd+opt step")

    parser.add_argument("--num_trials", type=int, default=10)

    args = parser.parse_args()
    profile_memory(args)