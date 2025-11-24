from cs336_basics import BasicsTransformerLM
import argparse
import time
import torch


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
    }
}
def benchmark_model(model, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 10000
    context_length = 512
    d_model = model_configs[args.model_size]["d_model"]
    num_layers = model_configs[args.model_size]["num_layers"]
    num_heads = model_configs[args.model_size]["num_heads"]
    d_ff = model_configs[args.model_size]["d_ff"]
    rope_theta = 10000

    model = BasicsTransformerLM(vocab_size, context_length, d_model, \
                num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, \
                rope_theta=rope_theta)

    input_data = torch.randint(0, 10000, (8, 512)).to(device)

    for _ in range(5):
        output = model(input_data)
    torch.cuda.synchronize()
    times = []
    num_trials = 10
    for trials in range(num_trials):
        torch.cuda.synchronize()
        start_time = time.time()
        model = model.to(device)
        model(input_data)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / num_trials
    # print the average time taken and the variance
    print(f"Average inference time for model size \
          {args.model_size}: {avg_time:.6f} seconds")
    print(f"Variance of inference times for model size {args.model_size}: \
          {torch.var(torch.tensor(times)).item():.6f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM")
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="Size of the model to benchmark"
    )
    args = parser.parse_args()
    benchmark_model(BasicsTransformerLM, args)