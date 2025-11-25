#!/bin/bash

# List of warmups
warmups=(float32 float16 bfloat16)

# List of model sizes
models=(small large xl 2.7B)

# Loop through all combinations
for w in "${warmups[@]}"; do
  for m in "${models[@]}"; do
    echo "Running: dtype=$w, model_size=$m"
    uv run python cs336_systems/benchmark.py \
      --model_size "$m" \
      --dtype "$w"
  done
done
