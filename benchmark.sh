#!/bin/bash

# List of warmups
warmups=(0 1)

# List of model sizes
models=(small large xl 2.7B)

# Loop through all combinations
for w in "${warmups[@]}"; do
  for m in "${models[@]}"; do
    echo "Running: warmup=$w, model_size=$m"
    uv run python cs336_systems/benchmarking_script.py \
      --model_size "$m" \
      --warmup "$w"
  done
done
