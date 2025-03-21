#!/bin/bash

models_dir=~/Projects/models
models=(
  "deepseek-r1-7B-Q4_K_M.gguf"
  "deepseek-r1-1.5B-Q8_0.gguf"
  "qwen2.5-1.5b-instruct-fp16.gguf"
  "qwen2-1_5b-instruct-q4_k_m.gguf"
  "qwen2-1_5b-instruct-q5_k_m.gguf"
  #"Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
  "qwen2-7b-instruct-q4_k_m.gguf"
  #"qwq-32b-q4_k_m.gguf"
)

index=0
while [ $index -lt ${#models[@]} ]; do
    model="$models_dir/${models[$index]}"
    set -x
    ./llama_cpp/build_musa/bin/llama-bench -m $model -ngl 99 -fa 0,1 -t 10
    set +x
    printf "\n\n"

    index=$((index + 1))
done
