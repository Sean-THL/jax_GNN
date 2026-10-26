#!/usr/bin/env bash
# Reproduce the earlier four-way comparison using the shared pipeline for all.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
cd "${PROJECT_DIR}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

# Check the backend before loading the full training dataset.
"${PYTHON}" -c 'import jax; print("Devices:", jax.devices(), flush=True)'
RUN_DIR="$(mktemp -d "${PROJECT_DIR}/checkpoints/full_run_XXXXXX")"
echo "Results directory: ${RUN_DIR}"

variants=(original_gat original_gatv2 new_gat new_gatv2)
for i in "${!variants[@]}"; do
    variant="${variants[$i]}"
    checkpoint="${RUN_DIR}/${variant}.msgpack"
    if [[ "${variant}" == original_* ]]; then
        convolution="${variant#original_}"
        train_command=("${PYTHON}" -m gnn train --pipeline padded-shared --model "${convolution}")
        test_command=("${PYTHON}" -m gnn test --pipeline padded-shared --model "${convolution}")
    else
        convolution="${variant#new_}"
        train_command=("${PYTHON}" -m gnn train --model "${convolution}")
        test_command=("${PYTHON}" -m gnn test --model "${convolution}")
    fi
    echo "Training ${variant}"
    "${train_command[@]}" \
        --train-data "${PROJECT_DIR}/training_data.json" \
        --batch-size "${BATCH_SIZE:-32}" \
        --epochs "${EPOCHS:-100}" \
        --patience "${PATIENCE:-15}" \
        --seed "${SEED:-42}" \
        --learning-rate "${LEARNING_RATE:-0.005}" \
        --validation-fraction 0.1 \
        --checkpoint "${checkpoint}" 2>&1 | tee "${RUN_DIR}/${variant}_train.log"
    echo "Testing ${variant}"
    "${test_command[@]}" \
        --test-data "${PROJECT_DIR}/testing_data.json" \
        --batch-size "${BATCH_SIZE:-32}" \
        --checkpoint "${checkpoint}" 2>&1 | tee "${RUN_DIR}/${variant}_test.log"
done
echo "Completed all four variants. Results: ${RUN_DIR}"
