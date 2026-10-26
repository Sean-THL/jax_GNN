#!/usr/bin/env bash
# Train the historical original baselines and the two new shared variants.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
cd "${PROJECT_DIR}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

# Check the backend before loading datasets.
"${PYTHON}" -c 'import jax; print("Devices:", jax.devices(), flush=True)'
RUN_DIR="$(mktemp -d "${PROJECT_DIR}/checkpoints/full_run_XXXXXX")"
echo "Results directory: ${RUN_DIR}"
echo "Historical originals use 20 validation graphs; new variants use 100 independent test graphs."

for variant in gat gatv2; do
    checkpoint="${RUN_DIR}/historical_${variant}.msgpack"
    echo "Training historical original ${variant}"
    "${PYTHON}" -m gnn train --pipeline historical --model "${variant}" \
        --batch-size "${BATCH_SIZE:-32}" \
        --epochs "${BASELINE_EPOCHS:-500}" \
        --patience "${BASELINE_PATIENCE:-50}" \
        --checkpoint "${checkpoint}" 2>&1 | tee "${RUN_DIR}/historical_${variant}_train.log"
    echo "Evaluating historical original ${variant} on its validation split"
    "${PYTHON}" -m gnn test --pipeline historical --model "${variant}" \
        --batch-size "${BATCH_SIZE:-32}" \
        --checkpoint "${checkpoint}" 2>&1 | tee "${RUN_DIR}/historical_${variant}_validation.log"
done

for variant in gat gatv2; do
    checkpoint="${RUN_DIR}/new_${variant}.msgpack"
    echo "Training new ${variant}"
    "${PYTHON}" -m gnn train --model "${variant}" \
        --train-data "${PROJECT_DIR}/training_data.json" \
        --batch-size "${BATCH_SIZE:-32}" \
        --epochs "${EPOCHS:-100}" \
        --patience "${PATIENCE:-15}" \
        --seed "${SEED:-42}" \
        --learning-rate "${LEARNING_RATE:-0.005}" \
        --validation-fraction 0.1 \
        --checkpoint "${checkpoint}" 2>&1 | tee "${RUN_DIR}/new_${variant}_train.log"
    echo "Testing new ${variant} on the independent test file"
    "${PYTHON}" -m gnn test --model "${variant}" \
        --test-data "${PROJECT_DIR}/testing_data.json" \
        --batch-size "${BATCH_SIZE:-32}" \
        --checkpoint "${checkpoint}" 2>&1 | tee "${RUN_DIR}/new_${variant}_test.log"
done
echo "Completed all four entry points. Results: ${RUN_DIR}"
