#!/usr/bin/env bash
# Reproduce the earlier four-way comparison using the shared pipeline for all.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

# Check the backend before loading the full training dataset.
"${PYTHON}" -c 'import jax; print("Devices:", jax.devices(), flush=True)'
RUN_DIR="$(mktemp -d "${PROJECT_DIR}/checkpoints/full_run_XXXXXX")"
echo "Results directory: ${RUN_DIR}"

variants=(original_gat original_gatv2 new_gat new_gatv2)
suffixes=('' _gatv2 _new _new_gatv2)
for i in "${!variants[@]}"; do
    variant="${variants[$i]}"
    suffix="${suffixes[$i]}"
    checkpoint="${RUN_DIR}/${variant}.msgpack"
    if [[ "${variant}" == original_* ]]; then
        convolution="${variant#original_}"
        train_command=("${PYTHON}" -c 'import sys; from jax_variants import train_main; train_main("original", sys.argv.pop(1))' "${convolution}")
        test_command=("${PYTHON}" -c 'import sys; from jax_variants import test_main; test_main("original", sys.argv.pop(1))' "${convolution}")
    else
        train_command=("${PYTHON}" "${PROJECT_DIR}/jax_train${suffix}.py")
        test_command=("${PYTHON}" "${PROJECT_DIR}/jax_test${suffix}.py")
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
