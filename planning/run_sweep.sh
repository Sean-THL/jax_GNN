#!/usr/bin/env bash
set -euo pipefail

# Representative values across the requested ranges. Add every integer to the
# arrays if a denser (and much more expensive) sweep is needed.
robots=( 10 20 30 40 50)
horizons=(10 20 30 40 50)
sampling_times=(0.05 0.1 0.2)

connectivity_mode=${CONNECTIVITY_MODE:-hard}
if [[ "$connectivity_mode" != "hard" && "$connectivity_mode" != "smooth" ]]; then
  echo "CONNECTIVITY_MODE must be hard or smooth" >&2
  exit 2
fi

output_dir="reports/sweep/${connectivity_mode}"
mkdir -p "$output_dir"
for robot_count in "${robots[@]}"; do
  for horizon in "${horizons[@]}"; do
    for dt in "${sampling_times[@]}"; do
      dt_label=${dt/./p}
      JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" .venv/bin/python -m planning \
        --scenario planning/examples/demo_scenario.json \
        --robots "$robot_count" \
        --connectivity-mode "$connectivity_mode" \
        --steps "$horizon" \
        --dt "$dt" \
        --output "${output_dir}/robots_${robot_count}_steps_${horizon}_dt_${dt_label}.json"
    done
  done
done
