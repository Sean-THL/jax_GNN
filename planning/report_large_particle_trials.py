"""Summarize the controlled large-population collision experiments."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('reports/trials/11-11/large_particle_collision_trials'))
    args = parser.parse_args()
    rows = []
    population_rows = []
    best = {}
    for path in sorted(args.root.glob('particles_*/*/weight_*.json')):
        payload = json.loads(path.read_text())
        row = dict(payload['summary'])
        row['evaluated_candidates'] = payload['planner_report']['evaluated_candidates']
        row['selected_particle'] = payload['planner_report']['index']
        row['result_file'] = str(path.relative_to(args.root))
        rows.append(row)
        if row['collision_filter_enabled'] and payload['planner_report'].get('population_acceptance'):
            population_rows.append((row['particles'],row['collision_weight'],payload['planner_report']['population_acceptance']))
    rows.sort(key=lambda row: (row['particles'], row['collision_filter_enabled'], row['collision_weight']))
    for count in sorted({row['particles'] for row in rows}):
        eligible = [row for row in rows if row['particles'] == count and not row['collision_filter_enabled'] and row['cp_success']]
        if eligible:
            chosen = min(eligible, key=lambda row: (row['physical_overlap_pairs'], row['violating_pairs'], row['violating_pair_intervals'], row['collision_weight']))
            best[str(count)] = {key: chosen[key] for key in ('collision_weight', 'physical_overlap_pairs', 'violating_pairs', 'result_file')}
    cp_results=[row for row in rows if not row['collision_filter_enabled'] and row['cp_success']]
    overall=None
    if cp_results:
        overall=min(cp_results,key=lambda row:(row['physical_overlap_pairs'],row['violating_pairs'],row['violating_pair_intervals'],row['collision_weight']))
        selected=json.loads((args.root/overall['result_file']).read_text())
        config=dict(selected['config'])
        required=config.pop('required_robustness')
        config.update(adversary_bound=(required-1)//2,collision_filter=True)
        scenario=dict(description='Best tested collision weight on the saved 50-robot geometry; hard CP and continuous collision filters enabled; goals remain soft.',
            horizon_steps=selected['parameters']['steps'],start=selected['start'],goal=selected['goal'],config=config)
        (args.root/'best_weight_scenario.json').write_text(json.dumps(scenario,indent=2)+'\n')
        (args.root/'best_result_reference.json').write_text(json.dumps(dict(summary=overall,parameters=selected['parameters'],
            note='CP-only selected result. Consult the matching joint-filter result before treating it as accepted.'),indent=2)+'\n')
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/'comparison.json').write_text(json.dumps(rows, indent=2)+'\n')
    (args.root/'best_weights.json').write_text(json.dumps(best, indent=2)+'\n')
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (args.root/'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# Collision weights and hard-filter comparison — 11/11', '',
        'Fixed setup: 50 planar double-integrator robots, GNN–CP–autodiff, required r = 21, '
        '320 future steps, dt = 0.05 s (16 s), 5 optimization updates, seed 42, acceleration noise '
        '0.1 m/s², R = 3 m, acceleration norm ≤ 1 m/s². Starts and independently assigned goals '
        'are copied exactly from the saved 11/09 experiment. Goal arrival remains a soft cost. '
        'All cost settings except collision weight are fixed.', '',
        'Robot radius is 0.20 m: a physical overlap means center separation < 0.40 m. '
        'The hard filter requires ≥ 0.45 m, including motion between waypoints, using the '
        'constant-acceleration dynamics. Counts below are distinct robot pairs that violate '
        'the threshold at any time; they are not numbers of frames or simultaneous collisions.', '',
        'CP-only acceptance is explicitly enabled only for part 1 of this experiment. '
        'Part 2 uses the default joint CP + collision filter. CP-selected means the selected '
        'trajectory meets the predicted hard-graph CP threshold at every sampled future waypoint. '
        'This is not an unconditional robustness guarantee under adaptive trajectory search.', '',
        'Best weight is ranked by fewest physical-overlap pairs, then fewest 0.45 m violating '
        'pairs, then fewest violating pair/control intervals. Ties prefer the smaller weight.', '',
        '| Particles | Final filter | Collision weight | CP-selected | CP-pass candidates | Joint accepted | Overlap pairs (<0.40 m) | Clearance pairs (<0.45 m) | Min distance (m) | Goal error (m) | Solve (s) |',
        '|---:|:---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        def val(key, digits=None):
            value = row.get(key)
            return '—' if value is None else (f'{value:.{digits}f}' if digits is not None else str(value))
        joint = val('accepted_candidates') if row['collision_filter_enabled'] else 'not checked'
        lines.append(f"| {row['particles']} | {'CP + collision' if row['collision_filter_enabled'] else 'CP only'} | "
                     f"{row['collision_weight']:g} | {'YES' if row['cp_success'] else 'no trajectory'} | "
                     f"{val('cp_passing_candidates')} | {joint} | {val('physical_overlap_pairs')} | "
                     f"{val('violating_pairs')} | {val('minimum_continuous_separation_m', 5)} | "
                     f"{val('goal_mean_distance_m', 5)} | {val('solving_seconds', 1)} |")
    lines += ['', '## Initial candidate ranking', '',
              'For fixed initial candidates the objective is Bᵢ + weight × Aᵢ. '
              'The following ranges identify every lowest-cost CP-passing initial candidate '
              'as the nonnegative collision weight varies. This analysis does not cover '
              'the new trajectories generated by optimization at untested weights.', '',
              '| Particles | Weight range (approximately) | Particle index (zero-based) | Overlap pairs | Clearance pairs |',
              '|---:|:---|---:|---:|---:|']
    for path in sorted(args.root.glob('particles_*/cp_only/initial_weight_frontier.json')):
        frontier=json.loads(path.read_text())
        count=frontier['parameters']['particles']
        for segment in frontier['segments']:
            upper=segment['weight_upper']
            interval=f"{segment['weight_lower']:.2f} to {'∞' if upper is None else f'{upper:.2f}'}"
            lines.append(f"| {count} | {interval} | {segment['particle']} | {segment['physical_overlap_pairs']} | {segment['violating_pairs']} |")
        initial_pass=sum(bound>=21 for bound in frontier['minimum_bounds_by_particle'])
        matching=[row for row in rows if row['particles']==count and not row['collision_filter_enabled']]
        if matching and all(row['cp_passing_candidates']==initial_pass for row in matching):
            lines += ['', f'For {count} particles, {initial_pass} initial candidates pass CP. '
                      f'Every completed full trial at this population reports exactly {initial_pass} '
                      f'CP-passing candidates in total, so none of its {count*5} updated candidates '
                      'passes CP. The selected CP-only incumbent is an initial trajectory.', '']
    if population_rows:
        lines += ['', '## Acceptance by optimization population', '',
                  'Population 0 is initialization; populations 1–5 follow each update. '
                  'These counts come directly from the instrumented runs.', '',
                  '| Particles | Weight | CP-pass counts, populations 0–5 | Joint-pass counts, populations 0–5 |',
                  '|---:|---:|:---|:---|']
        for count,weight,populations in sorted(population_rows):
            cp_counts=', '.join(str(item['cp_passing_candidates']) for item in populations)
            joint_counts=', '.join(str(item['accepted_candidates']) for item in populations)
            lines.append(f'| {count} | {weight:g} | {cp_counts} | {joint_counts} |')
    lines += ['', 'No-feasible results have no selected trajectory, so collision and goal metrics are '
              'undefined (—), not zero. CP-pass and rejection counters include initial candidates and '
              'all updated populations. Solve time covers optimization and final filtering; checkpoint '
              'setup, post-run metrics, report generation and disk output are excluded. '
              'Rows marked `shared_population_trial` in the CSV use one optimizer run for both '
              'acceptance rules: the final filter never changes particle updates. Their solve '
              'times are the same measured run, including hard collision screening, not two '
              'independently timed runs. A regression test compares both selections against '
              'separate optimization runs. Sequential trials may share compilation caches; '
              'times are observed solve times rather than isolated timing benchmarks.', '',
              '## Best tested weights', '', '```json', json.dumps(best, indent=2), '```', '',
              '## Reproduce', '', 'Run from the `jax_GNN` directory with CUDA available. Replace '
              '`50` with `100` for the other population. Use separate output folders for each policy.', '',
              '```bash', 'JAX_PLATFORMS=cuda .venv/bin/python -m planning.tune_collision_weight \\\n  --particles 50 --legacy-cp-only --weights 1000 6900 30000 100000 \\\n  --output reports/trials/11-11/large_particle_collision_trials/particles_50/cp_only', '```', '',
              'For the hard filter, omit `--legacy-cp-only`, use `--weights 10 BEST_WEIGHT`, and '
              'write to `particles_50/joint_filter`. Weight 10 is automatically included. '
              'Each `weight_*.json` contains the full configuration, geometry, parameters, selected '
              'trajectory (or null), controls, velocities and acceptance report.', '',
              'To evaluate both rules on exactly the same candidate populations in one run:', '',
              '```bash',
              'JAX_PLATFORMS=cuda .venv/bin/python -m planning.tune_collision_weight --particles 50 --weights 6900 30000 100000 --output reports/trials/11-11/large_particle_collision_trials/particles_50/joint_filter --compare-cp-only-output reports/trials/11-11/large_particle_collision_trials/particles_50/cp_only',
              '```']
    if overall is not None:
        lines += ['', '## Best selected result and runnable scenario', '',
                  f"The best completed CP-only result has {overall['physical_overlap_pairs']} physical-overlap pairs "
                  f"at weight {overall['collision_weight']:g} with {overall['particles']} particles. "
                  'This is the best tested selection, not a global optimum. '
                  '`best_result_reference.json` points to its full saved trajectory and parameters. '
                  '`best_weight_scenario.json` preserves the exact geometry and uses this weight '
                  'with the hard collision filter enabled. It therefore returns no feasible '
                  'trajectory if no candidate satisfies both filters; it never falls back '
                  'to the colliding CP-only result.', '', '```bash',
                  f"JAX_PLATFORMS=cuda .venv/bin/python -m planning --scenario {args.root}/best_weight_scenario.json --robots 50 --particles {overall['particles']} --steps 320 --iterations 5 --methods gnn_cp_autodiff --output {args.root}/best_weight_joint_rerun.json",
                  '```']
        historical=Path(__file__).resolve().parent.parent/'reports/trials/11-09/collision_weights_r21/summary.json'
        if historical.exists():
            old_rows=[row for row in json.loads(historical.read_text()) if row.get('cp_success') and 'physical_overlap_pairs' in row]
            if old_rows:
                old=min(old_rows,key=lambda row:(row['physical_overlap_pairs'],row['violating_pairs'],row['violating_pair_intervals'],row['collision_weight']))
                lines += ['', '## Previous 15-particle comparison', '',
                          f"The earlier 15-particle CP-only sweep on the same geometry selected weight {old['collision_weight']:g} "
                          f"with {old['physical_overlap_pairs']} physical-overlap pairs and {old['violating_pairs']} "
                          'clearance-violating pairs. That trajectory was also unsafe. This historical '
                          'result is separate from the new 50/100-particle trials. A larger population '
                          'can select a trajectory with more distinct overlapping pairs because '
                          'the objective minimizes weighted penalty severity and other costs, rather '
                          'than the count of distinct overlapping pairs.']
    if (args.root/'resource_retry.json').exists():
        lines += ['', '## Resource handling', '',
                  'The first 100-particle optimization was killed with exit code 137 before '
                  'saving a feasibility result. It is not counted as a no-feasible outcome. '
                  'Memory pressure was suspected, but kernel OOM logs were unavailable. '
                  'The retry evaluates base costs/gradients in batches of eight particles '
                  'and constructs CP graphs in chunks of approximately 256, aligned with '
                  'the checkpoint minibatch size. It retains all 100 particles, all 320 '
                  'steps and five updates. Tests check costs, gradients, CP bound values '
                  'and ordering against unchunked calculations. The 50-particle runs '
                  'precede this scheduling change, so population timings also reflect '
                  'different memory scheduling. See `resource_retry.json` for details.', '',
                  'The 100-particle trials test weights 10, 1,000, 10,000 and 100,000: '
                  'the default, two intermediate values, and the best full 50-particle '
                  'weight. The intermediate values test whether updated candidates '
                  'improve on the initial-candidate ranking. Reproduce them with:', '',
                  '```bash',
                  'JAX_PLATFORMS=cuda .venv/bin/python -m planning.tune_collision_weight --particles 100 --weights 1000 10000 100000 --progress --output reports/trials/11-11/large_particle_collision_trials/particles_100/joint_filter --compare-cp-only-output reports/trials/11-11/large_particle_collision_trials/particles_100/cp_only',
                  '```', '',
                  'These finite trials do not establish that the scenario is infeasible. '
                  'A no-feasible result means the candidates generated at that tested '
                  'weight and optimization budget contain no trajectory that passes both filters.']
    (args.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(best, indent=2))


if __name__ == '__main__':
    main()
