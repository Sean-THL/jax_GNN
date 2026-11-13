"""Report collision-weight sweeps for explicit robot counts and r targets."""
import argparse
import csv
import json
from pathlib import Path


def write_csv(path,rows):
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('reports/trials/11-12/robustness_collision_trials'))
    parser.add_argument('--dynamics',choices=('double_integrator','single_integrator'),default='double_integrator')
    args=parser.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    rows=[];payloads={}
    for path in sorted(root.glob('robots_*_r*/particles_*/*/weight_*.json')):
        payload=json.loads(path.read_text());row=dict(payload['summary'])
        if payload['config']['dynamics']!=args.dynamics:
            raise ValueError('Report folder contains a different dynamics model')
        row.update(robots=len(payload['start']),required_robustness=payload['config']['required_robustness'],
                   evaluated_candidates=payload['planner_report']['evaluated_candidates'],result_file=str(path.relative_to(root)))
        row['possible_robot_pairs']=row['robots']*(row['robots']-1)//2
        row['physical_overlap_pair_fraction']=(row['physical_overlap_pairs']/row['possible_robot_pairs']
                                               if 'physical_overlap_pairs' in row else None)
        rows.append(row);payloads[row['result_file']]=payload
    rows.sort(key=lambda row:(-row['robots'],row['required_robustness'],row['particles'],row['collision_weight'],row['collision_filter_enabled']))
    (root/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n');write_csv(root/'comparison.csv',rows)
    groups=sorted({(row['robots'],row['required_robustness'],row['particles']) for row in rows},key=lambda key:(-key[0],key[1],key[2]))
    best={}
    for key in groups:
        matching=[row for row in rows if (row['robots'],row['required_robustness'],row['particles'])==key]
        cp=[row for row in matching if not row['collision_filter_enabled'] and row['cp_success']]
        joint=[row for row in matching if row['collision_filter_enabled'] and row['cp_success']]
        best[f'robots_{key[0]}_r{key[1]}_particles_{key[2]}']={
            'cp_only':min(cp,key=lambda row:(row['physical_overlap_pairs'],row['violating_pairs'],row['violating_pair_intervals'],row['collision_weight'])) if cp else None,
            'joint_filter':min(joint,key=lambda row:(row['goal_mean_distance_m'],row['collision_weight'])) if joint else None}
    (root/'best_results.json').write_text(json.dumps(best,indent=2)+'\n')
    def fmt(row,key,digits=None):
        value=row.get(key)
        if value is None:return '—'
        return f'{value:.{digits}f}' if digits is not None else str(value)
    single=args.dynamics=='single_integrator'
    description=('Three setups: 50 robots with required r = 21 or 20, and 25 robots with required r = 10. '
                 if single else 'Two setups: 50 robots with required r = 20, and 25 robots with required r = 10. ')
    model=('single integrator p[k+1] = p[k] + dt*u[k], with velocity command u hard bounded '
           'to norm ≤ 3 m/s; position is the state and there is no acceleration bound'
           if single else 'double integrator with acceleration norm ≤ 1 m/s² and soft speed bound 3 m/s')
    lines=[f'# {"Single-integrator" if single else "Robustness targets"} collision-weight trials — 11/12','',
           description+
           'Each uses 50 and 100 particles. The 50-robot coordinates and assigned goals are '
           'copied from the saved 11/09 run. The 25-robot case uses its first 25 robot/goal '
           'pairs, without resampling. Targets are passed directly as integers to the '
           'tuning tool; they are not rounded through r = 2F + 1. The scenario loader’s '
           'default F-based behavior remains unchanged.','',
           f'Fixed: GNN–CP–autodiff, planar {model}, 320 future steps, dt = 0.05 s '
           f'(16 s), five updates, seed 42, {"velocity noise 0.1 m/s" if single else "acceleration noise 0.1 m/s²"}, R = 3 m. Goal arrival remains a soft cost. CP '
           'weight = 5, degree weight = 2, goal weight = 10, SVGD step size = 0.01. '
           'Only robot count, robustness target, particle count and collision weight vary.','',
           'Robot radius = 0.20 m. Physical overlap means center separation < 0.40 m. '
           'The hard filter requires ≥ 0.45 m throughout each constant-acceleration '
           'interval, including between waypoints. Pair counts are distinct robot pairs '
           'that violate the threshold at any time, not numbers of frames. '
           'There are 1,225 possible pairs for 50 robots and 300 for 25 robots; raw '
           'pair counts across team sizes therefore have different denominators. CP bounds '
           'are predicted on hard radius graphs at every sampled future waypoint. '
           'This is not an unconditional robustness guarantee under adaptive search.','',
           '## Best CP-only weight in each setup','',
           'Ranked by physical-overlap pairs, then clearance-violating pairs, then violating '
           'pair/control intervals. These are the best tested selections, not global optima.','',
           '| Robots | Required r | Particles | Weight | CP bound | Overlap pairs | Clearance pairs | Goal error (m) | Solve (s) |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in groups:
        entry=best[f'robots_{key[0]}_r{key[1]}_particles_{key[2]}']['cp_only']
        if entry is None:
            lines.append(f'| {key[0]} | {key[1]} | {key[2]} | no CP-feasible trajectory | — | — | — | — | — |')
        else:
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {entry['collision_weight']:g} | {fmt(entry,'minimum_cp_bound')} | {fmt(entry,'physical_overlap_pairs')} | {fmt(entry,'violating_pairs')} | {fmt(entry,'goal_mean_distance_m',4)} | {fmt(entry,'solving_seconds',1)} |")
    lines += ['', '## Hard-filter comparison at default and best CP-only weights','',
              'CP and joint pass counts include initialization plus all five updated populations. '
              'The joint filter checks every CP-passing candidate and chooses the lowest-cost '
              'candidate that also meets clearance. A no-feasible result has no selected '
              'trajectory; missing metrics below are undefined, not zero.','',
              '| Robots | Required r | Particles | Weight | Status | CP passing | Joint accepted | Selected CP bound | Min separation (m) | Goal error (m) |',
              '|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|']
    for key in groups:
        cp_best=best[f'robots_{key[0]}_r{key[1]}_particles_{key[2]}']['cp_only']
        weights={10.} if cp_best is None else {10.,cp_best['collision_weight']}
        joint_best=best[f'robots_{key[0]}_r{key[1]}_particles_{key[2]}']['joint_filter']
        if joint_best is not None:weights.add(joint_best['collision_weight'])
        for weight in sorted(weights):
            match=next((row for row in rows if (row['robots'],row['required_robustness'],row['particles'])==key and row['collision_filter_enabled'] and row['collision_weight']==weight),None)
            if match is None:continue
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {weight:g} | {match['status']} | {fmt(match,'cp_passing_candidates')} | {fmt(match,'accepted_candidates')} | {fmt(match,'minimum_cp_bound')} | {fmt(match,'minimum_continuous_separation_m',5)} | {fmt(match,'goal_mean_distance_m',4)} |")
    lines += ['', '## Every completed CP-only weight trial','',
              '| Robots | Required r | Particles | Weight | CP bound | Overlap pairs | Clearance pairs | Goal error (m) | Solve (s) |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['collision_filter_enabled']:continue
        lines.append(f"| {row['robots']} | {row['required_robustness']} | {row['particles']} | {row['collision_weight']:g} | {fmt(row,'minimum_cp_bound')} | {fmt(row,'physical_overlap_pairs')} | {fmt(row,'violating_pairs')} | {fmt(row,'goal_mean_distance_m',4)} | {fmt(row,'solving_seconds',1)} |")
    lines += ['', '## Reproduce','',
              'Run from `jax_GNN`. The runner resumes existing matching results and runs '
              f'the {"six" if single else "four"} setups sequentially on CUDA. Separate folders prevent mixing targets '
              'or populations. Add a new `--root` to recompute instead of resuming.','',
              '```bash',f'JAX_PLATFORMS=cuda .venv/bin/python -m planning.{"run_single_integrator_trials" if single else "run_robustness_collision_trials"}','```','',
              'For one setup, use the exact even-valued target in the trial CLI:','',
              '```bash','JAX_PLATFORMS=cuda .venv/bin/python -m planning.tune_collision_weight \\\n+  --robots 25 --required-robustness 10 --particles 100 \\\n+  --weights 10 1000 10000 100000 --progress \\\n+  --output reports/trials/11-12/robustness_collision_trials/robots_25_r10/particles_100/joint_filter \\\n+  --compare-cp-only-output reports/trials/11-12/robustness_collision_trials/robots_25_r10/particles_100/cp_only','```','',
              'CP-only and joint selections share the exact same optimized populations because '
              'final acceptance never feeds back into the gradient. Their solve times refer '
              'to one measured run, including collision screening. Scorer setup, post-run '
              'metrics and file output are excluded. Runs may share compilation caches. '
              'Memory batching retains every particle and waypoint.','',
              '`comparison.csv` and `comparison.json` include both policies for every completed '
              'weight. `best_results.json` points to the full saved trajectories, controls, '
              'velocities and reports. For a group with joint successes it also records the '
              'accepted result with the smallest goal error; weighted objective totals are '
              'not compared across different collision weights. `manifest.json` records '
              'commands and completion state. A no-feasible outcome applies only to the '
              'tested candidates and budget, not to every possible trajectory.']
    if (root/'adaptive_manifest.json').exists():
        lines += ['', '## Intermediate-weight follow-up', '',
                  'The 25-robot runs showed a gap between weight 10, whose updated '
                  'candidates passed CP, and weight 1,000, which lost CP feasibility '
                  'at 50 particles. Weight 100 was added for both 50 and 100 particles '
                  'to test this middle range. Its commands and completion state are '
                  'recorded separately in `adaptive_manifest.json`.', '', '```bash',
                  'JAX_PLATFORMS=cuda .venv/bin/python -m planning.run_robustness_collision_trials --cases 25:10:50 25:10:100 --weights 100 --manifest-name adaptive_manifest.json',
                  '```']
    if (root/'svgd_memory_validation.json').exists():
        memory=json.loads((root/'svgd_memory_validation.json').read_text())
        lines += ['', '## Memory handling', '',
                  'The first 100-particle run was killed before saving a feasibility result. '
                  'The SVGD pair-distance calculation materialized a full particle-pair-by-control '
                  'tensor and its square. It now computes the same direct squared distances '
                  'one upper-triangle row at a time. Reference tests check the resulting '
                  'SVGD directions, including nearly identical particles with large offsets.', '',
                  f"For 100 particles × 320 steps × 50 robots × 2 controls, the pair-distance "
                  f"temporary estimate drops from {memory['old_difference_and_square_temporary_gib']:.2f} GiB "
                  f"to {memory['new_largest_difference_and_square_temporary_mib']:.1f} MiB. "
                  f"A standalone full-size SVGD update completed in {memory['seconds']:.3f} s "
                  f"with process peak RSS {memory['peak_process_rss_mib']:.1f} MiB. This "
                  'standalone CPU check excludes the GNN and is not the full planning time. '
                  'All requested particles and planning settings are retained. The completed '
                  '50-particle first sweep precedes this memory change; subsequent sweeps '
                  'use it. No killed run is counted as a no-feasible trajectory.']
    if single:
        lines=[line.replace('planning.tune_collision_weight \\', 'planning.tune_collision_weight --dynamics single_integrator \\')
                    .replace('reports/trials/11-12/robustness_collision_trials/','reports/trials/11-12/single_integrator_trials/')
                    .replace('--weights 10 1000 10000 100000','--weights 10 100 1000 10000 100000 --cache-hard-batches') for line in lines]
        lines += single_comparison(root,rows,groups)
        if (root/'hard_batch_cache_validation.json').exists():
            lines += ['', '## Exact hard-batch prediction reuse', '',
                      'After the first five-weight SI sweep, the remaining runs enable '
                      '`--cache-hard-batches`. The frozen CP scorer reuses predictions only '
                      'when the entire ordered binary-graph minibatch is identical, including '
                      'node count, batch size, CP method and quantile. Minibatches are not '
                      'reordered or regrouped. The cache holds at most 4,096 batches. Smooth '
                      'weights, gradients and trajectory updates are never cached. CUDA '
                      'validation compares all CP outputs with uncached inference on 195 '
                      'diverse 50-node graphs and 195 diverse 25-node graphs, including '
                      'partial minibatches; all bounds match exactly. See '
                      '`hard_batch_cache_validation.json`.','',
                      'Timing comparisons also reflect this inference optimization and '
                      'compilation reuse. Historical DI timings and the first SI sweep '
                      'are uncached; subsequent SI sweeps use exact-batch reuse. These '
                      'are observed solve times, not a benchmark isolating dynamics. '
                      'Per-result reports record cache status and cumulative scorer hit/miss counts.']
        if (root/'validation.json').exists():
            validation=json.loads((root/'validation.json').read_text())
            lines += ['', '## Saved-result validation', '',
                      f"Independent validation completed for {validation['configurations']} configurations "
                      f"and {validation['policy_results']} policy results ({validation['candidate_evaluations']:,} "
                      'candidate evaluations). It checks the saved SI rollout equation, velocity '
                      'bound, exact robustness targets, soft-goal settings, CP bounds at all '
                      'waypoints and continuous linear collision metrics. Details are in '
                      '`validation.json`. Re-run after a completed matrix with:', '', '```bash',
                      '.venv/bin/python -m planning.validate_single_integrator_trials','```']
        if (root/'diagnostics/cli_smoke.json').exists():
            lines += ['', 'The public CLI also completed a CUDA smoke test for all four cost '
                      'methods (simple, soft_degree, gnn_cp_spsa, gnn_cp_autodiff). This uses '
                      'two particles, two steps and one update to check model dispatch and '
                      'output units; it is separate from the full GNN–CP–autodiff weight '
                      'matrix. See `diagnostics/cli_smoke.json`.']
    (root/'REPORT.md').write_text('\n'.join(lines).replace('\n+  --','\n  --')+'\n')
    print(json.dumps({key:{policy:None if row is None else dict(weight=row['collision_weight'],overlap_pairs=row.get('physical_overlap_pairs'),goal_error_m=row.get('goal_mean_distance_m'),status=row['status']) for policy,row in entry.items()} for key,entry in best.items()},indent=2),flush=True)


def single_comparison(root,rows,groups):
    from planning.run_single_integrator_trials import reference_trials
    references=reference_trials();matched=[]
    lookup={(row['robots'],row['required_robustness'],row['particles'],row['collision_weight'],row['collision_filter_enabled']):row for row in rows}
    for key,row in sorted(lookup.items()):
        if key[-1]:continue
        reference=references.get(key)
        if reference is None:continue
        path,payload=reference;old=payload['summary'];joint_key=(*key[:-1],True)
        old_joint=references.get(joint_key);new_joint=lookup.get(joint_key)
        matched.append(dict(robots=key[0],required_robustness=key[1],particles=key[2],collision_weight=key[3],
            di_cp_success=old['cp_success'],si_cp_success=row['cp_success'],
            di_overlap_pairs=old.get('physical_overlap_pairs'),si_overlap_pairs=row.get('physical_overlap_pairs'),
            di_goal_error_m=old.get('goal_mean_distance_m'),si_goal_error_m=row.get('goal_mean_distance_m'),
            di_solving_seconds=old['solving_seconds'],si_solving_seconds=row['solving_seconds'],
            di_joint_status=old_joint[1]['summary']['status'] if old_joint else 'not run',
            si_joint_status=new_joint['status'] if new_joint else 'not run',
            double_integrator_result=str(path),single_integrator_result=row['result_file']))
    (root/'matched_dynamics_comparison.json').write_text(json.dumps(matched,indent=2)+'\n')
    write_csv(root/'matched_dynamics_comparison.csv',matched)
    lines=['','## Matched double-integrator comparison','',
           'The runner repeats the exact weight/population combinations from the prior '
           '11/11 and 11/12 double-integrator sweeps. Matched starts and goals, horizon, '
           'dt, updates, seed, R and collision thresholds are preserved. The numeric '
           'control noise remains 0.1, but its units change from acceleration to velocity. '
           'SI controls have a hard 3 m/s speed limit; DI has a hard 1 m/s² acceleration '
           'limit and soft 3 m/s speed cost. The initialization is minimum-energy '
           'for each model, targeting arrival and a stopped final interval without '
           'fixing optimized endpoints. The cost keeps the goal, path, collision, '
           'speed, degree, GNN surrogate, terminal-velocity and control-effort terms; '
           'the control-effort units necessarily differ. These differences are part '
           'of the model comparison, so results should not be attributed solely '
           'to replacing one integration equation.','',
           '| Robots | Required r | Particles | DI best weight | DI overlap pairs | SI best weight | SI overlap pairs |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        matching=[row for row in matched if (row['robots'],row['required_robustness'],row['particles'])==group]
        old=[row for row in matching if row['di_cp_success'] and row['di_overlap_pairs'] is not None]
        new=[row for row in matching if row['si_cp_success'] and row['si_overlap_pairs'] is not None]
        if old and new:
            a=min(old,key=lambda row:(row['di_overlap_pairs'],row['collision_weight']))
            b=min(new,key=lambda row:(row['si_overlap_pairs'],row['collision_weight']))
            lines.append(f"| {group[0]} | {group[1]} | {group[2]} | {a['collision_weight']:g} | {a['di_overlap_pairs']} | {b['collision_weight']:g} | {b['si_overlap_pairs']} |")
    lines += ['', 'The comparison CSV contains each matched weight separately. A missing '
              'historical hard-filter run is marked `not run`; it is not inferred. '
              'For optimized SI paths, collision checks minimize pair separation '
              'along each straight segment. DI paths use quadratic constant-acceleration '
              'segments. Both include the start state. SI saved controls and velocities '
              'are the same held interval commands in m/s; SI reports contain no '
              'acceleration array.']
    return lines


if __name__=='__main__':main()
