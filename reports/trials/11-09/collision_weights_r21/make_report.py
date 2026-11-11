import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gnn-planner-matplotlib')
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('jax_GNN/reports/trials/11-09/collision_weights_r21')
rows=json.loads((p/'summary.json').read_text());eligible=[r for r in rows if r['cp_success']]
best=min(eligible,key=lambda r:(r['violating_pairs'],r['violating_pair_intervals'],-r['minimum_continuous_separation_m'],r['collision_weight']))
ties=[r['collision_weight'] for r in eligible if r['violating_pairs']==best['violating_pairs']]
baseline=next(r for r in rows if r['collision_weight']==10)
trial=json.loads((p/f"weight_{best['collision_weight']:g}.json").read_text());c=dict(trial['config']);c.pop('required_robustness');c['adversary_bound']=10;c['robot_radius']=0.2
scenario={'description':'Best tested count only; NOT collision-free. Robot radius 0.2 m.', 'horizon_steps':320,
          'units':{'distance':'m','time':'s','input':'m/s^2'},'start':trial['start'],'goal':trial['goal'],'config':c}
(p/'best_tested_scenario.json').write_text(json.dumps(scenario,indent=2)+'\n')
old=p/'REPORT.md';archive=p/'REPORT_initial_14_weights.md'
if old.exists() and not archive.exists():archive.write_text(old.read_text())
lines=['# Extended collision-weight trials: 50 robots, r = 21','',
       f"**No tested weight eliminates collisions.** {len(rows)} tested weights cover zero to one million. "
       f"The lowest tested weight tied for the fewest violating pairs is **{best['collision_weight']:g}**. "
       'It does not improve the best count from the earlier search; it selects the same initial trajectory at a smaller weight.','',
       'The best group has 310 pairs violating the 0.45 m center-distance target, versus 328 at weight 10. '
       'With the user-defined body radius of 0.2 m, the best group has 270 pairs with actual disk overlap, versus 287 at the baseline. '
       'Its closest center distance is 0.002784 m and its minimum surface gap is -0.397216 m. These remain unsafe trajectories.','',
       'Weights tied at the best safety-violation count: '+', '.join(f'{w:g}' for w in ties)+'.','',
       '## Geometry and success definitions','',
       '- Robot body radius: 0.2 m. Body diameter: 0.4 m.',
       '- Physical body overlap: center distance below 0.4 m.',
       '- Safety-distance violation: center distance below 0.45 m; this includes a 0.05 m surface gap.',
       '- CP success: accepted predicted hard-graph lower bounds at least 21, with the existing degree/feasibility reporting checks.',
       '- CP success does not certify collision safety or exact graph robustness.',
       '- The body radius was specified after the trials. Body-overlap metrics were recomputed on the saved trajectories; the optimization separation target stays 0.45 m.','',
       'A violating pair is a unique robot pair that violates at least once. Pair-intervals count individual robot-pair/control intervals with a violation. '
       'For each constant-acceleration interval, relative position is quadratic. Minimum separation is evaluated at endpoints and real stationary points obtained from a cubic. '
       'These checks include the motion between stored waypoints. They do not simulate contact forces or impacts.','',
       '## Fixed experiment settings','',
       '50 robots, GNN–CP–autodiff, 15 particles, 320 steps, dt=0.05 s (16 seconds), five optimization updates, seed 42, '
       'the same saved starts/goals and initial control noise 0.1. R=3 m, r=21, hard acceleration norm <=1 m/s². '
       'Goals and collision separation remain soft objectives. Only collision weight changes. Source: '
       '`reports/filtered_results/11-09/100robots_r21_four_methods.json`; its historical filename says 100 but its saved parameters specify 50 robots.','',
       '## Results','',
       '| Weight | CP success | Min CP | Safety-violating pairs (<0.45 m) | Body-overlap pairs (<0.4 m) | Closest centers (m) | Mean goal error (m) | Solve time (s) |',
       '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in rows:
 lines.append(f"| {r['collision_weight']:g} | {'YES' if r['cp_success'] else 'NO'} | {r['minimum_cp_bound']} | {r['violating_pairs']} | {r['physical_overlap_pairs']} | {r['minimum_continuous_separation_m']:.6f} | {r['goal_mean_distance_m']:.8f} | {r['solving_seconds']:.3f} |")
lines+=['','Weight 10 uses the existing saved baseline; other weights are new CUDA trials. Timings include planner compilation, exclude setup and collision analysis, and are not a controlled runtime benchmark.','',
        '## Why further weight tuning plateaued','',
        'All selected plans came from initialization. Every run accepted exactly 15 out of 90 evaluated candidates. '
        '`initial_population_cp.json` confirms that all 15 initial candidates pass CP with bound 22, so none of the 75 updated candidates passed in any trial. '
        'The weight changes which initial candidate is selected rather than producing a better CP-accepted updated plan.','',
        'For fixed initial trajectories, each cost is affine in collision weight: `cost_i(w) = base_i + w * collision_penalty_i`. '
        '`initial_weight_frontier.json` identifies every initial-candidate selection region. Its predicted indices and costs match the saved trials. '
        'The bounds below are approximate floating-point transition values, not exact guarantees for the full optimizer.','',
        '| Weight range (approximately) | Selected initial particle | Safety-violating pairs |',
        '| --- | ---: | ---: |']
frontier=json.loads((p/'initial_weight_frontier.json').read_text())
for segment in frontier['segments']:
 upper='infinity' if segment['weight_upper'] is None else f"{segment['weight_upper']:.1f}"
 lines.append(f"| {segment['weight_lower']:.1f} to {upper} | {segment['particle']} | {segment['violating_pairs']} |")
lines+=['','No initial selection region is collision-free. This establishes the limitation of ranking these fixed initial candidates, '
        'not impossibility of finding a collision-free optimized trajectory at an untested weight or with other optimization settings. '
        'The squared-depth penalty also does not minimize collision count directly, explaining why very large weights increase the count.','',
        '## Reproduction','', 'From `jax_GNN/`:','', '```bash',
        'JAX_PLATFORMS=cuda .venv/bin/python -m planning.tune_collision_weight --legacy-cp-only \\',
        '  --weights 0 3 30 100 300 1000 2000 3000 5000 6900 7500 10000 15000 30000 31200 100000 1000000 \\',
        '  --robot-radius 0.2', '```','',
        'The baseline weight 10 is imported automatically. Existing trial files are reused. '
        'The default scenario collision weight remains 10. To reproduce the best tested setting without changing it:','',
        '```bash','JAX_PLATFORMS=cuda .venv/bin/python -m planning \\',
        '  --scenario reports/trials/11-09/collision_weights_r21/best_tested_scenario.json \\',
        '  --robots 50 --particles 15 --steps 320 --iterations 5 \\',
        '  --output reports/trials/11-09/collision_weights_r21/best_rerun.json','```','',
        '`summary.csv` and `summary.json` contain both body-overlap and safety-distance metrics. '
        'The main CLI also supports `--collision-weight 6900`. This is an experimental best count, not a safe recommended setting.']
old.write_text('\n'.join(lines)+'\n')
plot_rows=[r for r in rows if r['collision_weight']>0];x=[r['collision_weight'] for r in plot_rows]
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
for ax in axes:ax.set_xscale('log');ax.set_xlabel('Collision weight');ax.grid(alpha=.2)
axes[0].plot(x,[r['violating_pairs'] for r in plot_rows],'-o',label='Safety violations (<0.45 m)',color='#24678c')
axes[0].plot(x,[r['physical_overlap_pairs'] for r in plot_rows],'-s',label='Body overlaps (<0.4 m)',color='#a44942')
axes[0].set_ylabel('Unique robot pairs');axes[0].set_title('No tested weight eliminates collisions');axes[0].legend(fontsize=9)
axes[1].plot(x,[r['minimum_continuous_separation_m'] for r in plot_rows],'-o',color='#a44942')
axes[1].axhline(.4,ls='-',color='#444',lw=1,label='Body diameter: 0.4 m')
axes[1].axhline(.45,ls='--',color='#32804b',label='Safety target: 0.45 m')
axes[1].set_yscale('log');axes[1].set_ylabel('Minimum center distance (m)');axes[1].set_title('Closest approach remains far below body diameter');axes[1].legend(fontsize=9)
fig.suptitle(f'{len(rows)} weights · 50 robots · r=21 · radius=0.2 m · all trials pass CP (weight 0 omitted on log axis)',fontsize=11)
fig.savefig(p/'collision_weight_search.png',dpi=170);fig.savefig(p/'collision_weight_search.pdf');plt.close(fig)
print('Best tested',best['collision_weight'],'body overlap pairs',best['physical_overlap_pairs'])
