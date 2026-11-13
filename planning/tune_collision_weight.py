"""Controlled collision-weight trials from an existing saved run."""
import argparse,csv,json,logging
from dataclasses import fields,replace,asdict
from pathlib import Path
from time import perf_counter
import numpy as np
from planning.collision_metrics import continuous_separation,linear_separation
from planning.dynamics import initial_controls,rollout,rollout_single_integrator
from planning.run_experiment import PROJECT,trajectory_summary
from planning.svgd_cp_planner import PlannerConfig,ConformalGraphScorer,plan,cp_cost
from planning.double_integrator_planner import plan_double_integrator
from planning.single_integrator_planner import plan_single_integrator

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=PROJECT/'reports/filtered_results/11-09/100robots_r21_four_methods.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--legacy-cp-only',action='store_true',help='Explicitly reproduce historical CP-only weight trials')
    parser.add_argument('--weights',type=float,nargs='+',default=[3,30,100,300,1000,10000])
    parser.add_argument('--particles',type=int,help='Override the saved population size; keep geometry, horizon, noise and seed fixed')
    parser.add_argument('--robots',type=int,help='Use the first N robots and their assigned goals from the saved geometry')
    parser.add_argument('--required-robustness',type=int,help='Explicit experimental r target (including even values); overrides the saved target')
    parser.add_argument('--dynamics',choices=('double_integrator','single_integrator'),help='Control dynamics; default preserves the saved model')
    parser.add_argument('--compare-cp-only-output',type=Path,help='Also save CP-only selection from the exact same optimized populations, without rerunning the GNN')
    parser.add_argument('--inspect-initial',action='store_true',help='Record initial-population hard CP bounds')
    parser.add_argument('--progress',action='store_true',help='Log optimization stages and per-population acceptance')
    parser.add_argument('--cache-hard-batches',action='store_true',help='Reuse frozen-model CP predictions only for identical ordered hard-graph minibatches')
    parser.add_argument('--robot-radius',type=float,default=0.2,help='Physical disk radius for overlap reporting')
    args=parser.parse_args()
    if args.progress:logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    if args.compare_cp_only_output is not None and args.legacy_cp_only:
        parser.error('shared comparison requires the primary hard collision filter')
    if args.compare_cp_only_output is not None:
        args.compare_cp_only_output.mkdir(parents=True,exist_ok=True)
    if args.output is None:
        args.output=PROJECT/('reports/trials/11-09/collision_weights_r21' if args.legacy_cp_only else 'reports/trials/11-11/collision_weights_joint_filter')
    if args.compare_cp_only_output is not None and args.compare_cp_only_output.resolve()==args.output.resolve():
        parser.error('CP-only and joint-filter results require separate output folders')
    args.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.source.read_text());baseline=source['results']['gnn_cp_autodiff']
    values=baseline.get('planner_config',source['config']);keys={f.name for f in fields(PlannerConfig)}
    source_config={k:v for k,v in values.items() if k in keys}
    source_config['collision_filter']=not args.legacy_cp_only
    config=PlannerConfig(**source_config)
    if not np.isfinite(args.robot_radius) or args.robot_radius<=0 or 2*args.robot_radius>config.minimum_separation:
        parser.error('robot radius must be positive and no greater than half minimum_separation')
    config=replace(config,robot_radius=args.robot_radius)
    if args.dynamics is not None:config=replace(config,dynamics=args.dynamics)
    if args.required_robustness is not None:
        if not 0 <= args.required_robustness < 26:parser.error('required robustness must be between 0 and 25')
        config=replace(config,required_robustness=args.required_robustness)
    start,goal=np.array(source['start'],dtype=np.float32),np.array(source['goal'],dtype=np.float32)
    params=dict(source['parameters'])
    if config.dynamics=='single_integrator':
        params['noise_units']='m/s'
        params['noise_fraction']=params['noise']/config.maximum_speed
    if args.robots is not None:
        if not 2 <= args.robots <= len(start):parser.error('robots must be between two and the saved robot count')
        start,goal=start[:args.robots],goal[:args.robots]
        params['robots']=args.robots
    if config.required_robustness > (len(start)+1)//2:
        parser.error('required robustness exceeds the complete-graph maximum for this robot count')
    if args.particles is not None:
        if args.particles < 2:parser.error('particles must be at least two')
        params['particles']=args.particles
    def filename(weight):return args.output/f'weight_{weight:g}.json'
    def update_summary(output):
        rows=[json.loads(p.read_text())['summary'] for p in output.glob('weight_*.json')]
        rows.sort(key=lambda r:r['collision_weight']);columns=list(dict.fromkeys(k for row in rows for k in row))
        with (output/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    def save(weight,waypoints,report,metrics,elapsed,origin,result_config=None,output=None):
        result_config=config if result_config is None else result_config
        output=args.output if output is None else output
        collision={}
        if waypoints is not None:
            collision=(linear_separation(start,waypoints,result_config.minimum_separation,robot_radius=result_config.robot_radius)
                if result_config.dynamics=='single_integrator' else
                continuous_separation(start,waypoints,report['velocities'],report['accelerations'],result_config.sampling_time,result_config.minimum_separation,result_config.initial_velocity,robot_radius=result_config.robot_radius))
        row={'collision_weight':float(weight),'cp_success':bool(metrics['meets_predicted_target']),
             'particles':params['particles'],'status':report['status'],
             'robots':len(start),'required_robustness':result_config.required_robustness,
             'dynamics':result_config.dynamics,
             'cp_passing_candidates':report.get('cp_passing_candidates',report.get('passing_candidates')),
             'accepted_candidates':report.get('passing_candidates'),
             'collision_rejected_candidates':report.get('collision_rejected_candidates',0),
             'collision_filter_enabled':result_config.collision_filter,
             'minimum_cp_bound':metrics['minimum_lower_bound'],**collision,'goal_mean_distance_m':metrics['goal_mean_distance'],
             'maximum_acceleration_m_s2':report.get('maximum_acceleration_norm'),'maximum_speed_m_s':report.get('maximum_velocity_norm'),
             'selected_population':report.get('selected_population'),'solving_seconds':float(elapsed),'origin':origin}
        payload={'summary':row,'config':asdict(replace(result_config,collision_weight=weight)),
                 'parameters':params,'source':str(args.source),'start':start.tolist(),'goal':goal.tolist(),
                 'checkpoint':source['checkpoint'],'calibration':source['calibration'],
                 'waypoints':None if waypoints is None else np.asarray(waypoints).tolist(),
                 'planner_report':report,'metrics':metrics}
        (output/f'weight_{weight:g}.json').write_text(json.dumps(payload,indent=2)+'\n');update_summary(output);print(json.dumps(row),flush=True)
    base_weight=config.collision_weight
    baseline_mode=baseline['planner_report'].get('collision_filter_enabled',False)
    if not filename(base_weight).exists() and baseline_mode==config.collision_filter and params==source['parameters'] and args.compare_cp_only_output is None and args.required_robustness is None and args.dynamics is None:
        save(base_weight,baseline['waypoints'],baseline['planner_report'],baseline,baseline['wall_seconds'],'existing_saved_baseline')
    requested=list(dict.fromkeys([base_weight]+args.weights))
    for weight in requested:
        targets=[(filename(weight),config.collision_filter)]
        if args.compare_cp_only_output is not None:
            targets.append((args.compare_cp_only_output/f'weight_{weight:g}.json',False))
        for target,policy in targets:
            if target.exists():
                prior=json.loads(target.read_text())
                if prior['config'].get('collision_filter',False)!=policy:
                    parser.error('output folder contains a different collision-filter policy; choose a new --output')
                if prior['parameters']!=params or prior['start']!=start.tolist() or prior['goal']!=goal.tolist():
                    parser.error('output folder contains different experiment parameters or geometry; choose a new --output')
                expected=asdict(replace(config,collision_filter=policy,collision_weight=weight))
                if json.dumps(prior['config'],sort_keys=True)!=json.dumps(expected,sort_keys=True):
                    parser.error('output folder contains a different planner configuration; choose a new --output')
    pending=[w for w in requested if not filename(w).exists() or
             (args.compare_cp_only_output is not None and not (args.compare_cp_only_output/f'weight_{w:g}.json').exists())]
    if not pending and not args.inspect_initial:return
    scorer=ConformalGraphScorer(Path(source['checkpoint']),Path(source['calibration']),config.cp_coverage,config.cp_kind)
    scorer.cache_hard_batches=args.cache_hard_batches
    controls=initial_controls(start,goal,params['particles'],params['steps'],config,params['noise'],params['seed'])
    if args.inspect_initial:
        positions=np.asarray((rollout_single_integrator(controls,start,config.sampling_time) if config.dynamics=='single_integrator' else rollout(controls,start,np.asarray(config.initial_velocity),config.sampling_time))[0])
        _,lower=cp_cost(positions,scorer,config)
        initial={'minimum_bounds_by_particle':lower.tolist(),
                 'passing_count':int((lower>=config.required_robustness).sum()),
                 'required_robustness':config.required_robustness}
        (args.output/'initial_population_cp.json').write_text(json.dumps(initial,indent=2)+'\n')
        print(json.dumps(initial),flush=True)
    for weight in pending:
        if weight<0:raise ValueError('collision weights must be nonnegative')
        settings=replace(config,collision_weight=weight);print(f'Starting collision weight {weight:g}',flush=True)
        began=perf_counter()
        if args.compare_cp_only_output is not None:
            planner=plan_single_integrator if settings.dynamics=='single_integrator' else plan_double_integrator
            selected,report=planner(start,goal,controls,scorer,settings,params['iterations'],params['seed'],compare_cp_only=True)
        else:
            selected,report=plan(start,goal,controls,scorer,settings,params['iterations'],params['seed'])
        elapsed=perf_counter()-began
        report['hard_batch_cache']=({'enabled':True,'scope':'scorer_lifetime',
            'hits':scorer.hard_batch_cache_hits,'misses':scorer.hard_batch_cache_misses,
            'stored_batches':len(scorer.hard_batch_cache)} if args.cache_hard_batches else {'enabled':False})
        comparison=report.pop('cp_only_comparison',None)
        origin='shared_population_trial' if comparison is not None else 'controlled_trial'
        if comparison is not None:
            comparison['planner_report']['hard_batch_cache']=dict(report['hard_batch_cache'])
            report['timing_note']='Same measured solve shared by both acceptance policies; includes joint collision screening.'
            comparison['planner_report']['timing_note']=report['timing_note']
        metrics=trajectory_summary(selected,start,goal,scorer,settings,elapsed)
        save(weight,selected,report,metrics,elapsed,origin)
        if comparison is not None:
            cp_settings=replace(settings,collision_filter=False)
            cp_metrics=trajectory_summary(comparison['waypoints'],start,goal,scorer,cp_settings,elapsed)
            save(weight,comparison['waypoints'],comparison['planner_report'],cp_metrics,elapsed,origin,
                 result_config=cp_settings,output=args.compare_cp_only_output)
if __name__=='__main__':main()
