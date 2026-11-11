"""Find informative collision weights from the initial candidate cost frontier.

Initial trajectories are fixed, so J_i(w)=B_i+w*A_i. This analysis does not
replace optimization; it identifies candidate-selection intervals to test.
"""
import argparse,csv,json
from dataclasses import fields,replace
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from planning.run_experiment import PROJECT
from planning.dynamics import initial_controls,rollout
from planning.collision_metrics import continuous_separation
from planning.svgd_cp_planner import PlannerConfig,ConformalGraphScorer,_smooth_cost_single,make_smooth_cp_value_and_grad,cp_cost


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'reports/trials/11-09/collision_weights_r21')
    parser.add_argument('--source',type=Path,default=PROJECT/'reports/filtered_results/11-09/100robots_r21_four_methods.json')
    parser.add_argument('--particles',type=int,help='Override saved population size')
    args=parser.parse_args();source=json.loads(args.source.read_text());values=source['results']['gnn_cp_autodiff']['planner_config']
    config=PlannerConfig(**{k:v for k,v in values.items() if k in {f.name for f in fields(PlannerConfig)}})
    config=replace(config,robot_radius=.2 if config.robot_radius is None else config.robot_radius)
    params=dict(source['parameters'])
    if args.particles is not None:params['particles']=args.particles
    args.output.mkdir(parents=True,exist_ok=True)
    start=np.array(source['start'],np.float32);goal=np.array(source['goal'],np.float32)
    controls=initial_controls(start,goal,params['particles'],params['steps'],config,params['noise'],params['seed'])
    positions,velocities=rollout(controls,start,np.array(config.initial_velocity),config.sampling_time)
    scorer=ConformalGraphScorer(Path(source['checkpoint']),Path(source['calibration']),config.cp_coverage,config.cp_kind)
    learned,_=make_smooth_cp_value_and_grad(scorer.model,scorer.quantile,config)(positions)
    _,lower=cp_cost(np.asarray(positions),scorer,config)
    eligible=np.flatnonzero(lower>=config.required_robustness)
    if not len(eligible):raise ValueError('No initial candidates pass CP')
    no_collision=replace(config,collision_weight=0.,speed_weight=0.)
    def other(p,v,u):
        speed=jnp.sqrt(jnp.sum(v*v,axis=-1)+1e-12)
        return (_smooth_cost_single(p,jnp.array(start),jnp.array(goal),no_collision)
                +config.speed_weight*jnp.mean(jnp.maximum(speed-config.maximum_speed,0.)**2)
                +config.terminal_velocity_weight*jnp.mean((v[-1]-jnp.array(config.goal_velocity))**2)
                +config.control_weight*jnp.mean(u*u))
    def coefficient(p):
        full=jnp.concatenate((jnp.array(start)[None],p))
        d=jnp.linalg.norm(full[:,:,None,:]-full[:,None,:,:]+1e-8,axis=-1)
        off=1.-jnp.eye(len(start))
        return (jnp.maximum(config.minimum_separation-d,0.)**2*off).sum()/(len(start)*(len(start)-1))
    b=np.asarray(jax.jit(jax.vmap(other))(positions,velocities,jnp.array(controls)),float)+np.asarray(learned,float)
    a=np.asarray(jax.jit(jax.vmap(coefficient))(positions),float)
    crossings=[]
    for i in eligible:
        for j in eligible[eligible<i]:
            if a[i]!=a[j]:
                w=(b[j]-b[i])/(a[i]-a[j])
                if w>0:crossings.append(float(w))
    boundaries=[0.]+sorted(set(crossings))+[float('inf')]
    segments=[]
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        weight=(max(1.,2*left) if np.isinf(right) else (right/2 if left==0 else np.sqrt(left*right)))
        index=int(eligible[np.argmin((b+weight*a)[eligible])])
        if segments and segments[-1]['particle']==index:segments[-1]['weight_upper']=None if np.isinf(right) else right
        else:segments.append({'weight_lower':left,'weight_upper':None if np.isinf(right) else right,'particle':index})
    by_index={}
    for p in args.output.glob('weight_*.json'):
        trial=json.loads(p.read_text());report=trial['planner_report'];w=trial['summary']['collision_weight']
        if trial['parameters']!=params:raise ValueError('Trial population or other parameters differ from frontier')
        predicted=int(eligible[np.argmin((b+w*a)[eligible])])
        if report['selected_population']==0:
            if predicted!=report['index']:raise AssertionError(f'Frontier mismatch for weight {w}: {predicted} != {report["index"]}')
            if abs((b+w*a)[predicted]-report['cost'])>0.2:raise AssertionError('Cost mismatch')
            by_index[report['index']]={k:v for k,v in trial['summary'].items() if k in ('violating_pairs','violating_pair_intervals','minimum_continuous_separation_m','zero_collision','physical_overlap_pairs','physical_overlap_pair_intervals','zero_body_overlap')}
    for segment in segments:
        i=segment['particle']
        if i not in by_index:
            by_index[i]=continuous_separation(start,np.asarray(positions[i]),np.asarray(velocities[i]),controls[i],config.sampling_time,config.minimum_separation,config.initial_velocity,robot_radius=config.robot_radius)
        segment.update(by_index[i])
    result={'scope':'Initial candidate cost envelope only; full optimization can produce other candidates.',
            'parameters':params,'minimum_bounds_by_particle':lower.tolist(),
            'base_costs':b.tolist(),'collision_coefficients':a.tolist(),'segments':segments}
    (args.output/'initial_weight_frontier.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(segments,indent=2),flush=True)


if __name__=='__main__':main()
