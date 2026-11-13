"""Validate the complete matched SI experiment and its saved motion metrics."""
import argparse
import datetime
import json
from pathlib import Path
import numpy as np
from planning.collision_metrics import linear_separation
from planning.run_single_integrator_trials import PROJECT,reference_trials


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('reports/trials/11-12/single_integrator_trials'))
    args=parser.parse_args();root=args.root
    references=reference_trials()
    expected={key[:4] for key in references if not key[-1]}
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['complete'] and manifest['jax_platforms']=='cuda'
    assert len(manifest['cases'])==6 and all(case['complete'] for case in manifest['cases'])
    source=json.loads((PROJECT/'reports/filtered_results/11-09/100robots_r21_four_methods.json').read_text())
    files=sorted(root.glob('robots_*_r*/particles_*/*/weight_*.json'))
    assert len(files)==2*len(expected)==54
    found=set();accepted=0;candidate_evaluations=0
    selected_checks=[]
    for path in files:
        payload=json.loads(path.read_text());cfg=payload['config'];params=payload['parameters'];report=payload['planner_report']
        n=len(payload['start']);particles=params['particles'];target=cfg['required_robustness'];weight=cfg['collision_weight'];policy=cfg['collision_filter']
        key=(n,target,particles,weight)
        assert key in expected and (*key,policy) not in found
        found.add((*key,policy))
        assert cfg['dynamics']=='single_integrator' and report['dynamics']=='single_integrator'
        assert report['control_units']=='m/s' and 'accelerations' not in report
        assert params['robots']==n and params['steps']==320 and params['iterations']==5 and params['seed']==42
        assert params['noise']==.1 and params['noise_units']=='m/s' and np.isclose(params['noise_fraction'],.1/3.)
        assert cfg['sampling_time']==.05 and cfg['maximum_speed']==3 and cfg['communication_radius']==3
        assert cfg['robot_radius']==.2 and cfg['minimum_separation']==.45 and not cfg['fixed_goal']
        assert cfg['cp_weight']==5 and cfg['goal_weight']==10 and cfg['degree_weight']==2 and cfg['svgd_step_size']==.01
        np.testing.assert_array_equal(payload['start'],np.asarray(source['start'][:n],np.float32))
        np.testing.assert_array_equal(payload['goal'],np.asarray(source['goal'][:n],np.float32))
        assert report['evaluated_candidates']==particles*6
        assert report['passing_candidates']+report['collision_rejected_candidates']==report['cp_passing_candidates']
        assert sum(item['cp_passing_candidates'] for item in report['population_acceptance'])==report['cp_passing_candidates']
        if policy:candidate_evaluations+=report['evaluated_candidates']
        positions=payload['waypoints']
        if positions is None:
            assert report['status']=='no_feasible_trajectory' and report['passing_candidates']==0
            continue
        assert report['status']=='feasible' and payload['summary']['cp_success']
        positions=np.asarray(positions);controls=np.asarray(report['controls']);velocities=np.asarray(report['velocities'])
        assert positions.shape==controls.shape==velocities.shape==(320,n,2)
        np.testing.assert_array_equal(controls,velocities)
        assert np.linalg.norm(controls,axis=-1).max()<=3.000001
        movement=np.diff(np.concatenate((np.asarray(payload['start'])[None],positions)),axis=0)
        np.testing.assert_allclose(movement,cfg['sampling_time']*controls,rtol=0.,atol=1e-5)
        bounds=payload['metrics']['lower_bounds_by_step']
        assert len(bounds)==320 and min(bounds)>=target
        assert min(bounds)==payload['summary']['minimum_cp_bound']==report['minimum_lower_bound']
        measured=linear_separation(payload['start'],positions,cfg['minimum_separation'],cfg['robot_radius'])
        for metric,value in measured.items():
            assert np.isclose(value,payload['summary'][metric],rtol=1e-12,atol=1e-12),(path,metric)
        if policy:
            accepted+=1
            assert measured['zero_collision'] and measured['zero_body_overlap']
            assert measured['minimum_continuous_separation_m']>=.45-1e-9
        selected_checks.append(dict(result_file=str(path.relative_to(root)),collision_filter=policy,
            minimum_cp_bound=min(bounds),minimum_continuous_separation_m=measured['minimum_continuous_separation_m'],
            physical_overlap_pairs=measured['physical_overlap_pairs'],maximum_speed_m_s=float(np.linalg.norm(controls,axis=-1).max())))
    assert found=={(*key,policy) for key in expected for policy in (False,True)}
    comparison=json.loads((root/'matched_dynamics_comparison.json').read_text())
    assert len(comparison)==27
    result=dict(complete=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        configurations=27,policy_results=len(files),candidate_evaluations=candidate_evaluations,
        joint_successful_configurations=accepted,selected_trajectory_checks=selected_checks,
        checks=['Exact prior DI weight/population cases reproduced.','Same saved start/goal geometry and exact r targets.',
                'Position-only SI rollout verified against every saved velocity control.',
                'Hard 3 m/s velocity bound; soft goal config retained.',
                'CP bounds at all 320 sampled waypoints meet requested r.',
                'Every saved continuous linear separation/body-overlap metric independently recomputed.',
                'No-feasible outputs contain null trajectories; accepted joint outputs, if any, pass continuous clearance.'])
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='selected_trajectory_checks'},indent=2))


if __name__=='__main__':main()
