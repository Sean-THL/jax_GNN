import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from planning.svgd_cp_planner import PlannerConfig
from planning import tune_collision_weight


class CollisionTrialOverrideTests(unittest.TestCase):
    def test_exact_even_targets_and_robot_prefix_reach_planner_and_saved_results(self):
        config=asdict(PlannerConfig(3.,.45,3.,.05,21,dynamics='double_integrator'))
        start=np.column_stack((np.arange(50)*.5,np.zeros(50)))
        metrics=dict(meets_predicted_target=False,minimum_lower_bound=None,goal_mean_distance=None)
        def report(count,policy):
            return dict(status='no_feasible_trajectory',collision_filter_enabled=policy,
                        cp_passing_candidates=0,passing_candidates=0,collision_rejected_candidates=0,
                        evaluated_candidates=count*2,selected_population=None)
        source=dict(config=config,start=start.tolist(),goal=(start+20).tolist(),
                    parameters=dict(robots=50,particles=2,steps=2,iterations=1,noise=.1,seed=42),
                    checkpoint='fake.msgpack',calibration='fake.json',
                    results={'gnn_cp_autodiff':dict(planner_config=config,planner_report=report(2,False),
                            waypoints=None,wall_seconds=.1,**metrics)})
        for robots,target,count,compare,dynamics in ((50,20,2,False,'double_integrator'),(25,10,4,True,'double_integrator'),(25,10,4,True,'single_integrator')):
            with self.subTest(robots=robots,target=target),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source_file=root/'source.json';source_file.write_text(json.dumps(source))
                output=root/'joint' if compare else root/'cp_only'
                args=['trial','--source',str(source_file),'--output',str(output),
                      '--robots',str(robots),'--particles',str(count),'--required-robustness',str(target),'--weights','10']
                if compare:args+=['--compare-cp-only-output',str(root/'cp_only')]
                else:args+=['--legacy-cp-only']
                if dynamics=='single_integrator':args+=['--dynamics',dynamics]
                primary=report(count,compare)
                if compare:primary['cp_only_comparison']=dict(waypoints=None,planner_report=report(count,False))
                def controls(s,g,p,t,c,noise,seed):
                    self.assertEqual((len(s),len(g),p,c.required_robustness),(robots,robots,count,target))
                    self.assertEqual(c.dynamics,dynamics)
                    return np.zeros((p,t,len(s),2))
                planner=('plan_single_integrator' if dynamics=='single_integrator' else 'plan_double_integrator') if compare else 'plan'
                with patch('sys.argv',args),patch.object(tune_collision_weight,'ConformalGraphScorer'),\
                     patch.object(tune_collision_weight,'initial_controls',side_effect=controls),\
                     patch.object(tune_collision_weight,planner,return_value=(None,primary)) as call,\
                     patch.object(tune_collision_weight,'trajectory_summary',return_value=metrics),\
                     contextlib.redirect_stdout(io.StringIO()):
                    tune_collision_weight.main()
                call.assert_called_once()
                self.assertEqual(call.call_args.args[4].required_robustness,target)
                result=json.loads((output/'weight_10.json').read_text())
                self.assertEqual(result['config']['required_robustness'],target)
                self.assertEqual(result['config']['dynamics'],dynamics)
                if dynamics=='single_integrator':
                    self.assertEqual(result['parameters']['noise_units'],'m/s')
                    self.assertAlmostEqual(result['parameters']['noise_fraction'],.1/3.)
                self.assertEqual(result['parameters']['robots'],robots)
                self.assertEqual(result['parameters']['particles'],count)
                np.testing.assert_array_equal(result['start'],start[:robots])
                if compare:
                    alternative=json.loads((root/'cp_only/weight_10.json').read_text())
                    self.assertFalse(alternative['config']['collision_filter'])
                    self.assertEqual(alternative['config']['required_robustness'],target)
                self.assertEqual(source_file.read_text(),json.dumps(source))
