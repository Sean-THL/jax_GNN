import json
from pathlib import Path
import unittest
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from gnn.models.gatv2 import GNN
from planning.run_experiment import config_from_scenario, scenario_positions
from planning.svgd_cp_planner import communication_records, make_smooth_cp_value_and_grad, PlannerConfig
from planning.dynamics import initial_controls, rollout


class LargeTeamTests(unittest.TestCase):
    def test_100_robot_scenario_and_dynamic_seed(self):
        path=Path(__file__).resolve().parents[1]/'planning/examples/double_integrator_100robots.json'
        scenario=json.loads(path.read_text());config,_=config_from_scenario(scenario['config'])
        start,goal=scenario_positions(scenario,100,config.required_robustness,config.communication_radius)
        self.assertEqual(start.shape,(100,2));self.assertEqual(goal.shape,(100,2))
        self.assertGreater(np.ptp(goal-start,axis=0).max(),0.5)
        controls=initial_controls(start,goal,2,scenario['horizon_steps'],config,0.,42)
        positions,velocities=rollout(controls,start,jnp.zeros(2),config.sampling_time)
        np.testing.assert_allclose(positions[0,-1],goal,atol=1e-4)
        self.assertLessEqual(np.linalg.norm(controls,axis=-1).max(),1.+1e-6)
        for record in communication_records(np.stack((start,goal)),3.):
            self.assertEqual(record['features'].shape,(100,2))
            self.assertGreaterEqual(record['features'][:,0].min(),config.required_robustness)

    def test_100_robot_gnn_position_gradients(self):
        model=GNN(2,8,26,nnx.Rngs(11),batchsize=1);model.eval()
        grid=np.stack(np.meshgrid(np.arange(10),np.arange(10)),axis=-1).reshape(100,2)*0.2
        evaluate=make_smooth_cp_value_and_grad(model,0.,PlannerConfig(3.,0.45,3.,0.05,25))
        cost,gradient=evaluate(jnp.asarray(grid[None,None],dtype=jnp.float32))
        self.assertTrue(np.isfinite(cost).all());self.assertTrue(np.isfinite(gradient).all())
        self.assertGreater(float(jnp.linalg.norm(gradient)),0.)

    def test_above_supported_team_size_is_explicit(self):
        with self.assertRaisesRegex(ValueError,'1..100'):
            communication_records(np.zeros((1,101,2)),3.)
