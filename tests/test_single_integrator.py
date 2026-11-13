from dataclasses import replace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from planning.dynamics import rollout_single_integrator,initial_controls,project_velocities
from planning.collision_metrics import linear_separation
from planning.svgd_cp_planner import PlannerConfig,plan
from planning.single_integrator_planner import plan_single_integrator


class PassingScorer:
    def lower_bounds(self,positions,*args):return np.ones(len(positions),dtype=int)


class SingleIntegratorTests(unittest.TestCase):
    def config(self):
        return PlannerConfig(3.,.45,3.,1.,1,dynamics='single_integrator',maximum_acceleration=0.,
            cp_weight=0.,degree_weight=0.,collision_weight=0.,path_weight=1.,terminal_velocity_weight=0.,
            control_weight=0.,svgd_step_size=0.,robot_radius=.2)

    def test_rollout_and_control_gradient_are_single_integration(self):
        controls=jnp.array([[[1.,0.]],[[0.,2.]]]);start=jnp.zeros((1,2))
        p,v=rollout_single_integrator(controls,start,.5)
        np.testing.assert_allclose(p,[[[.5,0.]],[[.5,1.]]])
        np.testing.assert_array_equal(v,controls)
        gradient=jax.grad(lambda u:rollout_single_integrator(u,start,.5)[0][-1].sum())(controls)
        np.testing.assert_allclose(gradient,.5)
        batch,_=rollout_single_integrator(jnp.stack((controls,controls)),start,.5)
        np.testing.assert_array_equal(batch[0],p)

    def test_velocity_projection_bounds_vector_norm(self):
        clipped=np.asarray(project_velocities(np.array([[[3.,4.]]]),2.))
        np.testing.assert_allclose(clipped,[[[1.2,1.6]]],rtol=1e-6)

    def test_seed_arrival_and_stop_are_not_fixed_planner_constraints(self):
        config=replace(self.config(),sampling_time=.1)
        start=np.array([[0.,0.],[0.,1.]]);goal=start+np.array([2.,0.])
        controls=initial_controls(start,goal,4,20,config,.1,42)
        np.testing.assert_array_equal(controls,initial_controls(start,goal,4,20,config,.1,42))
        positions,velocities=rollout_single_integrator(controls,start,.1)
        np.testing.assert_allclose(positions[:,-1],np.broadcast_to(goal,(4,2,2)),atol=1e-5)
        np.testing.assert_array_equal(velocities[:,-1],0.)

    def test_collision_between_clear_waypoints_is_rejected(self):
        start=np.array([[-1.,0.],[1.,0.]]);goal=-start
        u=np.array([[[[2.,0.],[-2.,0.]],[[0.,0.],[0.,0.]]]]*2)
        selected,report=plan(start,goal,u,PassingScorer(),self.config(),1)
        self.assertIsNone(selected)
        self.assertEqual(report['collision_rejection_reasons'],{'between_waypoint_collision':4})
        p,_=rollout_single_integrator(u[0],start,1.)
        metrics=linear_separation(start,p,.45,robot_radius=.2)
        self.assertGreater(metrics['minimum_waypoint_separation_m'],.45)
        self.assertEqual(metrics['physical_overlap_pairs'],1)
        self.assertAlmostEqual(metrics['minimum_continuous_separation_m'],0.)

    def test_shared_and_independent_acceptance_match_with_soft_goal(self):
        start=np.array([[-1.,0.],[1.,0.]]);goal=-start
        u=np.array([[[[2.,0.],[-2.,0.]],[[0.,0.],[0.,0.]]],
                    [[[2.,2.],[-2.,-2.]],[[0.,0.],[0.,0.]]]])
        selected,report=plan_single_integrator(start,goal,u,PassingScorer(),self.config(),1,compare_cp_only=True)
        joint,joint_report=plan(start,goal,u,PassingScorer(),self.config(),1)
        cp,cp_report=plan(start,goal,u,PassingScorer(),replace(self.config(),collision_filter=False),1)
        np.testing.assert_array_equal(selected,joint)
        np.testing.assert_array_equal(report['cp_only_comparison']['waypoints'],cp)
        self.assertEqual(report['index'],1);self.assertEqual(cp_report['index'],0)
        self.assertNotIn('accelerations',report)
        self.assertEqual(report['control_units'],'m/s')
        self.assertEqual(report['dynamics'],'single_integrator')
        self.assertGreater(np.linalg.norm(selected[-1]-goal),1.)
        self.assertTrue(report['separation_check']['zero_collision'])
        for key in ('passing_candidates','cp_passing_candidates','collision_rejected_candidates','population_acceptance'):
            self.assertEqual(report[key],joint_report[key])
            self.assertEqual(report['cp_only_comparison']['planner_report'][key],cp_report[key])

    def test_hard_speed_and_soft_unreachable_goal(self):
        cfg=replace(self.config(),maximum_speed=.5)
        start=np.array([[-1.,0.],[1.,0.]]);goal=start+100.
        u=np.ones((2,2,2,2))*100.
        selected,report=plan(start,goal,u,PassingScorer(),cfg,1)
        self.assertIsNotNone(selected)
        self.assertLessEqual(report['maximum_velocity_norm'],.500001)
        controls=np.array(report['controls'])
        np.testing.assert_allclose(np.diff(np.concatenate((start[None],selected)),axis=0),controls*cfg.sampling_time,atol=1e-6)
        self.assertGreater(np.linalg.norm(selected[-1]-goal),100.)

    def test_hard_goal_pin_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'fixed_goal'):
            plan(np.array([[0.,0.],[1.,0.]]),np.array([[1.,0.],[2.,0.]]),np.zeros((2,2,2,2)),PassingScorer(),replace(self.config(),fixed_goal=True),1)
