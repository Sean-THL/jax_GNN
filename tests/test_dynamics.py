import unittest
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from planning.dynamics import initial_controls, project_accelerations, rollout
from planning.svgd_cp_planner import PlannerConfig, communication_records, plan

class DegreeScorer:
    def lower_bounds(self, positions, radius, mode, sigma):
        return np.array([int(r['features'][:,0].min()) for r in communication_records(positions,radius,mode,sigma)])

class DynamicsTests(unittest.TestCase):
    def test_body_radius_cannot_exceed_half_safety_separation(self):
        start=np.array([[0.,0.],[0.5,0.]])
        config=PlannerConfig(3.,0.39,3.,0.05,1,dynamics='double_integrator',cp_weight=0.,robot_radius=0.2)
        with self.assertRaisesRegex(ValueError,'twice robot_radius'):
            plan(start,start,np.zeros((2,2,2,2)),DegreeScorer(),config,iterations=1)

    def test_constant_acceleration(self):
        controls=jnp.ones((2,20,3,2))*jnp.array([0.6,0.8])
        p,v=rollout(controls,jnp.zeros((3,2)),jnp.array([0.2,-0.1]),0.05)
        t=np.arange(1,21)*0.05
        expected_p=t[:,None]*np.array([0.2,-0.1])+0.5*t[:,None]**2*np.array([0.6,0.8])
        expected_v=np.array([0.2,-0.1])+t[:,None]*np.array([0.6,0.8])
        np.testing.assert_allclose(p[0,:,0],expected_p,atol=2e-7)
        np.testing.assert_allclose(v[0,:,0],expected_v,atol=1e-6)

    def test_projection_uses_vector_norm(self):
        value=np.asarray(project_accelerations(jnp.array([[3.,4.],[0.,0.]]),1.))
        np.testing.assert_allclose(value,[[0.6,0.8],[0.,0.]],atol=1e-7)
        self.assertTrue(np.isfinite(jax.jacfwd(project_accelerations)(jnp.zeros((1,2)),1.)).all())

    def test_far_seed_reaches_goal_at_rest(self):
        start=np.array([[0.,0.],[0.,0.5]])
        goal=start+[10.,0.]
        config=PlannerConfig(3.,0.3,3.,0.05,1,dynamics='double_integrator')
        controls=initial_controls(start,goal,3,200,config,0.05,42)
        p,v=rollout(controls,start,jnp.zeros(2),config.sampling_time)
        np.testing.assert_allclose(p[:,-1],np.broadcast_to(goal,(3,2,2)),atol=1e-4)
        np.testing.assert_allclose(v[:,-1],0.,atol=1e-5)
        self.assertLessEqual(np.linalg.norm(controls,axis=-1).max(),1.+1e-6)

    def test_position_gradient_reaches_controls(self):
        fn=lambda a:rollout(a,jnp.zeros((2,2)),jnp.zeros(2),0.05)[0][-1].sum()
        grad=jax.grad(fn)(jnp.zeros((4,2,2)))
        expected=0.05**2*(4-np.arange(4)-0.5)
        np.testing.assert_allclose(grad[:,0,0],expected,rtol=1e-6)

    def test_planner_follows_dynamics_and_input_bound(self):
        start=np.array([[0.,0.],[0.5,0.]])
        goal=start+[0.2,0.]
        config=PlannerConfig(3.,0.1,3.,0.05,1,dynamics='double_integrator',cp_weight=0.)
        controls=initial_controls(start,goal,2,40,config,0.01,42)
        selected,report=plan(start,goal,controls,DegreeScorer(),config,iterations=2)
        self.assertIsNotNone(selected)
        p,v=rollout(jnp.array(report['accelerations']),start,jnp.zeros(2),0.05)
        np.testing.assert_allclose(selected,p,atol=1e-7)
        np.testing.assert_allclose(report['velocities'],v,atol=1e-7)
        self.assertLessEqual(report['maximum_acceleration_norm'],1.+1e-6)
        with self.assertRaisesRegex(ValueError,'fixed_goal is unsupported'):
            plan(start,goal,controls,DegreeScorer(),replace(config,fixed_goal=True),iterations=1)

    def test_cp_failure_returns_no_plan(self):
        start=np.array([[0.,0.],[5.,0.]])
        config=PlannerConfig(1.,0.1,3.,0.05,1,dynamics='double_integrator',cp_weight=0.)
        selected,report=plan(start,start,np.zeros((2,4,2,2)),DegreeScorer(),config,iterations=1)
        self.assertIsNone(selected)
        self.assertEqual(report['status'],'no_feasible_trajectory')
