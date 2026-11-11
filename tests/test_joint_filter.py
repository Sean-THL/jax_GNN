import unittest
from dataclasses import replace
import numpy as np
from planning.svgd_cp_planner import PlannerConfig,plan
from planning.dynamics import rollout
from planning.double_integrator_planner import plan_double_integrator

class PassingScorer:
    def lower_bounds(self,positions,*args):return np.ones(len(positions),dtype=int)

class JointFilterTests(unittest.TestCase):
    def waypoint_config(self):
        return PlannerConfig(20.,.45,10.,1.,1,cp_weight=0.,degree_weight=0.,
                             collision_weight=0.,speed_weight=0.,svgd_step_size=0.,robot_radius=.2)

    def test_waypoint_crossing_rejected_between_clear_samples(self):
        start=np.array([[-1.,0.],[1.,0.]])
        goal=-start;particles=np.broadcast_to(goal,(2,1,2,2)).copy()
        selected,report=plan(start,goal,particles,PassingScorer(),self.waypoint_config(),iterations=1)
        self.assertIsNone(selected)
        self.assertEqual(report['cp_passing_candidates'],4)
        self.assertEqual(report['collision_rejected_candidates'],4)
        self.assertEqual(report['passing_candidates'],0)
        self.assertEqual(report['status'],'no_feasible_trajectory')

    def test_waypoint_chooses_more_expensive_safe_candidate(self):
        start=np.array([[-1.,0.],[1.,0.]])
        goal=-start
        particles=np.array([[np.zeros((2,2)),goal],[[[0.,1.],[0.,-1.]],goal]])
        cfg=replace(self.waypoint_config(),path_weight=1.)
        unsafe,_=plan(start,goal,particles,PassingScorer(),replace(cfg,collision_filter=False),iterations=1)
        safe,report=plan(start,goal,particles,PassingScorer(),cfg,iterations=1)
        np.testing.assert_array_equal(unsafe,particles[0])
        np.testing.assert_array_equal(safe,particles[1])
        self.assertGreaterEqual(report['separation_check']['minimum_continuous_separation_m'],.45)
        self.assertEqual(report['collision_rejected_candidates'],2)

    def test_acceleration_crossing_rejected_between_clear_samples(self):
        start=np.array([[-1.,0.],[1.,0.]])
        controls=np.array([[[[4.,0.],[-4.,0.]],[[-4.,0.],[4.,0.]]]]*2)
        cfg=replace(self.waypoint_config(),dynamics='double_integrator',maximum_acceleration=6.)
        positions,_=rollout(controls,start,np.zeros(2),1.)
        self.assertTrue((np.linalg.norm(np.asarray(positions)[:,:,0]-np.asarray(positions)[:,:,1],axis=-1)>.45).all())
        selected,report=plan(start,np.asarray(positions[0,-1]),controls,PassingScorer(),cfg,iterations=1)
        self.assertIsNone(selected)
        self.assertEqual(report['collision_rejection_reasons'],{'between_waypoint_collision':4})
        selected,report=plan_double_integrator(start,np.asarray(positions[0,-1]),controls,PassingScorer(),cfg,1,compare_cp_only=True)
        self.assertIsNone(selected)
        self.assertEqual(report['cp_only_comparison']['planner_report']['status'],'feasible')
        np.testing.assert_array_equal(report['cp_only_comparison']['waypoints'],positions[0])

    def test_dynamic_safe_candidate_wins_even_when_goal_is_soft(self):
        start=np.array([[-1.,0.],[1.,0.]])
        unsafe=np.array([[[4.,0.],[-4.,0.]],[[-4.,0.],[4.,0.]]])
        safe=np.array([[[4.,4.],[-4.,-4.]],[[-4.,-4.],[4.,4.]]])
        controls=np.stack((unsafe,safe));goal=np.array([[3.,0.],[-3.,0.]])
        cfg=replace(self.waypoint_config(),dynamics='double_integrator',maximum_acceleration=6.,path_weight=1.)
        selected,report=plan(start,goal,controls,PassingScorer(),cfg,iterations=1)
        expected,_=rollout(safe,start,np.zeros(2),1.)
        np.testing.assert_allclose(selected,expected,atol=1e-6)
        self.assertEqual(report['index'],1)
        self.assertEqual(report['collision_rejected_candidates'],2)
        self.assertEqual(report['cp_passing_candidates'],4)
        self.assertEqual(report['passing_candidates'],2)
        self.assertTrue(report['separation_check']['zero_collision'])
        self.assertGreater(np.linalg.norm(selected[-1]-goal),1.)

    def test_initial_overlap_rejected_even_if_future_separation_is_safe(self):
        start=np.array([[0.,0.],[.1,0.]])
        goal=np.array([[-1.,0.],[1.,0.]])
        selected,report=plan(start,goal,np.broadcast_to(goal,(2,1,2,2)).copy(),PassingScorer(),self.waypoint_config(),iterations=1)
        self.assertIsNone(selected)
        self.assertEqual(report['collision_rejected_candidates'],4)

    def test_shared_populations_match_separate_acceptance_runs(self):
        start=np.array([[-1.,0.],[1.,0.]])
        unsafe=np.array([[[4.,0.],[-4.,0.]],[[-4.,0.],[4.,0.]]])
        safe=np.array([[[4.,4.],[-4.,-4.]],[[-4.,-4.],[4.,4.]]])
        controls=np.stack((unsafe,safe));goal=np.array([[3.,0.],[-3.,0.]])
        cfg=replace(self.waypoint_config(),dynamics='double_integrator',maximum_acceleration=6.,path_weight=1.,svgd_step_size=.001)
        joint,report=plan_double_integrator(start,goal,controls,PassingScorer(),cfg,1,compare_cp_only=True)
        expected_joint,joint_report=plan(start,goal,controls,PassingScorer(),cfg,1)
        cp,cp_report=plan(start,goal,controls,PassingScorer(),replace(cfg,collision_filter=False),1)
        np.testing.assert_array_equal(joint,expected_joint)
        comparison=report['cp_only_comparison']
        np.testing.assert_array_equal(comparison['waypoints'],cp)
        for key in ('index','cost','minimum_lower_bound','selected_population','passing_candidates',
                    'cp_passing_candidates','collision_rejected_candidates','status','population_acceptance'):
            self.assertEqual(report[key],joint_report[key])
            self.assertEqual(comparison['planner_report'][key],cp_report[key])
        np.testing.assert_array_equal(comparison['planner_report']['accelerations'],cp_report['accelerations'])
