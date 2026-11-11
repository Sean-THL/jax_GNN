import unittest
import numpy as np
from planning.collision_metrics import continuous_separation

class CollisionMetricsTests(unittest.TestCase):
    def test_body_overlap_is_separate_from_clearance_violation(self):
        for separation,overlap in ((0.43,False),(0.39,True)):
            start=np.array([[0.,0.],[0.,separation]])
            zero=np.zeros((1,2,2))
            result=continuous_separation(start,start[None],zero,zero,1.,0.45,robot_radius=0.2)
            self.assertFalse(result['zero_collision'])
            self.assertEqual(result['physical_overlap_pairs'],int(overlap))
            self.assertAlmostEqual(result['minimum_surface_gap_m'],separation-0.4)

    def test_detects_crossing_between_clear_waypoints(self):
        start=np.array([[-1.,0.],[1.,0.]])
        v=np.array([[2.,0.],[-2.,0.]])
        result=continuous_separation(start,-start[None],v[None],np.zeros((1,2,2)),1.,0.45,v)
        self.assertAlmostEqual(result['minimum_continuous_separation_m'],0.)
        self.assertEqual(result['waypoint_pair_violations'],0)
        self.assertEqual(result['violating_pairs'],1)

    def test_accelerated_crossing_uses_interior_stationary_point(self):
        start=np.array([[-.5,0.],[.5,0.]])
        controls=np.array([[[2.,0.],[-2.,0.]]])
        result=continuous_separation(start,-start[None],controls.copy(),controls,1.,.2)
        self.assertLess(result['minimum_continuous_separation_m'],1e-7)
        self.assertFalse(result['zero_collision'])

    def test_parallel_motion_at_threshold_is_clear(self):
        start=np.array([[0.,0.],[0.,.45]])
        velocity=np.ones((2,2))
        result=continuous_separation(start,(start+velocity)[None],velocity[None],np.zeros((1,2,2)),1.,.45,velocity)
        self.assertTrue(result['zero_collision'])
        self.assertEqual(result['violating_pair_intervals'],0)
