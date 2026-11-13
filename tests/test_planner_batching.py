import unittest
from collections import OrderedDict
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from planning.dynamics import rollout
from planning.double_integrator_planner import bounded_value_and_grad
from planning.svgd_cp_planner import ConformalGraphScorer,PlannerConfig,_smooth_cost_single


class PlannerBatchingTests(unittest.TestCase):
    def test_cache_preserves_order_partial_batches_and_cp_settings(self):
        a=np.array([[0.,0.],[.1,0.],[0.,.1]],np.float32)
        b=np.array([[0.,0.],[3.,0.],[0.,3.]],np.float32)
        positions=np.stack((a,b,b,a,a,b,a))
        def predict(model,records,batch_size,representation):
            labels=np.array([int(record['features'][:,0].sum()) for record in records])
            probabilities=np.full((len(records),26),.001);probabilities[np.arange(len(records)),labels]=.975
            return probabilities,np.zeros(len(records),int)
        scorer=ConformalGraphScorer.__new__(ConformalGraphScorer)
        scorer.model=None;scorer.batch_size=2;scorer.graph_chunk_size=4
        scorer.hard_batch_cache=OrderedDict();scorer.hard_batch_cache_capacity=8
        scorer.hard_batch_cache_hits=0;scorer.hard_batch_cache_misses=0
        with patch('planning.svgd_cp_planner.predict_probabilities',side_effect=predict):
            for kind,quantile in (('one_sided',0.),('one_sided',1.),('lac',.05)):
                scorer.kind=kind;scorer.quantile=quantile;scorer.cache_hard_batches=False
                expected=scorer.lower_bounds(positions,1.)
                scorer.cache_hard_batches=True
                first=scorer.lower_bounds(positions,1.);second=scorer.lower_bounds(positions,1.)
                np.testing.assert_array_equal(first,expected);np.testing.assert_array_equal(second,expected)
        self.assertGreater(scorer.hard_batch_cache_hits,0)
        self.assertLessEqual(len(scorer.hard_batch_cache),8)

    def test_bounded_base_values_and_gradients_match_full_vmap(self):
        config=PlannerConfig(3.,.45,2.,.05,2)
        start=jnp.array([[0.,0.],[.3,0.],[0.,.3]])
        goal=start+1.
        def objective(u):
            p,_=rollout(u,start,jnp.zeros(2),config.sampling_time)
            return _smooth_cost_single(p,start,goal,config)
        for count in (2,11):
            u=jnp.asarray(np.random.default_rng(42).normal(0,.2,(count,4,3,2)),dtype=jnp.float32)
            expected=jax.jit(jax.vmap(jax.value_and_grad(objective)))(u)
            actual=bounded_value_and_grad(objective)(u)
            for a,b in zip(actual,expected):
                np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)

    def test_cp_chunking_preserves_bounds_order_and_partial_batch(self):
        positions=np.array([[[0.,0.],[d,0.],[0.,d]] for d in (.1,3.,.5,4.,.8,.2,5.,1.5,7.)],np.float32)
        sizes=[]
        def predict(model,records,batch_size,representation):
            sizes.append(len(records))
            labels=np.array([int(round(record['features'][:,0].sum()))%26 for record in records])
            probabilities=np.full((len(records),26),.001)
            probabilities[np.arange(len(records)),labels]=.975
            return probabilities,np.zeros(len(records),dtype=int)
        scorer=ConformalGraphScorer.__new__(ConformalGraphScorer)
        scorer.model=None;scorer.batch_size=2;scorer.quantile=0.
        with patch('planning.svgd_cp_planner.predict_probabilities',side_effect=predict):
            for mode in ('hard','smooth'):
                for kind in ('one_sided','lac'):
                    with self.subTest(mode=mode,kind=kind):
                        scorer.kind=kind;scorer.graph_chunk_size=10000
                        scorer.quantile=0. if kind=='one_sided' else .05
                        expected=scorer.lower_bounds(positions,1.,mode)
                        sizes.clear();scorer.graph_chunk_size=4
                        actual=scorer.lower_bounds(positions,1.,mode)
                        np.testing.assert_array_equal(actual,expected)
                        self.assertEqual(sizes,[4,4,1])


if __name__=='__main__':unittest.main()
