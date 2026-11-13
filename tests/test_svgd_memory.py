import unittest
import numpy as np
from planning.svgd_cp_planner import svgd_direction


def original_direction(particles,gradient):
    flat=particles.reshape(len(particles),-1).astype(np.float64)
    gradient=gradient.reshape(flat.shape).astype(np.float64)
    distance=np.sum((flat[:,None,:]-flat[None,:,:])**2,axis=-1)
    bandwidth=max(float(np.median(distance[np.triu_indices(len(flat),1)]))/np.log(len(flat)+1),1e-8)
    kernel=np.exp(-distance/bandwidth)
    return ((kernel@gradient+2*(flat*kernel.sum(axis=1,keepdims=True)-kernel@flat)/bandwidth)/len(flat)).reshape(particles.shape)


class SVGDMemoryTests(unittest.TestCase):
    def test_rowwise_distances_preserve_full_tensor_direction(self):
        rng=np.random.default_rng(42)
        for shape in ((2,3,2,2),(9,7,13,2)):
            with self.subTest(shape=shape):
                particles=rng.normal(size=shape);gradient=rng.normal(size=shape)
                np.testing.assert_allclose(svgd_direction(particles,gradient),original_direction(particles,gradient),rtol=1e-12,atol=1e-12)

    def test_near_identical_offset_particles_and_duplicates(self):
        rng=np.random.default_rng(7)
        particles=10000.+rng.normal(0.,1e-5,(6,4,3,2));particles[1]=particles[0]
        gradient=rng.normal(size=particles.shape)
        np.testing.assert_allclose(svgd_direction(particles,gradient),original_direction(particles,gradient),rtol=1e-12,atol=1e-10)
