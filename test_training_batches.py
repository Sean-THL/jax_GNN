"""Regression checks for padding without changing real graph computations."""
import unittest

import flax.nnx as nnx
import jax
import numpy as np
import optax

from jax_variants import batches, batch_arrays, loss_fn, model_class, prepare_records, train_step


def records():
    return prepare_records([
        {"degree": [1, 1], "edge_index": [[0, 1], [1, 0]], "robustness": 1},
        {"degree": [0], "edge_index": [[], []], "robustness": 0},
        {"degree": [2, 2, 2], "edge_index": [[0, 1, 2], [1, 2, 0]], "robustness": 1},
    ])


class PaddingTests(unittest.TestCase):
    def test_padding_is_disconnected_and_shapes_reused(self):
        data = records()
        padded = list(batches(data, 2, "new"))
        self.assertEqual(padded[0][0].x.shape, padded[1][0].x.shape)
        self.assertEqual(padded[0][0].edge_index.shape, padded[1][0].edge_index.shape)
        self.assertEqual([count for _, count in padded], [2, 1])
        for (batch, count), (compact, _) in zip(padded, batches(data, 2, "new", pad_new=False)):
            n, e = len(compact.x), compact.edge_index.shape[1]
            np.testing.assert_array_equal(batch.x[:n], compact.x)
            np.testing.assert_array_equal(batch.edge_index[:, :e], compact.edge_index)
            self.assertTrue(np.all(batch.edge_index[:, e:] == n))
            self.assertTrue(np.all(batch.batch[n:] == count))

    def test_logits_losses_and_gradients_match(self):
        for convolution in ("gat", "gatv2"):
            model = model_class("new", convolution)(2, 8, 26, nnx.Rngs(42), 2)
            for (padded, count), (compact, _) in zip(
                batches(records(), 2, "new"), batches(records(), 2, "new", pad_new=False)
            ):
                model.batchsize = count
                model.eval()
                np.testing.assert_allclose(
                    model(*batch_arrays(padded)[:4]), model(*batch_arrays(compact)[:4]),
                    rtol=1e-5, atol=1e-6,
                )
                padded_loss, padded_grad = nnx.value_and_grad(loss_fn)(model, *batch_arrays(padded))
                compact_loss, compact_grad = nnx.value_and_grad(loss_fn)(model, *batch_arrays(compact))
                np.testing.assert_allclose(padded_loss, compact_loss, rtol=1e-5, atol=1e-6)
                for actual, expected in zip(jax.tree.leaves(padded_grad), jax.tree.leaves(compact_grad)):
                    self.assertTrue(np.all(np.isfinite(actual)))
                    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)

    def test_empty_edges_and_jitted_updates(self):
        data = records()[1:2]
        for convolution in ("gat", "gatv2"):
            model = model_class("new", convolution)(2, 8, 26, nnx.Rngs(42), 1)
            optimizer = nnx.Optimizer(model, optax.adam(0.005), wrt=nnx.Param)
            batch, count = next(batches(data, 2, "new"))
            self.assertEqual(count, 1)
            loss = train_step(model, optimizer, *batch_arrays(batch))
            self.assertTrue(np.isfinite(float(loss)))

    def test_original_representation_keeps_legacy_slots(self):
        batch, _ = next(batches(records(), 2, "original"))
        self.assertEqual(batch.x.shape, (100, 2))
        self.assertEqual(batch.edge_index.shape, (2, 5000))
        self.assertTrue(np.all(batch.edge_index[:, 2:2500] == 0))
        self.assertTrue(np.all(batch.edge_index[:, 2500:] == 50))


if __name__ == "__main__":
    unittest.main()
