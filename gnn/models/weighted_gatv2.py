"""GATv2 with edge weights that gate messages as well as attention logits."""

from __future__ import annotations

import jax.numpy as jnp
from flax.nnx import leaky_relu
from jraphx.nn.conv import GATv2Conv
from jraphx.utils import scatter_add, scatter_softmax
from jraphx.utils.loop import add_self_loops as add_self_loops_fn


class WeightedGATv2Conv(GATv2Conv):
    """Checkpoint-compatible GATv2 whose scalar edge feature gates messages.

    For the compact binary graphs used during training every real edge has
    weight one, so this is algebraically identical to ``GATv2Conv``.  Smooth
    mode uses a fixed dense off-diagonal edge index and gates normalized
    attention by the equation-(7) weight.  Self loops retain unit gates and
    the original zero-valued edge feature.
    """

    def __call__(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        heads, channels = self.heads, self.out_features
        residual = None
        if isinstance(x, tuple):
            x_left, x_right = x
            num_nodes = x_right.shape[0] if x_right is not None else x_left.shape[0]
            if self.res is not None and x_right is not None:
                residual = self.res(x_right)
            x_left = self.lin_l(x_left).reshape(-1, heads, channels)
            x_right = (
                self.lin_r(x_right).reshape(-1, heads, channels)
                if x_right is not None else x_left
            )
        else:
            num_nodes = x.shape[0]
            if self.res is not None:
                residual = self.res(x)
            x_left = self.lin_l(x).reshape(-1, heads, channels)
            x_right = self.lin_r(x).reshape(-1, heads, channels)

        if self._add_self_loops:
            edge_index, edge_attr = add_self_loops_fn(
                edge_index, edge_attr=edge_attr, fill_value=self.fill_value,
                num_nodes=num_nodes,
            )
        row, col = edge_index[0], edge_index[1]
        combined = x_right[col] + x_left[row]
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.reshape(-1, 1)
            edge_features = self.lin_edge(edge_attr).reshape(-1, heads, channels)
            combined = combined + edge_features

        logits = jnp.sum(
            leaky_relu(combined, negative_slope=self.negative_slope) * self.att.value,
            axis=-1,
        )
        expanded_col = jnp.repeat(col, heads)
        attention = scatter_softmax(
            logits.reshape(-1), expanded_col, dim_size=num_nodes
        )

        if edge_attr is None:
            edge_weight = jnp.ones(row.shape, dtype=attention.dtype)
        else:
            edge_weight = edge_attr[:, 0].astype(attention.dtype)
        # jraphx adds zero-feature self loops.  They must remain active, while
        # zero-weight off-diagonal edges in the fixed dense graph stay silent.
        edge_weight = jnp.where(row == col, 1.0, edge_weight)
        expanded_weight = jnp.repeat(edge_weight, heads)
        weighted_attention = attention * expanded_weight
        normalizer = scatter_add(
            weighted_attention, expanded_col, dim_size=num_nodes
        )
        weighted_attention = weighted_attention / normalizer[expanded_col]
        attention = weighted_attention.reshape(-1, heads)

        if self.dropout is not None:
            attention = self.dropout(attention)
        messages = x_left[row] * attention[..., None]
        out = scatter_add(
            messages.reshape(-1, heads * channels), col, dim_size=num_nodes
        ).reshape(num_nodes, heads, channels)
        out = out.reshape(num_nodes, heads * channels) if self.concat else out.mean(axis=1)
        if residual is not None:
            out = out + residual
        if self.bias is not None:
            out = out + self.bias.value
        if return_attention_weights:
            return out, (edge_index, attention)
        return out
