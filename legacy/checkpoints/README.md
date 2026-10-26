# Archived Orbax checkpoints

These directories are the raw model and optimizer checkpoints produced by the
earliest padded GAT experiments. They are retained for provenance and are not
used by the maintained `python -m gnn` workflows, which use `.msgpack` files
under the generated top-level `checkpoints/` directory.

The directory names were normalized during repository cleanup:

| Original root name | Archived name |
| --- | --- |
| `pure_dict_state` | `original_gat_state` |
| `pure_dict_optimzier` | `original_gat_optimizer` |
| `pure_dict_state2` | `original_gatv2_state` |
| `pure_dict_optimzier2` | `original_gatv2_optimizer` |

The misspelling in the original optimizer paths is intentionally documented
here so older experiment notes can be interpreted.
