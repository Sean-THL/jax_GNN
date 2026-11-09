# Double-integrator changes, shown in one video

Open `double_integrator_explained.mp4`.

- Red numbered dots: 10 robots. Green numbered dots: their assigned goals, translated 10 m from the starts. Faint hollow red markers retain the starting positions.
- Dashed gray links at 50% opacity: hard communication edges at distance <=3 m.
- Blue dotted circle centered on robot 1: the 3 m communication radius, drawn once to avoid clutter. All robots use the same radius.
- Orange arrows: acceleration inputs u, with 1 plotted metre corresponding to 1 m/s². Forward arrows accelerate; backward arrows brake. This is an overlay scale, not an extra displacement applied to the robot.
- Header: time, waypoint index, sampling interval, maximum current acceleration, exact graph robustness and minimum degree. CP bounds refer to saved waypoints.

The scenario is `planning/examples/double_integrator_large.json`: dt=0.05 s,
200 steps (10 seconds), Euclidean acceleration input bound 1 m/s², R=3 m,
10 m goal translation, and a 20 x 10 m reference workspace. Workspace edges
are not hard walls. The sampled start box is [1,9] x [1,9] m; the first ten
robots form the compact seed of the pool, so they occupy only part of this box.
The blue circle shows a local communication radius, not a workspace boundary.

The video uses exact constant-acceleration motion between the sampled states:

```
p(t+tau) = p(t) + tau*v(t) + 0.5*tau^2*u(t)
v(t+tau) = v(t) + tau*u(t)
```

It plays at half speed. The controls came from GNN–CP–autodiff with 8 particles,
five SVGD updates and seed 42. The goal is a soft cost; endpoints are not pinned.
Mean final goal error is 0.06225 m; terminal mean speed is 0.00851 m/s. Maximum
acceleration over the selected plan is 0.84362 m/s², maximum speed 1.54927 m/s.
The speed limit and collision separation are soft costs. Final waypoint CP
bounds meet the target of 5. Rendered-frame checking does not certify every
instant of continuous motion.

From `jax_GNN/`, reproduce:

```bash
JAX_PLATFORMS=cuda .venv/bin/python -m planning.visualize_trajectory_updates \
  --scenario planning/examples/double_integrator_large.json \
  --robots 10 --particles 8 --iterations 5 --methods gnn_cp_autodiff \
  --record-only --output reports/visualizations/11-09/double_integrator_10m

.venv/bin/python -m planning.visualize_red_robots \
  --input reports/visualizations/11-09/double_integrator_10m --show-inputs
```
