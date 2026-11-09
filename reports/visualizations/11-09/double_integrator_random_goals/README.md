# Independent random robot goals

Video: `double_integrator_explained.mp4`.

Ten red numbered robots move toward independently sampled, numbered green
goals. The endpoint pools are sampled with different seeds, rather than a
common translation. Actual travel distances range from 66.87 m to
69.67 m. Each robot i is assigned goal i. The endpoint formations
stay compact enough to support the robustness target at R=3 m; the boxes are
sampling regions, not claims that ten robots fill the whole workspace.

Planar double integrator, dt=0.05 s, 800 steps (40 seconds), acceleration norm
bound 1 m/s², 8 particles, 5 SVGD updates, seed 42. Initial and goal regions are
[2,10] × [2,10] m and [70,78] × [2,10] m. Reference workspace: 80 × 12 m.
Goals and collision separation are soft costs; the robot paths and final
positions are the actual selected result, not forced by animation.

- Maximum acceleration: 0.59810 m/s².
- Mean final goal error: 0.00006 m.
- Minimum waypoint CP lower bound: 5.
- Minimum exact waypoint robustness: 5.
- Minimum selected robot separation: 0.02620 m
  (configured soft separation target: 0.3 m).

Dashed gray lines show hard-radius communication at opacity 0.5. The blue
circle shows R=3 m around robot 1; orange arrows show acceleration inputs.
Exact robustness and minimum degree are evaluated on rendered graphs, with
CP labels referring to sampled waypoints. Playback is half speed, 12 fps,
using the constant-acceleration equations between saved states. Frame sampling
does not certify every instant or guarantee collision avoidance.

Reproduce from `jax_GNN/` using the commands in the planner README's
'Independently sampled robot goals' section.
