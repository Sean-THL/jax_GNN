"""SVGD over bounded velocity controls for p[k+1] = p[k] + dt*u[k]."""
from dataclasses import replace
from planning.control_integrator_planner import plan_control_integrator


def plan_single_integrator(start,goal,initial_controls,scorer,config,iterations,seed=42,record_particles=False,compare_cp_only=False):
    return plan_control_integrator(start,goal,initial_controls,scorer,replace(config,dynamics='single_integrator'),iterations,seed,record_particles,compare_cp_only)
