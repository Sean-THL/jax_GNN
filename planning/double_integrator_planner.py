"""Compatibility entry point for bounded double-integrator SVGD."""
from dataclasses import replace
from planning.control_integrator_planner import bounded_value_and_grad,plan_control_integrator


def plan_double_integrator(start,goal,initial_controls,scorer,config,iterations,seed=42,record_particles=False,compare_cp_only=False):
    return plan_control_integrator(start,goal,initial_controls,scorer,replace(config,dynamics="double_integrator"),iterations,seed,record_particles,compare_cp_only)
