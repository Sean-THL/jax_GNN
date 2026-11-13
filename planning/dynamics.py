"""Planar single/double integrators with zero-order-held controls."""
import jax
import jax.numpy as jnp
import numpy as np


def project_accelerations(controls, bound):
    """Enforce a Euclidean acceleration bound per robot and control interval."""
    controls = jnp.asarray(controls)
    magnitude = jnp.sqrt(jnp.sum(controls**2, axis=-1, keepdims=True) + 1e-20)
    return controls * jnp.minimum(1.0, bound / magnitude)


def rollout(controls, start, initial_velocity, dt):
    """Return future positions/velocities; leading dimensions are batched.

    p[k+1] = p[k] + dt*v[k] + 0.5*dt**2*u[k]
    v[k+1] = v[k] + dt*u[k]
    """
    controls = jnp.asarray(controls)
    position = jnp.broadcast_to(jnp.asarray(start), controls.shape[:-3] + controls.shape[-2:])
    velocity = jnp.broadcast_to(jnp.asarray(initial_velocity), position.shape)
    def step(state, acceleration):
        p, v = state
        p_next = p + dt * v + 0.5 * dt**2 * acceleration
        v_next = v + dt * acceleration
        return (p_next, v_next), (p_next, v_next)
    _, (positions, velocities) = jax.lax.scan(step, (position, velocity), jnp.moveaxis(controls, -3, 0))
    return jnp.moveaxis(positions, 0, -3), jnp.moveaxis(velocities, 0, -3)


def initial_controls(start, goal, count, steps, config, noise, seed):
    """Minimum-energy seed with target terminal position and velocity.

    Adds control noise in the nullspace of the terminal equalities, then
    projects to the acceleration bound. Projection may relax terminal arrival;
    neither terminal position nor velocity is a hard planning constraint.
    """
    if config.dynamics == 'single_integrator':
        return initial_single_integrator_controls(start,goal,count,steps,config,noise,seed)
    if steps < 2 or count < 2 or noise < 0:
        raise ValueError("double integrator needs steps >= 2, particles >= 2, noise >= 0")
    dt = config.sampling_time
    duration = steps * dt
    start, goal = np.asarray(start), np.asarray(goal)
    v0 = np.broadcast_to(config.initial_velocity, start.shape)
    vg = np.broadcast_to(config.goal_velocity, start.shape)
    matrix = np.stack((dt**2 * (steps - np.arange(steps) - 0.5), np.full(steps, dt)))
    terminal = np.stack((goal - start - duration * v0, vg - v0))
    inverse = matrix.T @ np.linalg.inv(matrix @ matrix.T)
    baseline = np.einsum('tk,knd->tnd', inverse, terminal)
    perturb = np.random.default_rng(seed).normal(0., noise, (count, steps, *start.shape))
    terminal_noise = np.einsum('kt,ptnd->pknd', matrix, perturb)
    perturb -= np.einsum('tk,pknd->ptnd', inverse, terminal_noise)
    perturb[0] = 0.0
    controls = baseline[None] + perturb
    return np.asarray(project_accelerations(controls, config.maximum_acceleration), dtype=np.float64)


def project_velocities(controls,bound):
    """Project each velocity command onto the Euclidean speed ball."""
    return project_accelerations(controls,bound)


def rollout_single_integrator(controls,start,dt):
    """Position is the state; u is held velocity: p[k+1] = p[k] + dt*u[k]."""
    controls=jnp.asarray(controls)
    position=jnp.broadcast_to(jnp.asarray(start),controls.shape[:-3]+controls.shape[-2:])
    def step(p,u):
        following=p+dt*u
        return following,following
    _,positions=jax.lax.scan(step,position,jnp.moveaxis(controls,-3,0))
    return jnp.moveaxis(positions,0,-3),controls


def initial_single_integrator_controls(start,goal,count,steps,config,noise,seed):
    """Velocity seeds targeting arrival and final velocity, without fixing endpoints.

    The minimum-energy seed reaches the goal and uses goal_velocity on the
    final interval. Zero-sum noise preserves these seed properties before
    speed projection. Optimization is free to change both terminal values.
    """
    if steps<2 or count<2 or noise<0 or config.sampling_time<=0 or config.maximum_speed<=0:
        raise ValueError('single integrator needs positive dt/speed, steps/particles >= 2 and noise >= 0')
    start,goal=np.asarray(start),np.asarray(goal)
    if start.shape!=goal.shape or start.ndim!=2 or start.shape[-1]!=2:
        raise ValueError('start and goal must have shape (robots, 2)')
    final=np.broadcast_to(config.goal_velocity,start.shape)
    baseline=np.broadcast_to(((goal-start)/config.sampling_time-final)/(steps-1),(steps,*start.shape)).copy()
    baseline[-1]=final
    perturb=np.random.default_rng(seed).normal(0.,noise,(count,steps,*start.shape))
    perturb[:,-1]=0.
    perturb[:,:-1]-=perturb[:,:-1].mean(axis=1,keepdims=True)
    perturb[0]=0.
    return np.asarray(project_velocities(baseline[None]+perturb,config.maximum_speed),dtype=np.float64)
