"""Hard geometric acceptance checks, separate from learned CP predictions."""
import numpy as np
from planning.collision_metrics import continuous_separation,linear_separation


def separation_acceptance(start, positions, threshold, *, robot_radius=None,
                          velocities=None, controls=None, dt=None, initial_velocity=(0.,0.)):
    """Reject sampled collisions quickly, then check every motion interval.

    Double-integrator paths use quadratic constant-acceleration motion.
    Single-integrator and waypoint paths use straight segments. Includes start.
    """
    if not np.isfinite(threshold) or threshold<0:
        raise ValueError("separation threshold must be finite and nonnegative")
    if robot_radius is not None and (not np.isfinite(robot_radius) or robot_radius<=0 or threshold<2*robot_radius):
        raise ValueError("separation threshold must be at least twice a positive robot radius")
    start=np.asarray(start,dtype=float);positions=np.asarray(positions,dtype=float)
    if not np.isfinite(start).all() or not np.isfinite(positions).all():
        return {'zero_collision':False,'rejection_reason':'nonfinite_trajectory'}
    path=np.concatenate((start[None],positions))
    i,j=np.triu_indices(len(start),1)
    distances=np.linalg.norm(path[:,i]-path[:,j],axis=-1)
    sampled_min=float(distances.min())
    if sampled_min < threshold-1e-9:
        return {'zero_collision':False,'minimum_waypoint_separation_m':sampled_min,
                'separation_threshold_m':float(threshold),'rejection_reason':'waypoint_collision'}
    if controls is not None:
        if not np.isfinite(controls).all() or not np.isfinite(velocities).all():
            return {'zero_collision':False,'rejection_reason':'nonfinite_trajectory'}
        result=continuous_separation(start,positions,velocities,controls,dt,threshold,
                                     initial_velocity,robot_radius=robot_radius)
    else:
        result=linear_separation(start,positions,threshold,robot_radius)
    result['rejection_reason']=None if result['zero_collision'] else 'between_waypoint_collision'
    return result
