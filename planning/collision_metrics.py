"""Continuous pairwise separation for held velocity or acceleration controls."""
import numpy as np


def linear_separation(start,positions,threshold,robot_radius=None):
    """Exact pairwise minima for piecewise-constant velocity motion."""
    path=np.concatenate((np.asarray(start,dtype=float)[None],np.asarray(positions,dtype=float)))
    i,j=np.triu_indices(path.shape[1],1)
    relative=path[:-1,i]-path[:-1,j]
    change=(path[1:,i]-path[1:,j])-relative
    squared=np.sum(change*change,axis=-1)
    fraction=np.zeros_like(squared)
    np.divide(-np.sum(relative*change,axis=-1),squared,out=fraction,where=squared>0)
    fraction=np.clip(fraction,0.,1.)
    minimum=np.linalg.norm(relative+fraction[...,None]*change,axis=-1)
    violation=minimum<threshold-1e-9
    sampled=np.linalg.norm(path[:,i]-path[:,j],axis=-1)
    result=dict(zero_collision=bool(not violation.any()),minimum_continuous_separation_m=float(minimum.min()),
        minimum_waypoint_separation_m=float(sampled.min()),separation_threshold_m=float(threshold),
        violating_pairs=int(violation.any(axis=0).sum()),violating_pair_intervals=int(violation.sum()),
        waypoint_pair_violations=int((sampled<threshold-1e-9).sum()))
    if robot_radius is not None:
        diameter=2*float(robot_radius);overlap=minimum<diameter-1e-9
        result.update(robot_radius_m=float(robot_radius),body_diameter_m=diameter,
            physical_overlap_pairs=int(overlap.any(axis=0).sum()),physical_overlap_pair_intervals=int(overlap.sum()),
            zero_body_overlap=bool(not overlap.any()),minimum_surface_gap_m=float(minimum.min()-diameter))
    return result


def continuous_separation(start, positions, velocities, controls, dt, threshold, initial_velocity=(0.,0.), robot_radius=None):
    """Minimize squared pair distance over each control interval.

    Relative position is quadratic in normalized interval time s in [0,1].
    Check the endpoints and real stationary points (roots of a cubic).
    Counts are unique pairs and pair/control-interval violations, not time samples.
    """
    positions=np.asarray(positions,dtype=float);start=np.asarray(start,dtype=float)
    velocities=np.asarray(velocities,dtype=float);controls=np.asarray(controls,dtype=float)
    p=np.concatenate((start[None],positions))
    v=np.concatenate((np.broadcast_to(initial_velocity,start.shape)[None],velocities))
    i,j=np.triu_indices(len(start),1)
    relative=p[:-1,i]-p[:-1,j]
    b=dt*(v[:-1,i]-v[:-1,j])
    c=0.5*dt*dt*(controls[:,i]-controls[:,j])
    dot=lambda a,b:np.sum(a*b,axis=-1)
    c0=dot(relative,b);c1=dot(b,b)+2*dot(relative,c)
    c2=3*dot(b,c);c3=2*dot(c,c)
    distance0=dot(relative,relative)
    end=relative+b+c
    minimum=np.minimum(distance0,dot(end,end))
    # Zero relative acceleration gives a linear stationary equation.
    curved=c3>1e-24
    flat=~curved
    roots=np.zeros_like(c0)
    np.divide(-c0,c1,out=roots,where=flat & (np.abs(c1)>1e-24))
    valid=flat & (roots>=0) & (roots<=1)
    value=relative+roots[...,None]*b+(roots**2)[...,None]*c
    minimum=np.where(valid,np.minimum(minimum,dot(value,value)),minimum)
    indices=np.flatnonzero(curved)
    flat_min=minimum.reshape(-1)
    rflat,bflat,cflat=(a.reshape(-1,2) for a in (relative,b,c))
    # Bound CPU memory for long horizons/large teams.
    for begin in range(0,len(indices),50000):
        selected=indices[begin:begin+50000]
        a3=c3.ravel()[selected]
        companion=np.zeros((len(selected),3,3))
        companion[:,0,2]=-c0.ravel()[selected]/a3
        companion[:,1,0]=1.;companion[:,1,2]=-c1.ravel()[selected]/a3
        companion[:,2,1]=1.;companion[:,2,2]=-c2.ravel()[selected]/a3
        solutions=np.linalg.eigvals(companion)
        for column in range(3):
            root=solutions[:,column]
            valid=(np.abs(root.imag)<1e-7)&(root.real>=0)&(root.real<=1)
            s=root.real[valid];idx=selected[valid]
            q=rflat[idx]+s[:,None]*bflat[idx]+(s*s)[:,None]*cflat[idx]
            flat_min[idx]=np.minimum(flat_min[idx],dot(q,q))
    interval_distance=np.sqrt(np.maximum(minimum,0.))
    violation=interval_distance < threshold-1e-9
    waypoint_distance=np.linalg.norm(p[:,i]-p[:,j],axis=-1)
    result = {
        'minimum_continuous_separation_m':float(interval_distance.min()),
        'minimum_waypoint_separation_m':float(waypoint_distance.min()),
        'violating_pairs':int(violation.any(axis=0).sum()),
        'violating_pair_intervals':int(violation.sum()),
        'waypoint_pair_violations':int((waypoint_distance<threshold-1e-9).sum()),
        'zero_collision':bool(not violation.any()),
        'separation_threshold_m':float(threshold),
    }
    if robot_radius is not None:
        diameter=2*float(robot_radius)
        overlap=interval_distance < diameter-1e-9
        result.update(robot_radius_m=float(robot_radius),body_diameter_m=diameter,
                      physical_overlap_pairs=int(overlap.any(axis=0).sum()),
                      physical_overlap_pair_intervals=int(overlap.sum()),
                      zero_body_overlap=bool(not overlap.any()),
                      minimum_surface_gap_m=float(interval_distance.min()-diameter))
    return result
