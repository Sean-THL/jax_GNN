"""Readable moving network close-up, with a small route map and diagnostics."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gnn-planner-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
from planning.run_experiment import PROJECT, exact_robustness


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=PROJECT/'reports/visualizations/11-09/random_goals_20m_clear')
    parser.add_argument('--fps',type=int,default=20)
    args=parser.parse_args();folder=args.input
    metadata=json.loads((folder/'metrics.json').read_text())
    info=metadata['methods']['gnn_cp_autodiff'];config=info['config'];report=info['planner_report']
    with np.load(folder/'planner_snapshots.npz') as data:
        start,goal=data['start'],data['goal']
        path=np.concatenate((start[None],data['gnn_cp_autodiff__selected']))
    steps,robots=len(path)-1,len(start)
    camera_half=max(2.,float(np.abs(path-path.mean(axis=1,keepdims=True)).max())+0.6)
    dt=config['sampling_time'];duration=steps*dt;radius=config['communication_radius'];target=config['required_robustness']
    controls=np.array(report['accelerations']);velocities=np.concatenate((np.broadcast_to(config['initial_velocity'],start.shape)[None],np.array(report['velocities'])))
    cache={}
    def graph(positions):
        distance=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)
        adjacency=(distance<=radius)&~np.eye(robots,dtype=bool)
        key=adjacency.tobytes()
        if key not in cache:
            cache[key]=exact_robustness({'features':np.column_stack((adjacency.sum(1),np.ones(robots))),'edge_index':np.array(np.nonzero(adjacency))})
        separation=distance[~np.eye(robots,dtype=bool)].min()
        return adjacency,cache[key],int(adjacency.sum(1).min()),separation
    samples=[graph(p) for p in path]
    robustness=np.array([v[1] for v in samples],dtype=float);degree=np.array([v[2] for v in samples]);times=np.arange(steps+1)*dt
    cp_history=np.array([np.nan]+info['metrics']['lower_bounds_by_step'],dtype=float)
    fig=plt.figure(figsize=(14,9),dpi=100,facecolor='white')
    fig.text(0.055,0.955,f'{robots} robots · required robustness {target}',fontsize=23,fontweight='bold',color='#192b40')
    fig.text(0.055,0.913,'GNN–CP–autodiff  |  double integrator  |  R = 3 m  |  acceleration bound = 1 m/s²',fontsize=12,color='#536276')
    clock=fig.text(0.945,0.955,'',ha='right',fontsize=13,color='#192b40')
    ax=fig.add_axes([0.065,0.16,0.56,0.69]);ax.set_aspect('equal')
    ax.set_title('Robot network — camera follows the team',fontsize=14,pad=16)
    ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.grid(alpha=0.12)
    dot_size=130 if robots>10 else 210
    label_size=7 if robots>10 else 10
    ax.scatter(*goal.T,color='#159447',s=dot_size+30,edgecolors='white',linewidths=1.2,zorder=4)
    for r in range(robots):
        ax.annotate(str(r+1),goal[r],xytext=(10,9),textcoords='offset points',fontsize=label_size,color='#087234',fontweight='bold',clip_on=True)
    links=LineCollection([],colors='gray',linestyles='--',alpha=0.5,linewidths=0.5 if robots>10 else 1.2,zorder=1);ax.add_collection(links)
    dots=ax.scatter(*start.T,color='#dc3030',s=dot_size,edgecolors='white',linewidths=1.5,zorder=5)
    labels=[ax.text(*start[r],str(r+1),ha='center',va='center',color='white',fontsize=label_size,fontweight='bold',zorder=6,clip_on=True) for r in range(robots)]
    fig.text(0.065,0.075,'Red: robot   •   Green: its assigned goal   •   Dashed gray: communication link',fontsize=11,color='#43516a')
    fig.text(0.065,0.043,f'Fixed {2*camera_half:.1f} × {2*camera_half:.1f} m close-up. Links: distance ≤ {radius:g} m. Small map: full trip.',fontsize=10,color='#536276')
    overview=fig.add_axes([0.69,0.68,0.27,0.17]);overview.set_title('Full route — same robot positions',fontsize=12,pad=10)
    overview.set_aspect('equal');overview.set_xlim(0,34);overview.set_ylim(0,12);overview.grid(alpha=0.13)
    overview.scatter(*start.T,s=12,facecolors='none',edgecolors='#e47575',alpha=0.7)
    overview.scatter(*goal.T,s=18,color='#159447',zorder=3)
    for r in range(robots):
        overview.plot(*path[:,r].T,color='#9ba8b8',lw=0.6,alpha=0.35)
    mapdots=overview.scatter(*start.T,s=16,color='#dc3030',zorder=4)
    viewport=Rectangle((0,0),2*camera_half,2*camera_half,fill=False,edgecolor='#2685bc',linewidth=1.3);overview.add_patch(viewport)
    overview.tick_params(labelsize=8);overview.set_xlabel('World x (m)',fontsize=9)
    stats=fig.text(0.69,0.59,'',va='top',fontsize=12,linespacing=1.45,color='#192b40')
    chart=fig.add_axes([0.72,0.17,0.24,0.18]);chart.set_title('Graph values during motion',fontsize=12,pad=12)
    chart.set_xlim(0,duration);chart.set_ylim(0,max(10,robots));chart.set_yticks(sorted({0,target,robots-1}));chart.set_xlabel('Time (s)',fontsize=10);chart.grid(alpha=0.14)
    rline,=chart.plot([],[],color='#138253',lw=2,label='Exact r' if robots<=10 else 'CP bound (waypoint)')
    dline,=chart.plot([],[],color='#6c7481',lw=1.8,label='Min degree')
    chart.axhline(target,color='#c78316',ls='--',lw=1,label=f'Target r = {target}')
    cursor=chart.axvline(0,color='#2685bc',lw=1,alpha=0.7)
    chart.legend(loc='upper left',bbox_to_anchor=(0,-0.30),ncol=2,fontsize=9,frameon=False)
    moving=round(duration*args.fps);hold=args.fps
    def update(frame):
        progress=float(np.clip((frame-hold)/moving,0,1)*steps)
        k=min(int(progress),steps-1);tau=(progress-k)*dt
        positions=path[k]+tau*velocities[k]+0.5*tau*tau*controls[k]
        velocity=velocities[k]+tau*controls[k]
        adjacency,r,d,separation=graph(positions)
        pairs=np.array(np.nonzero(np.triu(adjacency,1))).T
        links.set_segments(positions[pairs] if len(pairs) else [])
        dots.set_offsets(positions);mapdots.set_offsets(positions)
        center=positions.mean(0);ax.set_xlim(center[0]-camera_half,center[0]+camera_half);ax.set_ylim(center[1]-camera_half,center[1]+camera_half)
        viewport.set_xy(center-camera_half)
        for i,label in enumerate(labels):label.set_position(positions[i])
        shown=min(int(progress),steps);time=progress*dt
        clock.set_text(f'{time:5.2f} / {duration:.1f} s')
        cp='initial state' if shown==0 else str(info['metrics']['lower_bounds_by_step'][shown-1])
        goal_error=np.linalg.norm(positions-goal,axis=1).mean()
        r_text=str(r) if r is not None else 'not computed'
        stats.set_text(f'Exact robustness r: {r_text}  (target {target})\nMinimum degree: {d}   |   CP: {cp}\nMean distance to goals: {goal_error:.2f} m\nMax speed: {np.linalg.norm(velocity,axis=1).max():.2f} m/s\nMax input |u|: {np.linalg.norm(controls[k],axis=1).max():.2f} m/s²\nMin separation: {separation:.2f} m')
        rline.set_data(times[:shown+1],(robustness if robots<=10 else cp_history)[:shown+1]);dline.set_data(times[:shown+1],degree[:shown+1]);cursor.set_xdata([time,time])
        return []
    update(hold+moving//2)
    fig.savefig(folder/'network_closeup_preview.png',dpi=120)
    output=folder/(f'network_closeup_{robots}robots_r{target}.mp4' if robots>10 else 'network_closeup_20m.mp4')
    animation=FuncAnimation(fig,update,frames=moving+2*hold+1,interval=1000/args.fps,blit=False)
    animation.save(output,writer=FFMpegWriter(fps=args.fps,codec='libx264',bitrate=4000,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart']))
    plt.close(fig);print(f'Saved {output}',flush=True)


if __name__=='__main__':main()
