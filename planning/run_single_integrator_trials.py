"""Repeat the prior DI collision experiments with single-integrator controls."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parent.parent
DI_ROOTS=(PROJECT/'reports/trials/11-11/large_particle_collision_trials',
          PROJECT/'reports/trials/11-12/robustness_collision_trials')


def reference_trials():
    trials={}
    for root in DI_ROOTS:
        for path in sorted(root.rglob('weight_*.json')):
            payload=json.loads(path.read_text());config=payload['config'];params=payload['parameters']
            if config.get('dynamics')!='double_integrator':continue
            key=(len(payload['start']),config['required_robustness'],params['particles'],config['collision_weight'],config['collision_filter'])
            trials[key]=(path,payload)
    return trials


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('reports/trials/11-12/single_integrator_trials'))
    parser.add_argument('--cases',nargs='+',help='Optional robots:r:particles subset; default repeats every prior case')
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    cases={}
    for key,(path,payload) in reference_trials().items():
        robots,target,particles,weight,policy=key
        if policy:continue
        cases.setdefault((robots,target,particles),{})[weight]=str(path)
    selected=set(tuple(map(int,value.split(':'))) for value in args.cases) if args.cases else set(cases)
    if not selected or not selected.issubset(cases):parser.error('requested cases must exist in the saved double-integrator experiments')
    manifest=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=False,
                  dynamics='single_integrator',control_units='m/s',maximum_speed=3.,noise=.1,
                  jax_platforms=os.environ.get('JAX_PLATFORMS'),cases=[])
    for key in sorted(selected,key=lambda value:(-value[0],-value[1],value[2])):
        robots,target,particles=key;weights=sorted(cases[key]);folder=root/f'robots_{robots}_r{target}'/f'particles_{particles}'
        command=[sys.executable,'-m','planning.tune_collision_weight','--dynamics','single_integrator',
                 '--robots',str(robots),'--required-robustness',str(target),'--particles',str(particles),
                 '--weights',*[str(weight) for weight in weights],'--progress','--cache-hard-batches','--output',str(folder/'joint_filter'),
                 '--compare-cp-only-output',str(folder/'cp_only')]
        entry=dict(robots=robots,required_robustness=target,particles=particles,weights=weights,
                   reference_results=cases[key],command=command,complete=False)
        manifest['cases'].append(entry)
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'Single integrator: robots={robots}, r={target}, particles={particles}, weights={weights}',flush=True)
        subprocess.run(command,cwd=PROJECT,check=True)
        entry['complete']=True
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        subprocess.run([sys.executable,'-m','planning.report_robustness_collision_trials','--root',str(root),'--dynamics','single_integrator'],cwd=PROJECT,check=True)
    manifest.update(complete=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
