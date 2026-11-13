"""Run controlled collision-weight sweeps at explicit robustness targets."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parent.parent
DEFAULT_CASES=('50:20:50','50:20:100','25:10:50','25:10:100')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('reports/trials/11-12/robustness_collision_trials'))
    parser.add_argument('--cases',nargs='+',default=DEFAULT_CASES,help='robots:r:particles triples')
    parser.add_argument('--weights',type=float,nargs='+',default=[10,1000,10000,100000])
    parser.add_argument('--manifest-name',default='manifest.json',help='Use a separate manifest for an adaptive follow-up sweep')
    args=parser.parse_args()
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/args.manifest_name
    manifest=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  jax_platforms=os.environ.get('JAX_PLATFORMS'),weights=args.weights,cases=[],complete=False)
    for case in args.cases:
        robots,target,particles=map(int,case.split(':'))
        folder=root/f'robots_{robots}_r{target}'/f'particles_{particles}'
        command=[sys.executable,'-m','planning.tune_collision_weight','--robots',str(robots),
                 '--required-robustness',str(target),'--particles',str(particles),'--weights',
                 *[str(w) for w in args.weights],'--progress','--output',str(folder/'joint_filter'),
                 '--compare-cp-only-output',str(folder/'cp_only')]
        entry=dict(robots=robots,required_robustness=target,particles=particles,command=command,complete=False)
        manifest['cases'].append(entry)
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'Starting robots={robots}, r={target}, particles={particles}',flush=True)
        subprocess.run(command,cwd=PROJECT,check=True)
        entry['complete']=True
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        subprocess.run([sys.executable,'-m','planning.report_robustness_collision_trials','--root',str(root)],cwd=PROJECT,check=True)
    manifest.update(complete=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
