"""Shared path resolution, split validation, and provenance for all stages."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from functools import lru_cache
from pathlib import Path
from datetime import datetime,timezone
HOME_DIR=Path(__file__).resolve().parent

def _consume_global_options():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--project-root');p.add_argument('--workflow-config')
    args,rest=p.parse_known_args()
    if args.project_root:os.environ['VOID_FILL_PROJECT_ROOT']=str(Path(args.project_root).expanduser().resolve())
    if args.workflow_config:os.environ['VOID_FILL_WORKFLOW_CONFIG']=str(Path(args.workflow_config).expanduser().resolve())
    sys.argv=[sys.argv[0],*rest]
_consume_global_options()

@lru_cache(maxsize=1)
def project_root():
    configured=os.environ.get('VOID_FILL_PROJECT_ROOT')
    if configured:return Path(configured).expanduser().resolve()
    candidate=HOME_DIR.parent
    if (candidate/'data_prepared').is_dir():return candidate
    raise RuntimeError('Set --project-root /absolute/path/to/data_archive (or use workflow.py --root ...). Pass the extracted data archive explicitly.')

def project_path(value):
    if str(value).startswith('@repo/'):
        return HOME_DIR.parent / str(value)[6:]
    p=Path(value).expanduser()
    return p if p.is_absolute() else project_root()/p

@lru_cache(maxsize=1)
def settings():
    config=json.loads((HOME_DIR.parent/'config/workflow.json').read_text())
    chosen=os.environ.get('VOID_FILL_WORKFLOW_CONFIG')
    if chosen:
        update=json.loads(Path(chosen).read_text())
        for key,value in update.items():
            if isinstance(value,dict) and isinstance(config.get(key),dict):config[key].update(value)
            else:config[key]=value
    if config['missing_predictors'] not in {'error','drop','native'}:raise ValueError('Unknown missing_predictors policy')
    return config

def workflow_path(name):return project_path(settings()['paths'][name])

def parse_stage_args(parser):
    args=parser.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,project_path(value))
    return args

def check_row_inventory(rows_dir,split_path):
    split=json.loads(Path(split_path).read_text());issues=[];groups={}
    for group in ['train','val','test']:
        expected={str(g) for g,v in split.items() if str(v).lower()==group}
        actual={p.stem for p in (Path(rows_dir)/group).glob('*.npz')};groups[group]=actual
        if actual!=expected:issues.append(f'{group}: missing {len(expected-actual)}, unexpected {len(actual-expected)}; examples missing={sorted(expected-actual)[:5]}, unexpected={sorted(actual-expected)[:5]}')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if groups[a]&groups[b]:issues.append(f'{a}/{b} overlap: {sorted(groups[a]&groups[b])[:5]}')
    if issues:raise ValueError('Row inventory does not match fixed split. Use a fresh row directory and inspect preparation failures.\n'+'\n'.join(issues))
    return {k:len(v) for k,v in groups.items()}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_provenance(output,args,source):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'project_root':str(project_root()),
       'source':str(source),'source_sha256':sha256(source),'settings':settings(),
       'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
       'split_sha256':sha256(workflow_path('split')) if workflow_path('split').is_file() else None}
    (output/'workflow_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
