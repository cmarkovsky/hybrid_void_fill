#!/usr/bin/env python3
"""Build realization metadata from the chips and the FIXED split; never repartition glaciers.

One row per glacier and retained realization: support (glacier & finite dh/dt),
the expected void target count and the achieved fraction. Every split glacier
must have a chip, verified attributes (region whim/ehim) and a consistent void
stack; otherwise nothing is written and metadata_failures.csv lists the problems.
Void pixels outside the support are counted (they are never targets).
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from workflow_support import workflow_path,parse_stage_args,write_provenance
import numpy as np,pandas as pd

def chip_records(path,gid,group,region):
    with np.load(path,allow_pickle=False) as d:
        gm=np.asarray(d['glacier_mask'],bool);truth=np.asarray(d['dhdt']);v=np.asarray(d['void_mask'],bool)
        ids=np.asarray(d['attempted_realization_index']) if 'attempted_realization_index' in d.files else np.arange(len(v))
    if v.ndim!=3 or v.shape[1:]!=truth.shape or gm.shape!=truth.shape:raise ValueError('Inconsistent chip array shapes')
    if len(ids)!=len(v) or len(np.unique(ids))!=len(ids):raise ValueError('attempted_realization_index does not match the void stack')
    valid=gm&np.isfinite(truth);n_valid=int(valid.sum())
    if n_valid==0:raise ValueError('No glacier pixel has finite dh/dt')
    out=[]
    for r,mask in enumerate(v):
        target=valid&mask
        out.append({'RGIId':gid,'region':region,'group':group,'realization':r,'realization_index':r,
                    'attempted_realization_index':int(ids[r]),'n_valid_glacier_pixels':n_valid,
                    'n_expected_void_pixels':int(target.sum()),'n_void_outside_support':int((mask&~valid).sum()),
                    'achieved_frac':float(target.sum()/n_valid)})
    return out

def build_metadata(chip_dir,split,attributes):
    idcol='RGIId' if 'RGIId' in attributes else 'rgiid'
    if 'region' not in attributes:raise ValueError('Attributes must include verified region labels')
    if attributes[idcol].duplicated().any():raise ValueError('Duplicate glacier attributes')
    attrs=attributes.set_index(idcol);records=[];failures=[]
    for gid,group in sorted(split.items()):
        try:
            group=str(group).lower()
            if group not in {'train','val','test'}:raise ValueError(f'Invalid split group {group!r}')
            if gid not in attrs.index:raise ValueError('Missing glacier attributes')
            region=str(attrs.loc[gid,'region']).lower()
            if region not in {'whim','ehim'}:raise ValueError('Unknown region '+region)
            path=Path(chip_dir)/f'{gid}.npz'
            if not path.exists():raise FileNotFoundError(f'Missing chip {path.name}')
            records.extend(chip_records(path,gid,group,region))
        except Exception as e:failures.append({'RGIId':gid,'error':f'{type(e).__name__}: {e}'})
    return pd.DataFrame(records),pd.DataFrame(failures)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for option,key in [('chip-dir','chips'),('split','split'),('attributes','attributes'),('output','metadata')]:p.add_argument('--'+option,type=Path,default=workflow_path(key))
    p.add_argument('--overwrite',action='store_true');a=parse_stage_args(p)
    if a.output.exists() and not a.overwrite:raise FileExistsError('Metadata exists; choose a new path or explicitly pass --overwrite')
    frame,failures=build_metadata(a.chip_dir,json.loads(a.split.read_text()),pd.read_csv(a.attributes))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if not failures.empty:
        failures.to_csv(a.output.parent/'metadata_failures.csv',index=False)
        raise RuntimeError(f'{len(failures)} split glaciers have missing/invalid inputs; metadata was not written')
    frame.to_csv(a.output,index=False) if a.output.suffix=='.csv' else frame.to_parquet(a.output,index=False)
    write_provenance(a.output.parent,a,__file__)
    print(frame[['RGIId','region','group']].drop_duplicates().groupby(['region','group']).size().to_string())
    outside=int(frame.n_void_outside_support.sum())
    if outside:print(f'NOTE: {outside:,} void pixels lie outside glacier & finite dh/dt and are not targets')
if __name__=='__main__':main()
