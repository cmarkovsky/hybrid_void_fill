#!/usr/bin/env python3
"""Inventory the fixed split, chip support and predictor availability for ALL glaciers."""
from __future__ import annotations
import argparse,json,sys,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from workflow_support import workflow_path,parse_stage_args,write_provenance
import numpy as np,pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for option,key in [('cube','cube'),('chip-dir','chips'),('split','split'),('attributes','attributes')]:p.add_argument('--'+option,type=Path,default=workflow_path(key))
    p.add_argument('--output-dir',type=Path,default=workflow_path('rows').parent/'input_validation')
    p.add_argument('--skip-cube',action='store_true');a=parse_stage_args(p)
    source=Path(__file__).resolve().parents[1]/'04_model/01_prepare_rows.py'
    spec=importlib.util.spec_from_file_location('preparation_validation',source);prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    split=prep.load_split(a.split);attrs=prep.load_glacier_attributes(a.attributes);ds=None;rows=[]
    if not a.skip_cube:ds=prep.CubeReader(a.cube)   # direct Zarr reads with a chunk cache
    for i,(gid,group) in enumerate(sorted(split.items()),1):
        record={'rgiid':gid,'group':group,'status':'ok'}
        try:
            chip=prep.load_npz(a.chip_dir/f'{gid}.npz');v=np.asarray(chip['void_mask'],bool)
            if v.ndim!=3:raise ValueError('void_mask must be a retained-realization stack')
            if gid not in attrs:raise ValueError('Missing glacier attributes')
            g=np.asarray(chip['glacier_mask'],bool)&np.isfinite(chip['dhdt']);targets=v&g
            x=np.asarray(chip['x']);y=np.asarray(chip['y'])
            if x.ndim!=1 or y.ndim!=1 or len(x)<2 or len(y)<2:raise ValueError('Need 1-D pixel-centre coordinates')
            # Chip coordinates are projected pixel centres. The archived grid
            # uses a nominal 30 m cell size with small projection/serialization
            # drift (up to ~0.00041 m in this inventory), so NumPy's default
            # relative tolerance is too strict for the y-axis values.
            pixel_tolerance_m=1e-3
            if (not np.allclose(np.abs(np.diff(x)),30,rtol=0,atol=pixel_tolerance_m)
                    or not np.allclose(np.abs(np.diff(y)),30,rtol=0,atol=pixel_tolerance_m)):
                raise ValueError('Expected uniform nominal 30-m grid; glacier metrics use 900 m2')
            if np.any(targets&~np.isfinite(chip['z'])):raise ValueError('Intended void contains missing elevation')
            dc=prep.extract_datacube_window(ds,chip) if ds is not None else None
            for r,t in enumerate(targets):
                rec={**record,'realization_index':r,'n_valid_glacier_pixels':int(g.sum()),'n_expected_void_pixels':int(t.sum())}
                if dc is not None:
                    for k in ['slope','aspect']:rec['n_missing_'+k]=int((t&~np.isfinite(dc[k])).sum())
                    rec['n_missing_slope_or_aspect']=int((t&(~np.isfinite(dc['slope'])|~np.isfinite(dc['aspect']))).sum())
                rows.append(rec)
        except Exception as e:rows.append({**record,'status':'error','error':str(e)})
        if i%100==0:print(f'Validated {i}/{len(split)}',flush=True)
    a.output_dir.mkdir(parents=True,exist_ok=True);frame=pd.DataFrame(rows);frame.to_csv(a.output_dir/'input_coverage.csv',index=False)
    write_provenance(a.output_dir,a,__file__)
    print(frame.groupby(['group','status']).size().to_string())
    if (frame.status=='error').any():raise RuntimeError('Input validation errors; see input_coverage.csv')
if __name__=='__main__':main()
