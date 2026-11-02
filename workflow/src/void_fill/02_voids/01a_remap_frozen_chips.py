#!/usr/bin/env python3
'Remap frozen glacier chips onto the aligned cube by one-to-one nearest lattice indices. Glacier and void masks, reference values, and realization identities remain fixed; coordinates and DEM elevations come from the aligned cube. Exact reference-value agreement is required.'
from __future__ import annotations
import argparse,hashlib,json,os,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from workflow_support import workflow_path,parse_stage_args,write_provenance,project_path
import numpy as np,pandas as pd

KEEP=('glacier_mask','void_mask','dhdt','is_never_void','attempted_realization_index')
DROP=('debris_thickness','has_debris')

def lattice_indices(values,origin,step):
    """Nearest lattice index for coordinates; returns (indices, max |offset| in metres)."""
    f=(np.asarray(values,'float64')-origin)/step;i=np.rint(f).astype('int64')
    return i,float(np.abs(f-i).max()*abs(step))

def remap_chip(chip,X,Y,load_window,z_source='cube'):
    """Return (new chip arrays, manifest record). load_window(rows,cols) -> (dhdt, z) cube arrays."""
    res=float(X[1]-X[0])
    cols,sx=lattice_indices(chip['x'],X[0],res);rows,sy=lattice_indices(chip['y'],Y[0],-res)
    if not (np.all(np.diff(cols)==1) and np.all(np.diff(rows)==1)):raise ValueError('chip does not map one-to-one onto the lattice')
    if cols[0]<0 or rows[0]<0 or cols[-1]>=len(X) or rows[-1]>=len(Y):raise ValueError('chip extends beyond the cube')
    if max(sx,sy)>=0.5*res:raise ValueError(f'chip offset {max(sx,sy):.2f} m is not below half a pixel')
    rs,cs=slice(int(rows[0]),int(rows[-1])+1),slice(int(cols[0]),int(cols[-1])+1)
    dh_cube,z_cube=load_window(rs,cs)
    dh=np.asarray(chip['dhdt'],'float32');have=np.isfinite(dh)
    if not np.array_equal(dh[have],np.asarray(dh_cube,'float32')[have]):
        raise ValueError(f'{int((dh[have]!=np.asarray(dh_cube,"float32")[have]).sum())} chip dh/dt values differ from the cube at the mapped pixels')
    glacier=np.asarray(chip['glacier_mask'],bool);targets=np.asarray(chip['void_mask'],bool)&(glacier&have)
    z_old=np.asarray(chip['z'],'float32');z_new=np.asarray(z_cube,'float32') if z_source=='cube' else z_old
    support=glacier&have;dz=(z_new-z_old)[support&np.isfinite(z_old)&np.isfinite(z_new)]
    out={k:np.asarray(chip[k]) for k in KEEP if k in chip}
    out.update(x=np.asarray(X[cs],'float64'),y=np.asarray(Y[rs],'float64'),z=z_new)
    for k in chip:
        if k not in out and k not in DROP and k not in ('x','y','z'):out[k]=np.asarray(chip[k])
    record={'row_offset':int(rows[0]),'col_offset':int(cols[0]),'height':len(rows),'width':len(cols),
            'max_shift_m':max(sx,sy),'dhdt_pixels_checked':int(have.sum()),
            'cube_only_dhdt_glacier_pixels':int((glacier&~have&np.isfinite(dh_cube)).sum()),
            'support_pixels':int(support.sum()),'support_missing_z':int((support&~np.isfinite(z_new)).sum()),
            'target_pixels_missing_z':int((targets.any(0)&~np.isfinite(z_new)).sum()),
            'z_median_abs_change_m':float(np.median(np.abs(dz))) if dz.size else np.nan,
            'z_mean_change_m':float(dz.mean()) if dz.size else np.nan,'z_source':z_source}
    return out,record

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-chips',type=Path,default=workflow_path('frozen_chips'),help='Frozen chips (read only)')
    p.add_argument('--cube',type=Path,default=workflow_path('cube'));p.add_argument('--output',type=Path,default=workflow_path('chips'))
    p.add_argument('--z-source',choices=('cube','chip'),default='cube');p.add_argument('--workers',type=int,default=8)
    a=parse_stage_args(p);src=project_path(a.source_chips)
    if src.resolve()==a.output.resolve():raise ValueError('Output must differ from the frozen chip directory')
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(f'{a.output} is not empty')
    import xarray as xr
    ds=xr.open_zarr(a.cube,consolidated=True);X,Y=ds.x.values,ds.y.values
    if not (np.allclose(np.diff(X),X[1]-X[0],rtol=0,atol=1e-6) and np.allclose(np.diff(Y),X[0]-X[1],rtol=0,atol=1e-6)):
        raise ValueError('Cube is not a regular north-up lattice')
    sub=ds[['dhdt','z']]
    def load_window(rs,cs):
        w=sub.isel(y=rs,x=cs).load();return w.dhdt.values,w.z.values
    files=sorted(src.glob('*.npz'));a.output.mkdir(parents=True,exist_ok=True)
    def one(path):
        rec={'rgiid':path.stem,'source_sha256':sha256(path)}
        try:
            with np.load(path,allow_pickle=False) as z:chip={k:z[k] for k in z.files}
            out,info=remap_chip(chip,X,Y,load_window,a.z_source)
            dst=a.output/path.name;np.savez_compressed(dst,**out)
            return {**rec,**info,'status':'ok','output_sha256':sha256(dst)}
        except Exception as e:return {**rec,'status':'error','error':f'{type(e).__name__}: {e}'}
    rows=[]
    with ThreadPoolExecutor(a.workers) as pool:
        for i,r in enumerate(pool.map(one,files),1):
            rows.append(r)
            if i%1000==0:print(f'  {i}/{len(files)} chips',flush=True)
    ds.close()
    manifest=pd.DataFrame(rows);manifest.to_csv(a.output.parent/'remap_manifest.csv',index=False)
    write_provenance(a.output.parent,a,__file__)
    ok=manifest[manifest.status=='ok']
    summary={'chips':len(manifest),'remapped':len(ok),'errors':int((manifest.status=='error').sum()),
             'max_shift_m':float(ok.max_shift_m.max()) if len(ok) else None,
             'chips_with_targets_missing_z':int((ok.target_pixels_missing_z>0).sum()) if len(ok) else None,
             'z_median_abs_change_m':float(ok.z_median_abs_change_m.median()) if len(ok) else None}
    (a.output.parent/'remap_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
    if summary['errors']:raise RuntimeError('Some chips failed; see remap_manifest.csv (failed chips were not written)')
if __name__=='__main__':main()
