#!/usr/bin/env python3
"""Stage 6: validate a rebuilt datacube and test whether the frozen chips map onto it.

Checks:
  1. Structure: exact 30 m centre spacing, GeoTransform, CRS, dtypes.
  2. Coverage on glacier pixels (rgi_code > 0): dh/dt, z, slope, aspect, and the
     number of glacier pixels with dh/dt and z but no slope (the edge-ring defect).
  3. Glacier ids: rasterized glaciers versus the fixed split.
  4. Chips (optional): each frozen chip's x/y is mapped to the nearest lattice
     cell. A chip is "remappable" when the mapping is one-to-one along both axes
     and the chip's dh/dt equals the cube's dh/dt pixel-for-pixel; its masks can
     then be carried to the new grid without regenerating voids. The comparison
     uses the chip's finite dh/dt pixels; extra cube coverage is reported apart.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cubelib import Grid,load_config,work_path,rgi_code,write_record,project_path
from workflow_support import workflow_path
import numpy as np

def chip_mapping(chip,grid):
    fc=(np.asarray(chip['x'],'float64')-grid.x0)/grid.res-0.5;fr=(grid.y0-np.asarray(chip['y'],'float64'))/grid.res-0.5
    cols,rows=np.rint(fc).astype('int64'),np.rint(fr).astype('int64')
    return cols,rows,float(np.abs(fc-cols).max()*grid.res),float(np.abs(fr-rows).max()*grid.res)

def check_chip(path,ds,grid):
    with np.load(path) as z:chip={k:z[k] for k in ('x','y','dhdt','glacier_mask')}
    cols,rows,dx,dy=chip_mapping(chip,grid)
    one_to_one=bool(np.all(np.diff(cols)==1) and np.all(np.diff(rows)==1))
    inside=bool(cols.min()>=0 and rows.min()>=0 and cols.max()<grid.width and rows.max()<grid.height)
    rec={'rgiid':path.stem,'max_x_shift_m':dx,'max_y_shift_m':dy,'one_to_one':one_to_one,'inside_grid':inside}
    if not (one_to_one and inside):return {**rec,'dhdt_match':np.nan,'glacier_iou':np.nan,'remappable':False}
    win=ds.isel(y=slice(int(rows[0]),int(rows[-1])+1),x=slice(int(cols[0]),int(cols[-1])+1))
    cube_dh=win['dhdt'].values;code=win['rgi_code'].values
    a,b=np.asarray(chip['dhdt'],'float32'),cube_dh.astype('float32');have=np.isfinite(a)
    # every dh/dt value the chip holds must be the identical cube value (the new cube may cover more pixels)
    match=float(np.mean(a[have]==b[have])) if have.any() else 1.0
    rec['cube_only_dhdt_pixels']=int((~have&np.isfinite(b)).sum())
    g=np.asarray(chip['glacier_mask'],bool);c=code==rgi_code(path.stem)
    iou=float((g&c).sum()/max(1,(g|c).sum()))
    return {**rec,'dhdt_match':match,'glacier_iou':iou,'remappable':match>=0.999}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datacube-config');p.add_argument('--cube',type=Path,help='Override the configured cube path')
    p.add_argument('--chips',type=Path,help='Optional frozen chip directory')
    p.add_argument('--split',type=Path,default=workflow_path('split'))
    p.add_argument('--max-chips',type=int,default=500,help='Random sample size; 0 = every chip');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--skip-coverage',action='store_true',help='Skip the full-cube coverage pass')
    a=p.parse_args();cfg=load_config(a.datacube_config);grid=Grid.read(work_path(cfg,'grid.json'))
    import pandas as pd,xarray as xr
    cube=project_path(a.cube or cfg['output']);ds=xr.open_zarr(cube,consolidated=True)
    out=work_path(cfg,'validation');out.mkdir(parents=True,exist_ok=True)
    report={'cube':str(cube),'failures':[]}
    def require(ok,msg):
        if not ok:report['failures'].append(msg)
    x,y=ds.x.values,ds.y.values
    require(np.array_equal(x,grid.x_centers()) and np.array_equal(y,grid.y_centers()),'x/y are not the exact grid centres')
    require(np.allclose(np.diff(x),grid.res,rtol=0,atol=1e-6) and np.allclose(np.diff(y),-grid.res,rtol=0,atol=1e-6),'centre spacing is not exactly 30 m')
    require(ds.spatial_ref.attrs.get('GeoTransform')==grid.geotransform_string(),'GeoTransform differs from grid.json')
    for v in ('dhdt','z','slope','aspect'):require(str(ds[v].dtype)=='float32',f'{v} is {ds[v].dtype}, expected float32')
    require(str(ds['rgi_code'].dtype)=='int32','rgi_code is not int32')
    report['structure']={'shape':list(ds.dhdt.shape),'origin':[grid.x0,grid.y0],'geotransform':ds.spatial_ref.attrs.get('GeoTransform')}
    if not a.skip_coverage:
        import dask
        g=ds['rgi_code']>0;dh=ds.dhdt.notnull();zz=ds.z.notnull();sl=ds.slope.notnull();asp=ds.aspect.notnull()
        n=dask.compute(g.sum(),(g&dh).sum(),(g&zz).sum(),(g&sl).sum(),(g&asp).sum(),(g&dh&zz&~sl).sum(),zz.sum(),(ds.z==0).sum())
        n=[int(v) for v in n];gp=n[0] or 1
        total=ds.z.size;report['grid_coverage']={'z_valid_fraction':n[6]/total,'z_exact_zero_pixels':n[7]}
        require(n[7]<=0.001*total,f'{n[7]:,} z pixels are exactly 0 (tile gaps filled with 0?)')
        report['glacier_coverage']={'glacier_pixels':n[0],'dhdt':n[1]/gp,'z':n[2]/gp,'slope':n[3]/gp,'aspect':n[4]/gp,
                                    'pixels_with_dhdt_and_z_but_no_slope':n[5]}
        print('glacier coverage',json.dumps(report['glacier_coverage']))
    for name in cfg['layers']:
        csv=work_path(cfg,'aligned',f'{name}_out_of_range.csv')
        if csv.exists():
            o=pd.read_csv(csv);on=ds['rgi_code'].values[o.row.values,o.col.values]>0 if len(o) else np.array([],bool)
            report.setdefault('out_of_range',{})[name]={'pixels_set_to_nan':len(o),'on_glacier':int(on.sum())}
            require(not on.any(),f'{name}: {int(on.sum())} out-of-range values were removed from glacier pixels')
    lookup=pd.read_csv(work_path(cfg,'glaciers','glaciers.csv'))
    split_path=project_path(a.split)
    if split_path.exists():
        split=json.loads(split_path.read_text());present=set(lookup.rgiid[lookup.n_pixels>0])
        absent=sorted(set(split)-present);report['split']={'glaciers':len(split),'absent_from_raster':len(absent),'examples':absent[:10]}
        require(not absent,f'{len(absent)} split glaciers have no rasterized pixel')
    if a.chips:
        files=sorted(project_path(a.chips).glob('*.npz'))
        if a.max_chips and len(files)>a.max_chips:files=list(np.random.default_rng(a.seed).choice(files,a.max_chips,replace=False))
        rows=[]
        for i,f in enumerate(files,1):
            rows.append(check_chip(Path(f),ds,grid))
            if i%100==0:print(f'  checked {i}/{len(files)} chips',flush=True)
        table=pd.DataFrame(rows);table.to_csv(out/'chip_remap.csv',index=False)
        report['chips']={'checked':len(table),'remappable':int(table.remappable.sum()),'one_to_one':int(table.one_to_one.sum()),
                         'max_shift_m':float(table[['max_x_shift_m','max_y_shift_m']].max().max()),
                         'median_dhdt_match':float(table.dhdt_match.median()),'median_glacier_iou':float(table.glacier_iou.median())}
        print('chips',json.dumps(report['chips']))
    report['passed']=not report['failures']
    write_record(out/'validation.json',report,__file__)
    print('PASSED' if report['passed'] else 'FAILED:\n  '+'\n  '.join(report['failures']));print('Wrote',out/'validation.json')
    if not report['passed']:sys.exit(1)

if __name__=='__main__':main()
