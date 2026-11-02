#!/usr/bin/env python3
"""Stage 5: assemble the aligned layers into a NEW Zarr datacube.

Every input raster must sit exactly on the grid (checked, never resampled here).
x/y are written from the grid definition (origin + (i + 0.5) * 30 m in float64),
so cell centres are exact lattice positions rather than coordinates inherited
from a stretched transform. Continuous layers are float32 with NaN missing;
rgi_code is int32 with 0 = no glacier (see glaciers/glaciers.csv). The output
path must not exist: the frozen manuscript cube is never overwritten.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cubelib import Grid,load_config,work_path,check_on_grid,write_record,project_path
import numpy as np

def layer_paths(cfg):
    paths={name:work_path(cfg,'aligned',f'{name}.tif') for name in cfg['layers']}
    paths.update(slope=work_path(cfg,'terrain','slope.tif'),aspect=work_path(cfg,'terrain','aspect.tif'),
                 rgi_code=work_path(cfg,'glaciers','rgi_code.tif'))
    return paths

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datacube-config');p.add_argument('--output',type=Path,help='Override the configured output path')
    a=p.parse_args();cfg=load_config(a.datacube_config);grid=Grid.read(work_path(cfg,'grid.json'))
    import rioxarray as rxr,xarray as xr
    out=project_path(a.output or cfg['output'])
    if out.exists():raise FileExistsError(f'{out} exists; choose a NEW cube path')
    paths=layer_paths(cfg);missing=[str(v) for v in paths.values() if not v.exists()]
    if missing:raise FileNotFoundError('Run the earlier stages first; missing '+', '.join(missing))
    for name,path in paths.items():check_on_grid(path,grid)
    x,y=grid.x_centers(),grid.y_centers();cs=int(cfg.get('chunk_size',2048))
    records={name:json.loads(path.with_suffix('.json').read_text()) for name,path in paths.items() if path.with_suffix('.json').exists()}
    records['terrain']=json.loads(work_path(cfg,'terrain','terrain.json').read_text())
    records['glaciers']=json.loads(work_path(cfg,'glaciers','glaciers.json').read_text())
    units=cfg['terrain'].get('units',{})
    meta={'slope':{'units':units.get('slope','degree'),'long_name':'Surface slope (Horn 3x3, 30 m)'},
          'aspect':{'units':units.get('aspect','degree'),'long_name':'Surface aspect, downslope azimuth clockwise from grid north (Horn 3x3)'},
          'rgi_code':{'long_name':'RGI 6.0 glacier code','encoding':'region*100000 + number; 0 = no glacier',
                      'lookup':str(work_path(cfg,'glaciers','glaciers.csv'))}}
    arrays={};opened=[]
    try:
        for name,path in paths.items():
            da=rxr.open_rasterio(path,chunks={'y':cs,'x':cs},lock=False).squeeze('band',drop=True);opened.append(da)
            if not (np.allclose(da.x.values,x,rtol=0,atol=1e-6) and np.allclose(da.y.values,y,rtol=0,atol=1e-6)):
                raise ValueError(f'{name}: pixel centres differ from the grid by more than 1e-6 m')
            spec=cfg['layers'].get(name,{})
            attrs={k:v for k,v in {**meta.get(name,{}),'units':spec.get('units',meta.get(name,{}).get('units')),
                   'long_name':spec.get('long_name',meta.get(name,{}).get('long_name'))}.items() if v is not None}
            arrays[name]=xr.DataArray(da.data,dims=('y','x'),coords={'y':y,'x':x},name=name,attrs=attrs)
        ds=xr.Dataset(arrays)
        # CF grid mapping written directly (rio.write_crs left finalizer errors at interpreter exit)
        from pyproj import CRS
        crs=CRS.from_wkt(grid.crs_wkt)
        ds=ds.assign_coords(spatial_ref=xr.DataArray(0,attrs={**crs.to_cf(),'crs_wkt':crs.to_wkt(),'spatial_ref':crs.to_wkt(),
                                                           'GeoTransform':grid.geotransform_string()}))
        for name in arrays:ds[name].attrs['grid_mapping']='spatial_ref'
        ds.x.attrs.update(units='m',long_name='x of pixel centre',axis='X');ds.y.attrs.update(units='m',long_name='y of pixel centre',axis='Y')
        dem_masked=bool(records.get('terrain',{}).get('dem_glacier_masked'))
        ds.attrs.update(title='HMA void-fill datacube (exact 30 m lattice)',grid_resolution_m=grid.res,
                        grid_origin_xy=[grid.x0,grid.y0],grid_shape=list(grid.shape),grid_anchor=grid.anchor,
                        layer_modes=json.dumps({k:v.get('mode') for k,v in records.items() if 'mode' in v}),
                        dem_glacier_masked=int(dem_masked),datacube_config=cfg['_config_path'])
        encoding={name:{'chunks':(cs,cs)} for name in arrays}
        print(f'Writing {out} ({len(arrays)} layers, {grid.height} x {grid.width})',flush=True)
        ds.to_zarr(out,mode='w-',consolidated=True,encoding=encoding)
    finally:
        for da in opened:da.close()
    write_record(out.parent/(out.name+'.provenance.json'),{'stage':'build','output':str(out),'grid':json.loads(work_path(cfg,'grid.json').read_text()),
                 'layers':{k:str(v) for k,v in paths.items()},'stage_records':records,'config':cfg},__file__)
    if dem_masked:print('WARNING: built from a glacier-masked DEM; slope/aspect are missing on glacier edges.')
    print('Wrote',out)

if __name__=='__main__':main()
