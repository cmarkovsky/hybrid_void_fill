#!/usr/bin/env python3
'Compute Horn 3×3 slope and aspect from the unmasked aligned DEM. Glacier masking is applied after terrain derivatives so boundary pixels retain their neighbourhood.'
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cubelib import Grid,load_config,work_path,horn_slope_aspect,geotiff_profile,strips,check_on_grid,fingerprint,write_record
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datacube-config');p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();cfg=load_config(a.datacube_config);grid=Grid.read(work_path(cfg,'grid.json'))
    import rasterio
    from rasterio.windows import Window
    zname=cfg['terrain']['from'];zpath=work_path(cfg,'aligned',f'{zname}.tif')
    zrec=json.loads(zpath.with_suffix('.json').read_text());check_on_grid(zpath,grid)
    outs={k:work_path(cfg,'terrain',f'{k}.tif') for k in ('slope','aspect')}
    for o in outs.values():
        if o.exists() and not a.overwrite:raise FileExistsError(f'{o} exists; pass --overwrite')
        o.parent.mkdir(parents=True,exist_ok=True)
    tmps={k:o.with_name(o.stem+'.partial.tif') for k,o in outs.items()}
    counts={'dem_valid':0,'slope_valid':0,'aspect_valid':0,'dem_valid_slope_missing':0}
    with rasterio.open(zpath) as z,rasterio.open(tmps['slope'],'w',**geotiff_profile(grid)) as ds,rasterio.open(tmps['aspect'],'w',**geotiff_profile(grid)) as da:
        for r,n in strips(grid.height,int(cfg.get('strip_rows',2048))):
            block=z.read(1,window=Window(-1,r-1,grid.width+2,n+2),boundless=True,fill_value=np.nan).astype('float32')
            slope,aspect=horn_slope_aspect(block,grid.res)
            dem=np.isfinite(block[1:-1,1:-1]);sv=np.isfinite(slope)
            counts['dem_valid']+=int(dem.sum());counts['slope_valid']+=int(sv.sum());counts['aspect_valid']+=int(np.isfinite(aspect).sum())
            counts['dem_valid_slope_missing']+=int((dem&~sv).sum())
            w=Window(0,r,grid.width,n);ds.write(slope,1,window=w);da.write(aspect,1,window=w)
    for k in outs:tmps[k].replace(outs[k])
    loss=counts['dem_valid_slope_missing']/counts['dem_valid'] if counts['dem_valid'] else None
    record={'stage':'terrain','method':'Horn 3x3 (gdaldem-equivalent, no edge computation)','pixel_size_m':grid.res,
            'dem':fingerprint(zpath),'dem_glacier_masked':zrec.get('glacier_masked',False),'counts':counts,
            'dem_valid_fraction_without_slope':loss,'units':cfg['terrain'].get('units'),'outputs':{k:str(v) for k,v in outs.items()}}
    write_record(work_path(cfg,'terrain','terrain.json'),record,__file__)
    print(f"slope valid {counts['slope_valid']:,} of {counts['dem_valid']:,} DEM pixels; {loss:.3%} of DEM pixels lack slope")
    if zrec.get('glacier_masked'):print('WARNING: DEM is glacier-masked, so glacier-edge pixels have no slope/aspect. Supply an unmasked DEM to fix this.')

if __name__=='__main__':main()
