#!/usr/bin/env python3
'Align configured raster layers to the exact 30 m grid. Lattice-compatible targets are cropped by whole pixels; the DEM is warped with bilinear resampling. Configured invalid ranges are checked and reported.'
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cubelib import (Grid,load_config,work_path,lattice_offset,resolve_source,geotiff_profile,strips,
                     fingerprint,write_record)
import numpy as np

def align_layer(name,spec,grid,cfg,overwrite=False,error_threshold=0.01):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window
    out=work_path(cfg,'aligned',f'{name}.tif')
    if out.exists() and not overwrite:raise FileExistsError(f'{out} exists; pass --overwrite')
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_name(out.stem+'.partial.tif')
    src_path=resolve_source(spec['source'],work_path(cfg),spec.get('nodata'))
    lo,hi=spec.get('valid_range',[-math.inf,math.inf])
    policy=spec.get('out_of_range','error')
    if policy not in {'error','nan'}:raise ValueError(f'{name}: out_of_range must be "error" or "nan"')
    stats={'n_valid':0,'sum':0.0,'min':math.inf,'max':-math.inf,'n_out_of_range':0};outliers=[]
    with rasterio.open(src_path) as src:
        nodata=spec.get('nodata');nodata=src.nodata if nodata is None else nodata
        col,row,on=lattice_offset(grid,src.transform) if src.crs==grid.crs else (math.nan,math.nan,False)
        if spec.get('require_lattice') and not on:
            raise ValueError(f'{name}: source is not on the grid lattice (offset col={col}, row={row}); refusing to resample it')
        mode='lattice_crop' if on else 'warp'
        vrt=None
        if mode=='warp':
            vrt=WarpedVRT(src,crs=grid.crs,transform=grid.transform,width=grid.width,height=grid.height,
                          resampling=Resampling[spec['resampling']],src_nodata=nodata,nodata=np.nan,dtype='float32',
                          tolerance=error_threshold,warp_mem_limit=1024,NUM_THREADS='ALL_CPUS')
        try:
            with rasterio.open(tmp,'w',**geotiff_profile(grid)) as dst:
                for r,n in strips(grid.height,int(cfg.get('strip_rows',2048))):
                    if mode=='lattice_crop':
                        # col/row locate the SOURCE origin on the grid; the grid starts at -col/-row in the source
                        w=Window(-int(round(col)),-int(round(row))+r,grid.width,n)
                        arr=src.read(1,window=w,boundless=True,fill_value=nodata if nodata is not None else np.nan).astype('float32')
                    else:
                        arr=vrt.read(1,window=Window(0,r,grid.width,n)).astype('float32')
                    if nodata is not None and not (isinstance(nodata,float) and math.isnan(nodata)):arr[arr==np.float32(nodata)]=np.nan
                    arr[~np.isfinite(arr)]=np.nan
                    with np.errstate(invalid='ignore'):bad=(arr<lo)|(arr>hi)
                    if bad.any():
                        stats['n_out_of_range']+=int(bad.sum())
                        rr,cc=np.nonzero(bad);outliers.extend(zip((rr+r).tolist(),cc.tolist(),arr[bad].astype(float).tolist()))
                        if policy=='nan':arr[bad]=np.nan
                    v=arr[np.isfinite(arr)]
                    if v.size:
                        stats['n_valid']+=int(v.size);stats['sum']+=float(v.sum(dtype='float64'))
                        stats['min']=min(stats['min'],float(v.min()));stats['max']=max(stats['max'],float(v.max()))
                    dst.write(arr,1,window=Window(0,r,grid.width,n))
        finally:
            if vrt is not None:vrt.close()
    if stats['n_out_of_range'] and policy=='error':
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{name}: {stats['n_out_of_range']:,} finite values outside valid_range {lo}..{hi} "
                         f"(e.g. {[round(o[2],4) for o in outliers[:5]]}); check units/nodata of {src_path}, or set out_of_range to \"nan\"")
    tmp.replace(out)
    outlier_csv=out.with_name(f'{name}_out_of_range.csv')
    if outliers:
        import pandas as pd
        o=pd.DataFrame(outliers,columns=['row','col','value']);o['x']=grid.x0+(o.col+0.5)*grid.res;o['y']=grid.y0-(o.row+0.5)*grid.res
        o.to_csv(outlier_csv,index=False)
    else:outlier_csv.unlink(missing_ok=True)
    stats['mean']=stats['sum']/stats['n_valid'] if stats['n_valid'] else None;stats.pop('sum')
    stats['valid_fraction']=stats['n_valid']/(grid.width*grid.height)
    record={'stage':'align','layer':name,'mode':mode,'source':fingerprint(src_path),'source_spec':spec,
            'lattice_offset_px':[col,row],'resampling':None if mode=='lattice_crop' else spec['resampling'],
            'warp_error_threshold_px':None if mode=='lattice_crop' else error_threshold,
            'nodata_used':nodata,'out_of_range_policy':policy,'out_of_range_csv':str(outlier_csv) if outliers else None,'glacier_masked':bool(spec.get('glacier_masked',False)),'stats':stats,'output':str(out)}
    write_record(out.with_suffix('.json'),record,__file__)
    print(f"{name}: {mode}; valid {stats['n_valid']:,} px ({stats['valid_fraction']:.4%}); range {stats['min']:.4g}..{stats['max']:.4g}"
          +(f"; {stats['n_out_of_range']:,} out-of-range values set to NaN" if stats['n_out_of_range'] else ''))
    return record

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datacube-config');p.add_argument('--layers',nargs='+',help='Subset of configured layers (default: all)')
    p.add_argument('--warp-error-threshold',type=float,default=0.01,help='GDAL approximate-transformer error in pixels for warped layers')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();cfg=load_config(a.datacube_config);grid=Grid.read(work_path(cfg,'grid.json'))
    names=a.layers or list(cfg['layers'])
    unknown=set(names)-set(cfg['layers'])
    if unknown:raise KeyError(f'Unknown layers {sorted(unknown)}')
    for name in names:align_layer(name,cfg['layers'][name],grid,cfg,a.overwrite,a.warp_error_threshold)

if __name__=='__main__':main()
