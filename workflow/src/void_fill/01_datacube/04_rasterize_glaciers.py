#!/usr/bin/env python3
'Rasterize RGI outlines as exact int32 region×100000+glacier codes. The pixel-centre rule matches the glacier chips. Overlaps are counted and resolved deterministically.'
from __future__ import annotations
import argparse,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cubelib import Grid,load_config,work_path,geotiff_profile,strips,rgi_code,fingerprint,write_record,project_path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datacube-config');p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();cfg=load_config(a.datacube_config);grid=Grid.read(work_path(cfg,'grid.json'))
    import geopandas as gpd,pandas as pd,rasterio
    from rasterio.features import rasterize
    from rasterio.enums import MergeAlg
    from rasterio.windows import Window,bounds as window_bounds
    from rasterio.windows import transform as window_transform
    from shapely.geometry import box
    gc=cfg['glaciers'];src=project_path(gc['outlines'])
    out=work_path(cfg,'glaciers','rgi_code.tif');lookup=out.with_name('glaciers.csv')
    if out.exists() and not a.overwrite:raise FileExistsError(f'{out} exists; pass --overwrite')
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_name(out.stem+'.partial.tif')
    gdf=gpd.read_file(src)
    if gdf.crs is None:raise ValueError('Outlines have no CRS')
    if gdf.crs!=grid.crs:gdf=gdf.to_crs(grid.crs)
    gdf['rgiid']=gdf[gc.get('id_field','RGIId')].astype(str);gdf['code']=gdf.rgiid.map(rgi_code)
    if gdf.code.duplicated().any():raise ValueError(f'Duplicate glacier ids: {gdf.rgiid[gdf.code.duplicated()].tolist()[:5]}')
    if not gdf.geometry.is_valid.all():gdf['geometry']=gdf.geometry.make_valid()
    gdf=gdf.sort_values('code').reset_index(drop=True);sindex=gdf.sindex
    pixels=Counter();overlap=0
    with rasterio.open(tmp,'w',**geotiff_profile(grid,'int32',0)) as dst:
        for r,n in strips(grid.height,int(cfg.get('strip_rows',2048))):
            w=Window(0,r,grid.width,n);t=window_transform(w,grid.transform)
            idx=sindex.query(box(*window_bounds(w,grid.transform)))
            arr=np.zeros((n,grid.width),'int32')
            if len(idx):
                sub=gdf.iloc[np.sort(idx)]
                shapes=list(zip(sub.geometry,sub.code))
                arr=rasterize(shapes,out_shape=(n,grid.width),transform=t,fill=0,dtype='int32',all_touched=bool(gc.get('all_touched',False)))
                cnt=rasterize(((g,1) for g in sub.geometry),out_shape=(n,grid.width),transform=t,fill=0,dtype='uint8',
                              all_touched=bool(gc.get('all_touched',False)),merge_alg=MergeAlg.add)
                overlap+=int((cnt>1).sum())
                u,c=np.unique(arr[arr>0],return_counts=True);pixels.update(dict(zip(u.tolist(),c.tolist())))
            dst.write(arr,1,window=w)
    tmp.replace(out)
    table=pd.DataFrame({'rgiid':gdf.rgiid,'code':gdf.code,'n_pixels':gdf.code.map(lambda c:pixels.get(c,0)).astype('int64')})
    table['raster_area_km2']=table.n_pixels*grid.res**2/1e6
    if 'Area' in gdf:table['rgi_area_km2']=gdf['Area'].astype(float);table['area_ratio']=table.raster_area_km2/table.rgi_area_km2
    table.to_csv(lookup,index=False)
    record={'stage':'glaciers','outlines':fingerprint(src),'n_outlines':len(gdf),'n_glaciers_rasterized':int((table.n_pixels>0).sum()),
            'n_glaciers_without_pixels':int((table.n_pixels==0).sum()),'overlap_pixels':overlap,'all_touched':bool(gc.get('all_touched',False)),
            'encoding':'code = region*100000 + number; 0 = no glacier','area_ratio_quantiles':table.area_ratio.quantile([.01,.05,.5,.95,.99]).to_dict() if 'area_ratio' in table else None,
            'outputs':{'raster':str(out),'lookup':str(lookup)}}
    write_record(out.with_name('glaciers.json'),record,__file__)
    print(f"{record['n_glaciers_rasterized']:,} of {len(gdf):,} glaciers rasterized; {record['n_glaciers_without_pixels']} without a pixel centre; {overlap} overlap pixels")

if __name__=='__main__':main()
