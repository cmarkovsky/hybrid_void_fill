#!/usr/bin/env python3
"""Stage 1: define the datacube grid as a whole-pixel window of the anchor (dh/dt) lattice.

The extent is the bounding box of the glacier outlines plus a buffer, snapped
OUTWARD to whole anchor pixels. Pixel size stays exactly 30 m; nothing is
stretched to hit an exact extent. The buffer leaves room for chip padding and
for the 3x3 slope/aspect kernel at the outermost glaciers.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cubelib import load_config,work_path,canonical_lattice,snap_grid,write_record,fingerprint,project_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datacube-config',help='Datacube profile (default config/datacube.json)')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();cfg=load_config(a.datacube_config)
    import rasterio,geopandas as gpd
    gcfg=cfg['grid'];anchor=cfg['layers'][gcfg['anchor']];res=float(gcfg['resolution_m'])
    out=work_path(cfg,'grid.json')
    if out.exists() and not a.overwrite:raise FileExistsError(f'{out} exists; pass --overwrite to redefine the grid')
    anchor_path=project_path(anchor['source'])
    with rasterio.open(anchor_path) as src:
        if src.crs is None or not src.crs.is_projected:raise ValueError('Anchor must be in a projected metre CRS')
        ax0,ay0=canonical_lattice(src.transform,res)
        crs=src.crs;aw,ah=src.width,src.height
    outlines=gpd.read_file(project_path(gcfg['extent_from']))
    if outlines.crs is None:raise ValueError('Outlines have no CRS')
    if outlines.crs!=crs:outlines=outlines.to_crs(crs)
    grid,(c0,r0)=snap_grid(ax0,ay0,aw,ah,res,tuple(outlines.total_bounds),float(gcfg['buffer_m']),crs.to_wkt(),str(anchor_path))
    grid.write(out)
    write_record(work_path(cfg,'grid.provenance.json'),{'stage':'define_grid','config':cfg,
        'anchor':fingerprint(anchor_path),'anchor_lattice_origin':[ax0,ay0],'anchor_shape':[ah,aw],
        'window_in_anchor':{'col_off':c0,'row_off':r0,'width':grid.width,'height':grid.height},
        'outline_bounds':[float(v) for v in outlines.total_bounds],'grid':str(out)},__file__)
    print(f'Grid {grid.height} x {grid.width} at {res} m; origin ({grid.x0!r}, {grid.y0!r}); anchor offset col {c0}, row {r0}')
    print('Wrote',out)

if __name__=='__main__':main()
