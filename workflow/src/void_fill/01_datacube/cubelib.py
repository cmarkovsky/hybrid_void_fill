'Shared grid, raster, terrain, and provenance helpers. The cube uses the native reference dh/dt lattice with exactly 30 m pixels and lattice origins recorded to 1e-6 m.'
from __future__ import annotations
import hashlib,json,math,os,re,subprocess,sys
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

PACKAGE_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE_ROOT))
from workflow_support import project_path,project_root  # noqa: E402

RES_TOL=1e-6      # metres: pixel size must equal the target within this
PIXEL_TOL=1e-3    # pixels: lattice offsets must be integers within this (3 cm)
ORIGIN_DECIMALS=6 # lattice origin is stored rounded to 1 micrometre
RGI_PATTERN=re.compile(r'^RGI60-(\d{2})\.(\d{5})$')

def load_config(path=None):
    path=path or os.environ.get('VOID_FILL_DATACUBE_CONFIG')
    path=Path(path) if path else PACKAGE_ROOT/'config/datacube.json'
    if not path.is_absolute() and not path.exists():path=PACKAGE_ROOT/path
    cfg=json.loads(path.read_text());cfg['_config_path']=str(path.resolve());return cfg

def work_path(cfg,*parts):return project_path(cfg['work_dir']).joinpath(*parts)

@dataclass(frozen=True)
class Grid:
    crs_wkt:str
    res:float
    x0:float   # west edge of column 0
    y0:float   # north edge of row 0
    width:int
    height:int
    anchor:str='' # raster whose lattice defines the grid
    @property
    def transform(self):
        from affine import Affine
        return Affine(self.res,0.0,self.x0,0.0,-self.res,self.y0)
    @property
    def shape(self):return (self.height,self.width)
    @property
    def bounds(self):return (self.x0,self.y0-self.height*self.res,self.x0+self.width*self.res,self.y0)
    @property
    def crs(self):
        from rasterio.crs import CRS
        return CRS.from_wkt(self.crs_wkt)
    def x_centers(self):return self.x0+(np.arange(self.width,dtype='float64')+0.5)*self.res
    def y_centers(self):return self.y0-(np.arange(self.height,dtype='float64')+0.5)*self.res
    def geotransform_string(self):return ' '.join(repr(float(v)) for v in (self.x0,self.res,0.0,self.y0,0.0,-self.res))
    def write(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps({**asdict(self),'bounds':self.bounds,'geotransform':self.geotransform_string()},indent=2)+'\n')
    @classmethod
    def read(cls,path):
        d=json.loads(Path(path).read_text());return cls(**{k:d[k] for k in cls.__dataclass_fields__})

def canonical_lattice(transform,res):
    """Return (x0,y0) of a north-up raster whose pixel size equals res, else raise."""
    if transform.b or transform.d:raise ValueError('Rotated/sheared rasters are not supported')
    if abs(transform.a-res)>RES_TOL or abs(-transform.e-res)>RES_TOL:
        raise ValueError(f'Pixel size ({transform.a!r}, {-transform.e!r}) is not exactly {res} m')
    return round(transform.c,ORIGIN_DECIMALS),round(transform.f,ORIGIN_DECIMALS)

def lattice_offset(grid,transform):
    """Fractional (col,row) of a raster's origin on the grid lattice, and whether both are whole pixels."""
    col=(transform.c-grid.x0)/grid.res;row=(grid.y0-transform.f)/grid.res
    same_res=abs(transform.a-grid.res)<=RES_TOL and abs(-transform.e-grid.res)<=RES_TOL and not transform.b and not transform.d
    on=same_res and abs(col-round(col))<=PIXEL_TOL and abs(row-round(row))<=PIXEL_TOL
    return col,row,on

def snap_grid(anchor_x0,anchor_y0,anchor_width,anchor_height,res,bounds,buffer_m,crs_wkt,anchor=''):
    """Smallest whole-pixel sub-grid of the anchor lattice covering bounds+buffer (clipped to the anchor)."""
    xmin,ymin,xmax,ymax=bounds
    c0=max(0,math.floor((xmin-buffer_m-anchor_x0)/res));c1=min(anchor_width,math.ceil((xmax+buffer_m-anchor_x0)/res))
    r0=max(0,math.floor((anchor_y0-(ymax+buffer_m))/res));r1=min(anchor_height,math.ceil((anchor_y0-(ymin-buffer_m))/res))
    if c1<=c0 or r1<=r0:raise ValueError('Requested bounds do not intersect the anchor raster')
    return Grid(crs_wkt,float(res),round(anchor_x0+c0*res,ORIGIN_DECIMALS),round(anchor_y0-r0*res,ORIGIN_DECIMALS),c1-c0,r1-r0,anchor),(c0,r0)

def strips(height,rows):
    for start in range(0,height,rows):yield start,min(rows,height-start)

def rgi_code(rgiid):
    """RGI60-14.17287 -> 1417287 (exact integer; never via float)."""
    m=RGI_PATTERN.match(str(rgiid))
    if not m:raise ValueError(f'Unrecognized RGI id {rgiid!r}')
    return int(m.group(1))*100000+int(m.group(2))

def rgi_id(code):
    code=int(code);return f'RGI60-{code//100000:02d}.{code%100000:05d}'

def horn_slope_aspect(block,res):
    """Slope (deg) and aspect (deg clockwise from north, downslope azimuth) with Horn's 3x3 kernel.

    block carries a one-pixel halo on every side; outputs cover the interior.
    Any non-finite value in the 3x3 neighbourhood (or the centre) yields NaN,
    as gdaldem does without -compute_edges. Flat cells have undefined aspect (NaN).
    Rows run north->south, columns west->east.
    """
    z=np.asarray(block,dtype='float64')
    a,b,c=z[:-2,:-2],z[:-2,1:-1],z[:-2,2:]
    d,e,f=z[1:-1,:-2],z[1:-1,1:-1],z[1:-1,2:]
    g,h,i=z[2:,:-2],z[2:,1:-1],z[2:,2:]
    p=((c+2*f+i)-(a+2*d+g))/(8*res)   # dz/d(east)
    q=((a+2*b+c)-(g+2*h+i))/(8*res)   # dz/d(north)
    with np.errstate(invalid='ignore'):
        slope=np.degrees(np.arctan(np.hypot(p,q)))
        aspect=np.mod(np.degrees(np.arctan2(-p,-q)),360.0)
    bad=~np.isfinite(e)|~np.isfinite(slope)
    slope[bad]=np.nan;aspect[bad]=np.nan;aspect[(p==0)&(q==0)]=np.nan
    return slope.astype('float32'),aspect.astype('float32')

def geotiff_profile(grid,dtype='float32',nodata=None):
    floating=np.dtype(dtype).kind=='f'
    return {'driver':'GTiff','dtype':dtype,'count':1,'width':grid.width,'height':grid.height,'crs':grid.crs,
            'transform':grid.transform,'nodata':(np.nan if floating else 0) if nodata is None else nodata,
            'tiled':True,'blockxsize':512,'blockysize':512,'compress':'deflate','predictor':3 if floating else 2,
            'zlevel':4,'BIGTIFF':'YES','num_threads':'ALL_CPUS'}

def check_on_grid(path,grid):
    """Raise unless a raster sits exactly on the grid (same CRS, shape, transform)."""
    import rasterio
    with rasterio.open(path) as src:
        if src.crs is None or src.crs!=grid.crs:raise ValueError(f'{path}: CRS differs from the grid')
        if (src.height,src.width)!=grid.shape:raise ValueError(f'{path}: shape {src.shape} != grid {grid.shape}')
        col,row,on=lattice_offset(grid,src.transform)
        if not on or round(col) or round(row):raise ValueError(f'{path}: transform {tuple(src.transform)[:6]} is not the grid transform')

def fingerprint(path,full=False):
    """Size, mtime and SHA-256 (full, or of the first+last 16 MiB for large rasters)."""
    path=Path(path);st=path.stat();h=hashlib.sha256();chunk=16*1024*1024
    with path.open('rb') as f:
        if full or st.st_size<=2*chunk:
            for blk in iter(lambda:f.read(chunk),b''):h.update(blk)
        else:
            h.update(f.read(chunk));f.seek(-chunk,os.SEEK_END);h.update(f.read(chunk))
    return {'path':str(path),'bytes':st.st_size,'mtime_utc':datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat(),
            'sha256':h.hexdigest(),'sha256_scope':'full' if full or st.st_size<=2*chunk else 'first+last 16MiB'}

def write_record(path,record,source_file):
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'project_root':str(project_root()),
            'script':str(source_file),'script_sha256':fingerprint(source_file,full=True)['sha256'],**record}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(record,indent=2,default=str)+'\n')

def resolve_source(spec,work_dir,nodata=None):
    """A raster path, a directory of tiles, or a glob -> one readable path (tiles are mosaicked with gdalbuildvrt).

    Gaps between tiles must read as nodata, never 0: the mosaic is built with an explicit
    -vrtnodata (the layer's nodata, or NaN when the tiles define none).
    """
    src=str(spec)
    p=project_path(src) if not any(ch in src for ch in '*?[') else None
    if p is not None and p.is_file():return p
    if p is not None and p.is_dir():tiles=sorted(x for x in p.rglob('*') if x.suffix.lower() in {'.tif','.tiff','.vrt'})
    else:
        base=project_root();tiles=sorted(base.glob(src)) if not Path(src).is_absolute() else sorted(Path('/').glob(src.lstrip('/')))
    if not tiles:raise FileNotFoundError(f'No raster found for source {spec!r}')
    if len(tiles)==1:return tiles[0]
    vrt=Path(work_dir)/'mosaics'/(re.sub(r'[^A-Za-z0-9_.-]+','_',src).strip('_')[:80]+'.vrt');vrt.parent.mkdir(parents=True,exist_ok=True)
    listing=vrt.with_suffix('.txt');listing.write_text('\n'.join(map(str,tiles))+'\n')
    cmd=['gdalbuildvrt','-overwrite','-vrtnodata','nan' if nodata is None else repr(float(nodata))]
    if nodata is not None:cmd+=['-srcnodata',repr(float(nodata))]
    subprocess.run([*cmd,'-input_file_list',str(listing),str(vrt)],check=True)
    return vrt
