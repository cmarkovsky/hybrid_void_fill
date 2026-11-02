#!/usr/bin/env python3
'Draw Figure S3 from the archived geometry-selected reconstruction arrays.'
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,ListedColormap
from matplotlib.patches import Patch,Rectangle
from matplotlib.colorbar import ColorbarBase
parser=argparse.ArgumentParser()
parser.add_argument('--source-dir',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent/'figures/figS03_reconstruction_examples.pdf')
args=parser.parse_args()
RAW=args.source_dir.resolve()
case_order=[('ehim_low','RGI60-15.01282','Eastern Himalaya'),('ehim_high','RGI60-15.01453','Eastern Himalaya'),('whim_low','RGI60-14.16513','Western Himalaya'),('whim_high','RGI60-14.16787','Western Himalaya')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.5,'savefig.facecolor':'white'})
W,H=7.15,8.15
fig=plt.figure(figsize=(W,H),facecolor='white')
left,right,gap,group_gap=.12,.10,.085,.18
panel_w=(W-left-right-7*gap-group_gap)/8
panel_h=1.07
xs=[left+i*(panel_w+gap)+(group_gap if i>=5 else 0) for i in range(8)]
titles=['Reference','Observed','Constant','Hypsometric','Hybrid','Constant\nerror','Hypsometric\nerror','Hybrid\nerror']
for col,title in enumerate(titles):
    fig.text((xs[col]+panel_w/2)/W,7.98/H,title,ha='center',va='center',fontsize=8.6,fontweight='medium',linespacing=1.12)
selection=pd.read_csv(RAW/'case_selection.csv').set_index('case')
saturation=pd.read_csv(RAW/'spatial_examples_saturation.csv')
field_cmap=matplotlib.colormaps['RdBu'].copy();field_cmap.set_bad((1,1,1,0))
error_cmap=matplotlib.colormaps['PuOr_r'].copy();error_cmap.set_bad((1,1,1,0))
native_color='#bdbdbd';observed_color='#eeeeee';target_color='#f4c247'
all_axes=[];checks=[]

def add_ax(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
def label(v):
    if abs(v)<1e-12:return '0'
    return format(v,'.3g').replace('-','\N{MINUS SIGN}')

for row,(case,rgi,region) in enumerate(case_order):
    path=RAW/f'plot_data_{case}_{rgi}.npz'
    with np.load(path,allow_pickle=False) as loaded:d={k:loaded[k] for k in loaded.files}
    g=d['glacier'];valid=d['valid_support'];targets=d['targets'];void=d['void']
    x=d['x'];y=d['y'];dx=abs(float(np.median(np.diff(x))));dy=abs(float(np.median(np.diff(y))))
    assert np.isclose(dx,30) and np.isclose(dy,30)
    origin='lower' if y[0]<y[-1] else 'upper'
    if x[0]>x[-1]:
        d={k:np.fliplr(a) if a.ndim==2 else a for k,a in d.items()};g=d['glacier'];valid=d['valid_support'];targets=d['targets'];void=d['void']
    extent=(float(x.min()-dx/2),float(x.max()+dx/2),float(y.min()-dy/2),float(y.max()+dy/2))
    gx=np.broadcast_to(x,g.shape)[g];gy=np.broadcast_to(y[:,None],g.shape)[g]
    # Padding below the map reserves a clear band for the 500-m scale bar.
    xmin,xmax=float(gx.min()-dx/2-70),float(gx.max()+dx/2+70)
    ymin,ymax=float(gy.min()-dy/2-200),float(gy.max()+dy/2+90)
    xc,yc=(xmin+xmax)/2,(ymin+ymax)/2
    xspan,yspan=xmax-xmin,ymax-ymin
    ratio=panel_w/panel_h
    if xspan/yspan>ratio:yspan=xspan/ratio
    else:xspan=yspan*ratio
    xlim=(xc-xspan/2,xc+xspan/2);ylim=(yc-yspan/2,yc+yspan/2)
    bottom=6.33-row*1.85
    header_y=bottom+panel_h+.18
    fig.text(left/W,header_y/H,f'({chr(97+row)})  {region} | {rgi}',fontsize=8.5,fontweight='bold',ha='left',va='center')
    f=float(selection.loc[case,'achieved_frac']);u=float(selection.loc[case,'mean_u_void'])
    fig.text((W-right)/W,header_y/H,f'Void fraction {f:.3f} | normalized elevation {u:.3f}',fontsize=7.8,ha='right',va='center',color='#303030')
    vlim=float(d['dhdt_limit']);elim=float(d['error_limit'])
    norm_v=Normalize(-vlim,vlim);norm_e=Normalize(-elim,elim)
    row_axes=[]
    for col in range(8):
        ax=add_ax(xs[col],bottom,panel_w,panel_h);row_axes.append(ax);all_axes.append(ax)
        ax.set_facecolor('white');ax.set_xticks([]);ax.set_yticks([])
        base=np.where(g&~valid,1.,np.nan)
        ax.imshow(np.ma.masked_invalid(base),extent=extent,origin=origin,cmap=ListedColormap([native_color]),vmin=0,vmax=1,interpolation='nearest')
        if col<5:
            key=['reference','reference','reconstruction_constant','reconstruction_hypsometric','reconstruction_hybrid'][col]
            mask=valid&~targets if col==1 else valid
            data=np.where(mask,d[key],np.nan)
            ax.imshow(np.ma.masked_invalid(data),extent=extent,origin=origin,cmap=field_cmap,norm=norm_v,interpolation='nearest')
            if col==1:
                overlay=np.where(targets,1.,np.nan)
                ax.imshow(np.ma.masked_invalid(overlay),extent=extent,origin=origin,cmap=ListedColormap([target_color]),vmin=0,vmax=1,interpolation='nearest')
                # Present in the legend for consistent interpretation of native missing cells.
                yy,xx=np.nonzero(void&g&~valid)
                for iy,ix in zip(yy,xx):
                    ax.add_patch(Rectangle((x[ix]-dx/2,y[iy]-dy/2),dx,dy,facecolor='white',edgecolor='#404040',hatch='////',linewidth=0))
        else:
            background=np.where(valid&~targets,1.,np.nan)
            ax.imshow(np.ma.masked_invalid(background),extent=extent,origin=origin,cmap=ListedColormap([observed_color]),vmin=0,vmax=1,interpolation='nearest')
            key=['error_constant','error_hypsometric','error_hybrid'][col-5]
            ax.imshow(np.ma.masked_invalid(np.where(targets,d[key],np.nan)),extent=extent,origin=origin,cmap=error_cmap,norm=norm_e,interpolation='nearest')
        ax.set_xlim(xlim);ax.set_ylim(ylim);ax.set_aspect('equal',adjustable='box')
        for spine in ax.spines.values():spine.set_color('#555555');spine.set_linewidth(.45)
        if col==0:
            bar_x=xlim[0]+.075*xspan;bar_y=ylim[0]+.025*yspan
            ax.plot([bar_x,bar_x+500],[bar_y,bar_y],color='black',lw=1.2,solid_capstyle='butt',clip_on=False)
            ax.text(bar_x+250,ylim[0]+.067*yspan,'500 m',ha='center',va='center',fontsize=6.4,color='black')
            ax.annotate('',xy=(.91,.91),xytext=(.91,.79),xycoords='axes fraction',arrowprops={'arrowstyle':'-|>','lw':.65,'color':'black','mutation_scale':6})
            ax.text(.91,.95,'N',transform=ax.transAxes,ha='center',va='center',fontsize=6.5)
    # Full-width horizontal colorbars avoid compressing the map columns.
    for start,stop,norm,cmap,title in [(0,4,norm_v,field_cmap,r'Elevation-change rate (m a$^{-1}$)'),(5,7,norm_e,error_cmap,r'Prediction $-$ reference (m a$^{-1}$)')]:
        x0=xs[start]+.10;x1=xs[stop]+panel_w-.10
        cax=add_ax(x0,bottom-.20,x1-x0,.065)
        cb=ColorbarBase(cax,cmap=cmap,norm=norm,orientation='horizontal',extend='both',extendfrac=.025)
        lim=norm.vmax;ticks=[-lim,-lim/2,0,lim/2,lim]
        cb.set_ticks(ticks);cb.set_ticklabels([label(t) for t in ticks]);cb.ax.tick_params(labelsize=7.5,length=2,pad=1)
        cb.set_label(title,fontsize=8.2,labelpad=1)
        cb.outline.set_linewidth(.45)
    # Validate the reconstructed values, observed cells, and archived clipping counts.
    rowcheck={'case':case,'rgiid':rgi,'dhdt_limit':vlim,'error_limit':elim,'xlim':list(xlim),'ylim':list(ylim),'native_missing':int(np.count_nonzero(g&~valid)),'targets':int(np.count_nonzero(targets)),'saturation_counts':{}}
    for method in ['constant','hypsometric','hybrid']:
        recon=d['reconstruction_'+method];error=d['error_'+method]
        assert np.allclose(recon[valid&~targets],d['reference'][valid&~targets],atol=1e-10)
        assert np.allclose(error[targets],(recon-d['reference'])[targets],atol=1e-10)
    panel_keys=[('Reference','reference',valid,vlim),('Masked observations','reference',valid&~targets,vlim),('Constant','reconstruction_constant',valid,vlim),('Hypsometric','reconstruction_hypsometric',valid,vlim),('Hybrid XGBoost','reconstruction_hybrid',valid,vlim),('Constant − reference','error_constant',targets,elim),('Hypsometric − reference','error_hypsometric',targets,elim),('Hybrid XGBoost − reference','error_hybrid',targets,elim)]
    for panel,key,mask,lim in panel_keys:
        count=int(np.count_nonzero(np.abs(d[key][mask])>lim))
        expected=int(saturation[(saturation['case']==case)&(saturation.panel==panel)].iloc[0].saturated)
        assert count==expected,(case,panel,count,expected)
        rowcheck['saturation_counts'][panel]=count
    checks.append(rowcheck)

legend=[Patch(facecolor=target_color,label='Withheld'),Patch(facecolor=observed_color,label='Observed (error panels)'),Patch(facecolor=native_color,label='No reference'),Patch(facecolor='white',edgecolor='#555555',hatch='////',label='Masked, no reference')]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.003),ncol=4,frameon=False,fontsize=7.5,handlelength=1.6,handleheight=.9,columnspacing=1.5,handletextpad=.55)
fig.canvas.draw()
for i,ax in enumerate(all_axes):
    row,col=divmod(i,8)
    expected=np.array([xs[col]/W,(6.33-row*1.85)/H,panel_w/W,panel_h/H])
    assert np.allclose(ax.get_position().bounds,expected,atol=1e-9),(i,ax.get_position().bounds,expected)

dest=args.output.resolve();dest.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(dest,dpi=600,metadata={'Title':'Figure S3 - Reconstruction examples','Author':'Cameron Markovsky'})
fig.savefig(dest.with_suffix('.png'),dpi=600)
plt.close(fig)
(dest.parent/'figS03_checks.json').write_text(json.dumps({'all_32_map_axes_aligned':True,'archived_color_limits_retained':True,'cases':checks},indent=2)+'\n')
print(dest)
