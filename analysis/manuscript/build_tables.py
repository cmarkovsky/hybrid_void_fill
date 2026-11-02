#!/usr/bin/env python3
'Build manuscript Tables 5–10 from the default evaluation evidence, excluding RGI60-14.17287.'
from pathlib import Path
from decimal import Decimal, ROUND_HALF_EVEN
import pandas as pd
import numpy as np
import json

import argparse
_ap=argparse.ArgumentParser(description='Build manuscript Tables 5-10 (excluding RGI60-14.17287) from recalculated evidence. '
    'Uses the default manuscript evaluation.')
_ap.add_argument('--evidence',type=Path,required=True,help='directory with point_metrics.csv, main_paired_intervals.csv, transfer_metrics.csv, coordinate_all_metrics.csv')
_ap.add_argument('--out',type=Path,required=True)
_args=_ap.parse_args()
SRC=_args.evidence
OUT=_args.out
(OUT/'tables').mkdir(parents=True,exist_ok=True)
points=pd.read_csv(SRC/'point_metrics.csv').query('excluded').copy()
ci=pd.read_csv(SRC/'main_paired_intervals.csv').query('excluded and size_km==100').copy()
transfer=pd.read_csv(SRC/'transfer_metrics.csv')
coords=pd.read_csv(SRC/'coordinate_all_metrics.csv').query('excluded').copy()

REGIONS=[('whim','Western'),('ehim','Eastern'),('all','Combined')]
METHODS=[('constant','Constant'),('hypsometric','Hypsometric'),('xgb','Hybrid')]
CAPS={
4:r'Glacier-balanced pixel reconstruction errors on identical withheld targets (m a$^{-1}$). Each glacier receives equal total weight, divided equally among its realizations and then their target pixels. Bias is predicted minus reference elevation-change rate. Western and Eastern counts are 2,130,058 and 410,664 target instances.',
5:r'Hybrid-minus-hypsometric error contrasts with 95\% percentile intervals in brackets (m a$^{-1}$). Intervals use 2,000 paired spatial-block draws with 100\,km blocks. Negative values favor hybrid filling. Bias contrasts subtract the magnitudes of population mean signed errors.',
6:r'Glacier-wide specific reconstruction errors over valid reference-data area (m a$^{-1}$). Absolute, squared, and signed errors are averaged over realizations and then equally over glaciers to calculate MAE, RMSE, and bias. RMSE is the square root of the averaged squared error. NMAD is calculated across glacier-mean signed errors.',
7:r'Per-glacier integrated volume-rate reconstruction errors (m$^3$ a$^{-1}$). Each error equals glacier-wide specific error multiplied by valid reference-data area. Metrics give equal weight to realizations within each glacier and equal weight to glaciers within each population. Observed values are retained, so reconstruction error arises entirely from withheld pixels.',
8:r'Regional-transfer performance on common test targets. Glacier MAE and signed bias are in m a$^{-1}$. Integrated RMSE describes per-glacier volume-rate errors in m$^3$ a$^{-1}$. Each regional hybrid uses 1,000 source glaciers. The full-data hybrid uses all 8,179 training glaciers from both regions.',
9:r'Matched hybrid models with absolute projected coordinates excluded (off) or included (on), alongside constant and hypsometric baselines. The off model uses 25 predictors, and the on model adds projected pixel-center easting and northing. Both models use identical capped training rows, labels, weights, seed, and fixed settings. MAE and Bias describe glacier-wide specific errors in m a$^{-1}$. Vol. RMSE and Vol. bias describe per-glacier integrated volume-rate errors in m$^3$ a$^{-1}$. Biases are signed.'}
FILES={4:'tab_pixel.tex',5:'tab_paired.tex',6:'tab_glacier.tex',7:'tab_volume.tex',8:'tab_transfer.tex',9:'tab_coordinates.tex'}
LABELS={4:'tab:pixel',5:'tab:paired',6:'tab:glacier',7:'tab:volume',8:'tab:transfer',9:'tab:coordinates'}
TITLES={4:'Pixel reconstruction errors',5:'Paired hybrid-minus-hypsometric contrasts',6:'Glacier-wide specific reconstruction errors',7:'Integrated volume-rate reconstruction errors',8:'Regional-transfer sensitivity',9:'Coordinate comparison'}
EXCLUSION=r'Evaluation excludes RGI60-14.17287 and its five realizations. Western, Eastern, and combined counts are 1,422, 332, and 1,754 glaciers.'

def fmt(value,signed=False,scientific=False,tex=False):
    d=Decimal(str(float(value)))
    if not d.is_finite():raise ValueError(value)
    if d==0:return '+0.000' if signed else '0.000'
    e=d.copy_abs().adjusted()
    q=d.quantize(Decimal(1).scaleb(e-2),rounding=ROUND_HALF_EVEN)
    e=q.copy_abs().adjusted()
    prefix='+' if signed and q>=0 else ''
    if scientific:
        mant=q.scaleb(-e)
        s=f'{mant:.2f}'
        if tex:return prefix+s+r'\times10^{'+str(e)+'}'
        supers=str(e).translate(str.maketrans('-0123456789','⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
        return prefix+s+' × 10'+supers
    return prefix+f'{q:.{max(0,2-e)}f}'

def value(scope,method,kind,metric):
    return float(points.loc[points.scope.eq(scope)&points.method.eq(method)&points.kind.eq(kind),metric].iloc[0])

tables={}
for number,kind,cols in [(4,'pixel',['mae','rmse','nmad','bias']),(6,'specific',['mae','rmse','nmad','bias']),(7,'volume',['mae','rmse','bias'])]:
    rows=[]
    for scope,region in REGIONS:
        values={m:{c:value(scope,m,kind,c) for c in cols} for m,_ in METHODS}
        best={c:min(METHODS,key=lambda x:abs(values[x[0]][c]) if c=='bias' else values[x[0]][c])[0] for c in cols}
        for method,label in METHODS:
            rows.append(dict(group=region,method=label,values=[values[method][c] for c in cols],bold=[best[c]==method for c in cols]))
    tables[number]=dict(headers=['Region','Method']+[c.upper() if c!='bias' else 'Bias' for c in cols],rows=rows,metric_cols=cols,scientific=[kind=='volume']*len(cols))

rows=[]
for kind,metric,name in [('pixel','mae','Pixel MAE'),('pixel','abs_bias','Pixel absolute bias'),('specific','mae','Glacier MAE'),('specific','nmad','Glacier NMAD'),('specific','abs_bias','Glacier absolute bias')]:
    vals=[]
    for scope in ['whim','ehim']:
        s=ci.loc[ci.scope.eq(scope)&ci.kind.eq(kind)&ci.baseline.eq('hypsometric')&ci.comparison.eq('xgb')&ci.metric.eq(metric)].iloc[0]
        point=abs(value(scope,'xgb',kind,'bias'))-abs(value(scope,'hypsometric',kind,'bias')) if metric=='abs_bias' else value(scope,'xgb',kind,metric)-value(scope,'hypsometric',kind,metric)
        vals.append([point,float(s.ci_low),float(s.ci_high)])
    rows.append(dict(metric=name,values=vals))
tables[5]=dict(headers=['Metric','Western','Eastern'],rows=rows)

rows=[]
for scope,region in REGIONS[:2]:
    group=[]
    for method,label in [('constant','Constant'),('hypsometric','Hypsometric'),('xgb','Hybrid (full data)')]:
        group.append(dict(group=region,method=label,values=[value(scope,method,'specific','mae'),value(scope,method,'specific','bias'),value(scope,method,'volume','rmse')]))
    for source,label in [('whim','Hybrid (Western-trained)'),('ehim','Hybrid (Eastern-trained)')]:
        s=transfer.loc[transfer.target.eq(scope)&transfer.source.eq(source)].iloc[0]
        group.append(dict(group=region,method=label,values=[float(s.glacier_mae),float(s.glacier_bias),float(s.integrated_rmse)]))
    for row in group:
        row['bold']=[row is min(group,key=lambda x:abs(x['values'][j]) if j==1 else x['values'][j]) for j in range(3)]
    rows.extend(group)
tables[8]=dict(headers=['Target','Model','Glacier MAE','Bias','Integrated RMSE'],metric_cols=['mae','bias','rmse'],scientific=[False,False,True],rows=rows)

rows=[]
for scope,region in REGIONS:
    group=[]
    for method,label in [('constant','Constant'),('hypsometric','Hypsometric')]:
        group.append(dict(group=region,method=label,values=[value(scope,method,'specific','mae'),value(scope,method,'specific','bias'),value(scope,method,'volume','rmse'),value(scope,method,'volume','bias')]))
    for arm,label in [('off','Hybrid (coords off)'),('on_absolute','Hybrid (coords on)')]:
        s=coords.loc[coords.scope.eq('combined' if scope=='all' else scope)&coords.arm.eq(arm)].iloc[0]
        group.append(dict(group=region,method=label,values=[float(s.glacier_mae),float(s.glacier_bias),float(s.integrated_rmse),float(s.integrated_bias)]))
    for row in group:
        row['bold']=[row is min(group,key=lambda x:abs(x['values'][j]) if j in [1,3] else x['values'][j]) for j in range(4)]
    rows.extend(group)
tables[9]=dict(headers=['Region','Method / arm','MAE','Bias','Vol. RMSE','Vol. bias'],metric_cols=['mae','bias','rmse','bias'],scientific=[False,False,True,True],rows=rows)

md=['All results exclude RGI60-14.17287 and its five realizations. Western, Eastern, and combined counts are 1,422, 332, and 1,754 glaciers. Values are rounded to three significant figures. Bold identifies the lowest error metric or bias magnitude within each population, using unrounded values.']
audit=[]
for number in range(4,10):
    t=tables[number]
    cap=CAPS[number]
    assert ';' not in cap and '\u2014' not in cap
    align='lrr' if number==5 else 'll'+'r'*(len(t['headers'])-2)
    tex=[r'\begin{table}[!htbp]',r'\centering\small',r'\setlength{\tabcolsep}{5pt}',r'\caption{'+cap+'}',r'\label{'+LABELS[number]+'}',r'\begin{tabular}{'+align+'}',r'\toprule',' & '.join(t['headers'])+r' \\',r'\midrule']
    md.extend(['',f'**Table {number+1}. {TITLES[number]}**',''])
    if number in [4,5,6]:md.append('All values are in m a⁻¹.\n')
    elif number==7:md.append('All values are in m³ a⁻¹.\n')
    elif number==8:md.append('Glacier MAE and Bias are in m a⁻¹. Integrated RMSE is in m³ a⁻¹.\n')
    else:md.append('MAE and Bias are in m a⁻¹. Vol. RMSE and Vol. bias are in m³ a⁻¹.\n')
    md+=['| '+' | '.join(t['headers'])+' |','| '+' | '.join(['---']+(['---:']*2 if number==5 else ['---']+['---:']*(len(t['headers'])-2)))+' |']
    previous=None
    for row in t['rows']:
        if number==5:
            def contrast(v,latex=False):
                return fmt(v[0],True,tex=latex)+(r'\,[' if latex else ' [')+fmt(v[1],True,tex=latex)+', '+fmt(v[2],True,tex=latex)+']'
            tex.append(row['metric']+' & '+' & '.join('$'+contrast(v,True)+'$' for v in row['values'])+r' \\')
            md.append('| '+row['metric']+' | '+' | '.join(contrast(v) for v in row['values'])+' |')
            for scope,v in zip(['whim','ehim'],row['values']):
                audit.append(dict(table=number+1,group=scope,method=row['metric'],values=v))
            continue
        if previous is not None and previous!=row['group']:tex.append(r'\midrule')
        first=row['group'] if previous!=row['group'] else ''
        texvals=[];mdvals=[]
        for j,v in enumerate(row['values']):
            signed=t['metric_cols'][j]=='bias';sci=t['scientific'][j]
            a=fmt(v,signed,sci,True);b=fmt(v,signed,sci)
            texvals.append(r'{\boldmath$'+a+'$}' if row['bold'][j] else '$'+a+'$')
            mdvals.append('**'+b+'**' if row['bold'][j] else b)
        tex.append(first+' & '+row['method']+' & '+' & '.join(texvals)+r' \\')
        md.append('| '+row['group']+' | '+row['method']+' | '+' | '.join(mdvals)+' |')
        previous=row['group'];audit.append(dict(table=number+1,**row))
    note=EXCLUSION+' Values are rounded to three significant figures. '
    note+='Point estimates and interval endpoints are rounded independently.' if number==5 else 'Bold identifies the lowest error metric and the bias closest to zero within each population, using unrounded values.'
    tex += [r'\bottomrule',r'\end{tabular}',r'\belowtable{'+note+'}',r'\end{table}']
    text='\n'.join(tex)+'\n'
    (OUT/'tables'/FILES[number]).write_text(text)

(OUT/'tables/tables_copy_ready.md').write_text('\n'.join(md)+'\n')
(OUT/'tables/table_values_unrounded.json').write_text(json.dumps(audit,indent=2)+'\n')

# Verify the reported manuscript percentages and population.
def get(scope,method,kind,metric):return value(scope,method,kind,metric)
expect=[('whim','pixel',16.4),('ehim','pixel',14.7),('whim','specific',20.1),('ehim','specific',16.9),('all','specific',19.4),('whim','volume',19.8),('ehim','volume',16.1)]
for scope,kind,percent in expect:
    reduction=100*(get(scope,'hypsometric',kind,'mae')-get(scope,'xgb',kind,'mae'))/get(scope,'hypsometric',kind,'mae')
    assert float(fmt(reduction))==percent,(scope,kind,reduction)
assert [int(points.loc[points.scope.eq(scope)&points.kind.eq('specific')&points.method.eq('constant'),'n_glaciers'].iloc[0]) for scope,_ in REGIONS]==[1422,332,1754]
assert sum(len(tables[n]['rows']) for n in tables)==54
assert all('NMAD' not in (OUT/'tables'/FILES[n]).read_text().split(r'\belowtable')[1] for n in [7,8,9])
print(OUT)
