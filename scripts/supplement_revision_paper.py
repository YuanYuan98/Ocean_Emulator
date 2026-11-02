"""Complete the paper Figure 4 and Tables 1/3/4 comparison on compute nodes.

Read existing final rollouts; never retrain or modify checkpoints. Per-model
caches make a new allocation resumable. All fields retain the common wet mask.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import detrend
import cartopy.crs as ccrs

from compare_revision_baselines import (
    BASE, RUNS, DEPTH, DZ, LAYERS, REGIONS, COLORS, UNITS,
    gaussian_area, weighted_mean, correlation, seasonal_anomaly,
)

KEYS = list(RUNS)
SHORT = ['Samudra 2', 'AdaLoss', 'Frozen', 'Tendency', 'Neptune']
LEVELS = [0, 10, 14]
NAMES = {'thetao':'Temperature', 'so':'Salinity', 'uo':'Zonal velocity',
         'vo':'Meridional velocity', 'ke':'Kinetic energy'}
SNAP = 580


def reduce(key, root):
    out = root / 'paper_supplement' / key
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        print('Cached:', key, flush=True)
        return
    pred = xr.open_zarr(RUNS[key][1]).rename({'lat':'y','lon':'x'})
    truth = xr.open_zarr(BASE / 'datasets/onedeg/OM4.zarr').sel(time=pred.time)
    np.testing.assert_array_equal(pred.time.values, xr.open_zarr(RUNS['samudra2'][1]).time.values)
    for dim in ['y','x']:
        np.testing.assert_allclose(pred[dim], truth[dim], atol=1e-9)
    lat, lon = truth.y.values, truth.x.values
    area = gaussian_area(lat, lon)
    masks = [truth[f'mask_{i}'].values.astype(bool) for i in range(19)]
    days = np.array([t.dayofyr for t in pred.time.values])
    same_day = np.flatnonzero(days == days[SNAP])
    saved = {'lat':lat, 'lon':lon, 'date':str(pred.time.values[SNAP]), 'area':area}
    checks = []

    def read(ds, var, level, indices=None):
        field = ds[f'{var}_{level}']
        if indices is not None:
            field = field.isel(time=indices)
        a = field.transpose('time','y','x').values.astype('float64')
        if not np.isfinite(a[:,masks[level]]).all():
            raise ValueError(f'Nonfinite wet field {key} {var} {level}')
        return np.where(masks[level], a, 0)

    def zonal(a, level):
        return weighted_mean(a[...,None,:], (area*masks[level])[:,None,:])

    for var in ['thetao','so']:
        z = {who:[] for who in ['truth','pred']}
        for layer, (lo,hi,_) in enumerate(LAYERS):
            a = np.zeros((len(days),len(lat),len(lon))); b = np.zeros_like(a)
            thick = np.zeros(area.shape)
            for level in np.flatnonzero((DEPTH>=lo)&(DEPTH<=hi)):
                print(key, var, int(level), flush=True)
                av, bv = read(truth,var,level), read(pred,var,level)
                w = DZ[level]*masks[level]; a += av*w; b += bv*w; thick += w
                for who, field in [('truth',av),('pred',bv)]:
                    z[who].append(zonal(field,level))
                    if level in LEVELS+[15]:
                        saved[f'{var}_{who}_raw_{level}'] = np.where(masks[level],field[SNAP],np.nan)
                        saved[f'{var}_{who}_anom_{level}'] = np.where(masks[level],field[SNAP]-field[same_day].mean(axis=0),np.nan)
            wet = thick>0; a /= np.where(wet,thick,1); b /= np.where(wet,thick,1)
            saved[f'{var}_raw_rmse_map_{layer}'] = np.where(wet, np.sqrt(np.mean((b-a)**2,axis=0)), np.nan)
            cached = np.load(root/key/'diagnostics.npz')
            np.testing.assert_allclose(weighted_mean(saved[f'{var}_raw_rmse_map_{layer}']**2,area*wet),np.mean(cached[f'{var}_{layer}_rmse_time']**2),rtol=1e-9)
            checks.append(f'{var}_{layer}: map/time RMSE identity')
            del a,b,av,bv
        for who in ['truth','pred']:
            zz = np.stack(z[who],axis=1)  # time,depth,lat
            saved[f'{var}_{who}_zonal_mean'] = zz.mean(axis=0)
            saved[f'{var}_{who}_zonal_snapshot'] = zz[SNAP]
            saved[f'{var}_{who}_zonal_anom_series'] = seasonal_anomaly(zz,days)

    # Appendix velocity and KE snapshots need only this day-of-year subset.
    for level in LEVELS+[15]:
        for who,ds in [('truth',truth),('pred',pred)]:
            u, v = read(ds,'uo',level,same_day), read(ds,'vo',level,same_day)
            idx = int(np.flatnonzero(same_day==SNAP)[0])
            for var,field in [('uo',u),('vo',v),('ke',(u*u+v*v)/2)]:
                saved[f'{var}_{who}_raw_{level}'] = np.where(masks[level],field[idx],np.nan)
                saved[f'{var}_{who}_anom_{level}'] = np.where(masks[level],field[idx]-field.mean(axis=0),np.nan)
    np.savez_compressed(out/'fields.npz',**saved)
    (out/'complete.json').write_text(json.dumps({'model':key,'n_times':len(days),'snapshot':str(pred.time.values[SNAP]),'checks':checks,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))


def paper_timeseries(var, old):
    # Match Fig 4b with per-model panels so metric labels stay readable.
    fig,axs=plt.subplots(5,3,figsize=(13,10),layout='constrained')
    dates=pd.to_datetime([s[:10] for s in old['samudra2']['dates']])
    for j,(_,_,label) in enumerate(LAYERS):
        allseries=[old['samudra2'][f'{var}_{j}_truth_detrended']]+[old[k][f'{var}_{j}_pred_detrended'] for k in KEYS]
        lim=np.max(np.abs(np.concatenate(allseries)))*1.05
        for i,key in enumerate(KEYS):
            ax=axs[i,j];a=allseries[0];b=allseries[i+1]
            ax.plot(dates,a,color=COLORS['truth'],ls='--',lw=1,label='OM4')
            ax.plot(dates,b,color=COLORS[key],lw=.85,label=SHORT[i])
            r2=1-np.sum((a-b)**2)/np.sum((a-a.mean())**2)
            ax.text(.02,.95,f'R² = {r2:.3f}',va='top',transform=ax.transAxes,fontsize=10)
            ax.set_ylim(-lim,lim);ax.grid(alpha=.15)
            ax.set_xlim(dates[0],dates[-1])
            ax.set_xticks([dates[0],pd.Timestamp("2016-01-01"),pd.Timestamp("2018-01-01"),pd.Timestamp("2020-01-01"),dates[-1]], ["2014","2016","2018","2020","2022"])
            if i==0:ax.set_title(label)
            if j==0:ax.set_ylabel(SHORT[i]+'\n'+UNITS[var])
            if i<4:ax.tick_params(labelbottom=False)
    fig.suptitle(NAMES[var]+' — detrended global mean (OM4 dashed)')
    return fig


def paper_table_figure(rows, winners, var, label, paper):
    fig,ax=plt.subplots(figsize=(12.5,10));ax.axis('off')
    table=ax.table(cellText=rows,colLabels=['Region','Metric']+SHORT,cellLoc='center',colWidths=[.20,.22,.116,.116,.116,.116,.116],bbox=[0,.02,1,.91])
    table.auto_set_font_size(False);table.set_fontsize(14)
    for (r,c),cell in table.get_celld().items():
        cell.set_linewidth(.3);cell.set_edgecolor('#aaaaaa')
        if r==0:cell.set_facecolor('#e2e8f0');cell.set_text_props(weight='bold',fontsize=12)
        elif (r-1)//4%2==1:cell.set_facecolor('#f6f6f6')
        if r and c>=2 and (c-2) in winners[r-1]:cell.set_facecolor('#cbd5e1');cell.set_text_props(weight='bold')
    ax.set_title(f'{NAMES[var]} | {label} | paper Table {paper} metrics',fontsize=18,pad=12)
    unit=UNITS[var];vunit={'thetao':'°C²','so':'psu²','ke':'m⁴/s⁴'}[var]
    fig.text(.5,.012,f'Var Corr: dimensionless; Var RMSE: {vunit}; Direct / Detrend RMSE: {unit}. Bold shaded = best unrounded value.',ha='center',fontsize=12)
    return fig


def render(root):
    target = root/'paper_supplement'; figs=target/'figures'; figs.mkdir(exist_ok=True)
    data={k:np.load(target/k/'fields.npz') for k in KEYS}
    old={k:np.load(root/k/'diagnostics.npz') for k in KEYS}
    truth=data['samudra2']; lat,lon,area=truth['lat'],truth['lon'],truth['area']
    manifest=[]; annotations=[]
    plt.rcParams.update({'font.size':11,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False})

    def save(fig, name, title, caption):
        fig.savefig(figs/f'{name}.png',dpi=190,bbox_inches='tight')
        fig.savefig(figs/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig)
        manifest.append({'file':str((figs/f'{name}.png').resolve()),'title':title,'caption':caption})

    def metric(a,b,w):
        return float(np.sqrt(weighted_mean((b-a)**2,w))),correlation(a,b,w)

    def map_panel(ax,field,limits,cmap):
        m=ax.pcolormesh(lon,lat,field,transform=ccrs.PlateCarree(),cmap=cmap,vmin=limits[0],vmax=limits[1],rasterized=True)
        ax.set_global(); ax.set_facecolor('#dedede')
        return m

    def maps(var,kind):
        error = kind=='error'
        rows=KEYS if error else ['truth']+KEYS
        fig,axs=plt.subplots(len(rows),3,figsize=(13,2.0*len(rows)),subplot_kw={'projection':ccrs.Robinson(central_longitude=210)},layout='constrained')
        for j,lev in enumerate(LEVELS):
            t=truth[f'{var}_truth_{"raw" if kind in ["raw","error"] else "anom"}_{lev}']
            fields=[]
            for key in rows:
                a=t if key=='truth' else data[key][f'{var}_pred_{"raw" if kind in ["raw","error"] else "anom"}_{lev}']
                fields.append(a-t if error else a)
            values=np.concatenate([f[np.isfinite(f)] for f in fields])
            if kind=='raw' and var in ['thetao','so']:
                limits=np.percentile(values,[.25,99.75]); cmap='viridis' if var=='so' else 'inferno'
            else:
                limit=np.percentile(np.abs(values),99); limits=(-limit,limit); cmap='RdBu_r'
            for i,(key,f) in enumerate(zip(rows,fields)):
                ax=axs[i,j];m=map_panel(ax,f,limits,cmap)
                if i==0:ax.set_title(f'{DEPTH[lev]:g} m')
                if j==0:ax.text(-.03,.5,'OM4' if key=='truth' else SHORT[KEYS.index(key)],transform=ax.transAxes,ha='right',va='center',rotation=90)
                if key!='truth':
                    ta=truth[f'{var}_truth_anom_{lev}'];pa=data[key][f'{var}_pred_anom_{lev}']
                    # The paper snapshot helper flattens valid cells without
                    # area weighting; retain that convention for annotations.
                    good=np.isfinite(ta)&np.isfinite(pa)
                    rm=float(np.sqrt(np.mean((pa[good]-ta[good])**2)))
                    co=float(np.corrcoef(ta[good],pa[good])[0,1])
                    rawrm,_=metric(truth[f'{var}_truth_raw_{lev}'],data[key][f'{var}_pred_raw_{lev}'],area)
                    ax.text(.02,.02, f'raw RMSE {rawrm:.3g}' if error else f'anom RMSE {rm:.3g}; r {co:.3f}',transform=ax.transAxes,fontsize=9,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
                    if kind=='raw':annotations.append({'model':key,'variable':var,'depth_m':float(DEPTH[lev]),'raw_snapshot_rmse':rawrm,'anomaly_snapshot_rmse':rm,'anomaly_snapshot_corr':co})
            fig.colorbar(m,ax=axs[:,j],orientation='horizontal',fraction=.025,pad=.02,label=UNITS[var],extend='both')
        fig.suptitle(f'{NAMES[var]} — {"prediction minus OM4" if error else kind+" state"} — 2022-09-30')
        cn={'thetao':'温度','so':'盐度','uo':'纬向速度 u','vo':'经向速度 v','ke':'动能 KE'}[var]
        title=f'{cn}：'+{'raw':'原始状态地图快照','error':'原始状态误差地图','anom':'去季节异常地图快照'}[kind]
        caption=('Figure 4(d) / Appendix snapshot extension. ' if kind!='error' else 'Added explicit error diagnostic: prediction minus OM4 at the same timestamp. ')+f'Predicted frame 580, 2022-09-30; actual levels 2.5, 775, 2400 m (paper labels 700/2000 refer to indices 10/14). '+('Colors show raw states; annotations use each run’s own day-of-year anomalies, as in Figure 4(d). ' if kind=='raw' else '')+('Raw RMSE annotations are area-weighted; the seasonal cycle is retained. ' if error else 'Anomaly RMSE and correlation are unweighted over valid wet cells, matching the paper plotting helper. ')+('Common scale per depth; 0.25/99.75 percentiles pooled across models.' if kind=='raw' and var in ['thetao','so'] else 'Common symmetric scale per depth; pooled 99th-percentile absolute magnitude; colorbar extensions indicate saturation.')
        if var=='ke':caption+=' KE = (u²+v²)/2; remove its own seasonal cycle. This is not KE computed from velocity anomalies.'
        save(fig,f'{var}_snapshot_{kind}',title,caption)

    for var in ['thetao','so']:
        fig=paper_timeseries(var,old)
        save(fig,f'{var}_paper_timeseries',('温度' if var=='thetao' else '盐度')+'：论文式去趋势时间序列', 'Figure 4(b) / Appendix Figures 18–19 style, extended to five models. Volume-weighted global mean, separately linearly detrended. Shared y-axis per depth across all five models; scales differ between depths. R² is shown for each model; raw series remain in original Figures 1–2.')

        for kind in ['mean','snapshot']:
            fig,axs=plt.subplots(3,2,figsize=(12,10),layout='constrained')
            vals=[truth[f'{var}_truth_zonal_{kind}']]+[data[k][f'{var}_pred_zonal_{kind}'] for k in KEYS]
            limits=(-2,30) if var=='thetao' else (30,37)
            for ax,key,v in zip(axs.flat,['truth']+KEYS,vals):
                m=ax.pcolormesh(lat,DEPTH,v,cmap='inferno' if var=='thetao' else 'viridis',vmin=limits[0],vmax=limits[1],shading='nearest',rasterized=True)
                ax.invert_yaxis();ax.set(xlabel='Latitude (°N)',ylabel='Depth (m)',title='OM4' if key=='truth' else SHORT[KEYS.index(key)],facecolor='#dedede')
                if key!='truth':
                    a=truth[f'{var}_truth_zonal_anom_series'];b=data[key][f'{var}_pred_zonal_anom_series']
                    if kind=='snapshot':a,b=a[SNAP],b[SNAP]
                    good=np.isfinite(a)&np.isfinite(b)
                    rm=np.sqrt(np.mean((a[good]-b[good])**2));co=np.corrcoef(a[good],b[good])[0,1]
                    ax.text(.03,.05,f'anom RMSE {rm:.4g}; r {co:.3f}',transform=ax.transAxes,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
                    annotations.append({'model':key,'variable':var,'zonal_kind':kind,'anomaly_zonal_rmse':float(rm),'anomaly_zonal_corr':float(co)})
            fig.colorbar(m,ax=axs,orientation='horizontal',fraction=.035,pad=.03,label=UNITS[var],extend='both')
            save(fig,f'{var}_zonal_state_{kind}',('温度' if var=='thetao' else '盐度')+'：纬向平均原始状态'+('（八年平均）' if kind=='mean' else '（2022-09-30）'), 'Figure 4(c) '+('time-averaged state cross-section.' if kind=='mean' else 'extension: single-timestamp cross-section on 2022-09-30.')+' Raw fields shown. Wet zonal averaging; at fixed Gaussian latitude this equals dx weighting. Annotation follows notebook: unweighted RMSE/correlation over valid deseasonalized (time, depth, latitude) profiles for the mean figure; over (depth, latitude) for the snapshot. These annotations do not measure the raw image error. Common color scale, extensions show values outside limits.')
        for kind in ['raw','error']:maps(var,kind)
        fig,axs=plt.subplots(5,3,figsize=(13,10),subplot_kw={'projection':ccrs.Robinson(central_longitude=210)},layout='constrained')
        for j,(_,_,label) in enumerate(LAYERS):
            vv=[data[k][f'{var}_raw_rmse_map_{j}'] for k in KEYS]
            mx=np.percentile(np.concatenate([v[np.isfinite(v)] for v in vv]),99)
            for i,key in enumerate(KEYS):
                m=map_panel(axs[i,j],vv[i],(0,mx),'magma')
                if i==0:axs[i,j].set_title(label)
                if j==0:axs[i,j].text(-.03,.5,SHORT[i],transform=axs[i,j].transAxes,rotation=90,ha='right',va='center')
            fig.colorbar(m,ax=axs[:,j],orientation='horizontal',fraction=.025,pad=.02,label=UNITS[var],extend='max')
        save(fig,f'{var}_raw_rmse_maps',('温度' if var=='thetao' else '盐度')+'：全时段 RMSE 地图','Temporal RMSE at each wet cell, computed on thickness-weighted layer-mean raw states over all 598 predictions. Retains mean and seasonal errors; differs from paper-table Direct RMSE on seasonal anomalies. Common scale per layer, pooled 99th percentile. Area mean of squared map values was checked against the mean squared rollout RMSE.')

    for var in ['uo','vo']:maps(var,'anom')
    # Appendix Fig 13 helper computes KE of deseasonalized velocities,
    # rather than deseasonalizing KE itself (the variance-table diagnostic).
    from matplotlib.colors import LogNorm
    fig,axs=plt.subplots(6,3,figsize=(13,12),subplot_kw={'projection':ccrs.Robinson(central_longitude=210)},layout='constrained')
    for j,lev in enumerate(LEVELS):
        fields=[]
        for key in ['truth']+KEYS:
            source=truth if key=='truth' else data[key];who='truth' if key=='truth' else 'pred'
            fields.append((source[f'uo_{who}_anom_{lev}']**2+source[f'vo_{who}_anom_{lev}']**2)/2)
        positive=np.concatenate([a[np.isfinite(a)&(a>0)] for a in fields])
        lo,hi=np.percentile(positive,[1,99])
        for i,(key,a) in enumerate(zip(['truth']+KEYS,fields)):
            ax=axs[i,j]
            m=ax.pcolormesh(lon,lat,np.maximum(a,lo),transform=ccrs.PlateCarree(),cmap='magma',norm=LogNorm(lo,hi),rasterized=True)
            ax.set_global();ax.set_facecolor('#dedede')
            if i==0:ax.set_title(f'{DEPTH[lev]:g} m')
            if j==0:ax.text(-.03,.5,'OM4' if key=='truth' else SHORT[KEYS.index(key)],transform=ax.transAxes,ha='right',va='center',rotation=90)
        fig.colorbar(m,ax=axs[:,j],orientation='horizontal',fraction=.025,pad=.02,label='m²/s² (log scale)',extend='both')
    fig.suptitle('EKE of deseasonalized velocities — 2022-09-30')
    save(fig,'eke_snapshot','附录 Figure 13：去季节速度的 EKE 快照','Matches the Appendix Figure 13 plotting helper: EKE = (u′² + v′²)/2, with each velocity’s own day-of-year climatology removed before squaring. This differs from the full-period-mean EKE definition in Appendix Eq. 9 and from temporal variance of KE in Table 4; the implementation distinction is explicit. 2022-09-30; actual levels 2.5/775/2400 m. Common log scale per depth, pooled positive 1st/99th percentiles; zeros displayed at the lower bound.')
    # Correct Fig 4(e) comparison: both T and u are at 3100 m.
    fig,axs=plt.subplots(6,2,figsize=(11,12),subplot_kw={'projection':ccrs.Robinson(central_longitude=210)},layout='constrained')
    for j,var in enumerate(['thetao','uo']):
        fields=[truth[f'{var}_truth_anom_15']]+[data[k][f'{var}_pred_anom_15'] for k in KEYS]
        lim=np.percentile(np.abs(np.concatenate([a[np.isfinite(a)] for a in fields])),99)
        for i,(key,a) in enumerate(zip(['truth']+KEYS,fields)):
            m=map_panel(axs[i,j],a,(-lim,lim),'RdBu_r')
            if i==0:axs[i,j].set_title(NAMES[var]+' anomaly at 3100 m')
            if j==0:axs[i,j].text(-.03,.5,'OM4' if key=='truth' else SHORT[KEYS.index(key)],transform=axs[i,j].transAxes,rotation=90,ha='right',va='center')
        fig.colorbar(m,ax=axs[:,j],orientation='horizontal',fraction=.025,pad=.02,label=UNITS[var],extend='both')
    save(fig,'deep_T_u_3100m','论文 Figure 4(e)：3100 m 温度与同深度 u 异常','Both temperature and zonal velocity are at 3100 m on 2022-09-30, matching Figure 4(e). Each run’s day-of-year climatology removed. Original report Figure 11 used surface u and is a separate diagnostic, not the paper panel. Shared symmetric scales across models per variable, pooled 99th percentile. Spatial association is not causal proof of imprinting.')

    df=pd.read_csv(root/'all_metrics.csv')
    metrics=[('variance_corr','Var Corr ↑'),('variance_rmse','Var RMSE ↓'),('detrend_rmse','Detrend RMSE ↓'),('direct_rmse','Direct RMSE ↓')]
    table_records=[]
    for var,paper in [('thetao',1),('so',3),('ke',4)]:
        for layer,(_,_,label) in enumerate(LAYERS):
            rows=[]; winners=[]
            for region in REGIONS:
                for metric,label_metric in metrics:
                    sub=df[(df.variable==var)&(df.layer==label)&(df.region==region)].set_index('model').reindex(KEYS)
                    values=sub[metric].to_numpy(); assert np.isfinite(values).all()
                    best=np.max(values) if metric=='variance_corr' else np.min(values)
                    winners.append(np.flatnonzero(values==best))
                    rows.append([region,label_metric]+[f'{v:#.3g}' for v in values])
                    for k,v in zip(KEYS,values):table_records.append({'variable':var,'layer':label,'region':region,'metric':metric,'model':k,'value':float(v)})
            fig=paper_table_figure(rows,winners,var,label,paper)
            save(fig,f'table_{var}_{layer}',f'表 S{paper}.{layer+1}：'+{'thetao':'温度','so':'盐度','ke':'动能'}[var]+f'，{label}',f'Paper Table {paper} definitions, all seven regions and all five models; split by depth for readable comparison. Three significant digits. Var Corr higher is better; all RMSE lower is better. Best determined before rounding. All fields deseasonalized independently; shared wet masks and Gaussian cell areas. Fresh common-protocol calculation, not copied paper values; land-zero dilution and small area-weight conventions in legacy code can change published numbers. No significance claim from one rollout.')
    pd.DataFrame(table_records).to_csv(target/'paper_table_values.csv',index=False)
    (target/'snapshot_zonal_metrics.json').write_text(json.dumps(annotations,indent=2,allow_nan=False))
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('Rendered',len(manifest),'figures/tables; table entries',len(table_records),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--render-only',action='store_true');args=ap.parse_args()
    (args.root/'paper_supplement').mkdir(exist_ok=True)
    if not args.render_only:
        for key in KEYS:reduce(key,args.root)
    render(args.root)
