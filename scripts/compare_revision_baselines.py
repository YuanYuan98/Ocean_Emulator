"""Common OM4 diagnostics for final-epoch revision runs.

Run on a compute node. Each model writes an atomic completion marker, so a
four-hour allocation can resume at the next model. No model inference here.
Definitions follow Samudra 2 Appendix A and viz_jupyter/functions.py, with
explicit wet masks and physical units. The report records deliberate differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import detrend, periodogram

BASE = Path('/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace')
LOSS = BASE / 'experiments/loss_revision_20261003'
NEPTUNE = BASE / 'experiments/neptune_13m_om4_20261014'
RUNS = {
    'samudra2': ('Samudra 2', BASE / 'checkpoints/2026-01-30-samudra_om4_onedeg_20_cap_eval/predictions.zarr', None),
    'adaloss': ('AdaLoss', LOSS / 'inference/adaloss/predictions.zarr', LOSS / 'controllers/adaloss.json'),
    'frozen_final': ('Frozen final weights', LOSS / 'inference/frozen_final/predictions.zarr', LOSS / 'controllers/frozen_final.json'),
    'tendency': ('Tendency normalization', LOSS / 'inference/tendency/predictions.zarr', LOSS / 'controllers/tendency.json'),
    'neptune': ('Neptune (OM4 adaptation)', NEPTUNE / 'inference/neptune_om4/predictions.zarr', NEPTUNE / 'controllers/neptune.json'),
}
DEPTH = np.array([2.5,10,22.5,40,65,105,165,250,375,550,775,1050,1400,1850,2400,3100,4000,5000,6000])
DZ = np.diff([0,5,15,30,50,80,130,200,300,450,650,900,1200,1600,2100,2700,3500,4500,5500,6750])
LAYERS = [(0,700,'0–700 m'),(700,2000,'700–2000 m'),(2000,7000,'2000–7000 m')]
REGIONS = {'Global':None,'Gulf Stream':(300,320,25,45),'Kuroshio':(150,170,25,45),
           'Agulhas':(40,60,-50,-30),'Malvinas':(311,331,-51,-31),
           'Niño 3.4':(190,240,-5,5),'Tropical Pacific':(130,290,-30,30)}
COLORS = {'truth':'#202020','samudra2':'#D97706','adaloss':'#2563EB','frozen_final':'#8B5CF6','tendency':'#159779','neptune':'#C84278'}
UNITS = {'thetao':'°C','so':'psu','uo':'m/s','vo':'m/s','ke':'m²/s²','zos':'m'}


def ready(key):
    _, path, status = RUNS[key]
    return path.is_dir() and (status is None or json.loads(status.read_text())['status'] == 'complete')


def weighted_mean(a, weights):
    """Last two dimensions, retaining time if present; never include land."""
    good = np.isfinite(a) & (weights > 0)
    den = np.sum(np.where(good, weights, 0), axis=(-2,-1))
    num = np.sum(np.where(good, a, 0) * weights, axis=(-2,-1))
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den>0)


def correlation(a, b, weights):
    valid = np.isfinite(a) & np.isfinite(b)
    w = np.where(valid, weights, 0)
    da, db = a-weighted_mean(a,w), b-weighted_mean(b,w)
    denominator = np.sqrt(weighted_mean(da**2,w)*weighted_mean(db**2,w))
    return float(weighted_mean(da*db,w)/denominator) if denominator > 0 else np.nan


def seasonal_anomaly(a, dayofyear):
    """Notebook convention: each run's own test-period day-of-year climatology."""
    out = a.copy()
    for day in np.unique(dayofyear):
        select = dayofyear == day
        # Dry cells are zero internally, but remain excluded by the explicit mask.
        out[select] -= a[select].mean(axis=0, dtype=np.float64)
    return out


def time_metrics(a,b):
    delta = a-b
    ss = np.sum((a-a.mean())**2)
    return {'rmse':float(np.sqrt(np.mean(delta**2))), 'mae':float(np.mean(abs(delta))),
            'r2':float(1-np.sum(delta**2)/ss) if ss>0 else None,
            'corr':float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None}


def gaussian_area(lat, lon):
    nodes, weights = np.polynomial.legendre.leggauss(len(lat))
    np.testing.assert_allclose(np.degrees(np.arcsin(nodes)),lat,atol=1e-7)
    np.testing.assert_allclose(np.diff(lon),360/len(lon),atol=1e-9)
    return np.broadcast_to(weights[:,None]*2*np.pi*6.371e6**2/len(lon),(len(lat),len(lon))).copy()


def region_weights(area, lat, lon):
    result = {}
    for name, box in REGIONS.items():
        if box is None:
            result[name] = area
        else:
            west,east,south,north=box
            result[name]=area*((lat[:,None]>=south)&(lat[:,None]<=north)&(lon[None,:]>=west)&(lon[None,:]<=east))
    return result


def spectral_diagnostics(a, lat, lon, mask, dayofyear):
    """Surface spectra and ACF, on a common land mask for every run.

    Spatial spectrum: time-mean-removed snapshots, plane detrend, Hann,
    radial bins with strict Nyquist cutoff (fixes overflow into last old bin).
    Temporal spectrum: seasonal anomalies, one-sided density, cycles/year.
    """
    out={}
    for name, box in REGIONS.items():
        if box is None:
            continue
        west,east,south,north=box
        yi=np.flatnonzero((lat>=south)&(lat<=north)); xi=np.flatnonzero((lon>=west)&(lon<=east))
        x=a[:,yi[:,None],xi].astype(float)
        wet=mask[yi[:,None],xi]
        x=np.where(wet,x,0)
        temporal=seasonal_anomaly(x,dayofyear)
        f,psd=periodogram(temporal,fs=365.25/5,axis=0,detrend='constant',scaling='density')
        w=np.cos(np.deg2rad(lat[yi]))[:,None]*wet
        out[f'{name}_f']=f
        out[f'{name}_temporal']=weighted_mean(psd,w)
        x-=x.mean(axis=0)
        h,ww=x.shape[-2:]
        yy,xx=np.meshgrid(np.linspace(-1,1,h),np.linspace(-1,1,ww),indexing='ij')
        design=np.stack([xx.ravel(),yy.ravel(),np.ones(h*ww)],axis=1)
        coef=np.linalg.lstsq(design,x.reshape(len(x),-1).T,rcond=None)[0]
        residual=x-(design@coef).T.reshape(x.shape)
        win=np.outer(np.hanning(h+1)[:-1],np.hanning(ww+1)[:-1])
        dx=np.deg2rad(np.diff(lon[xi]).mean())*6.371e6*np.cos(np.deg2rad(lat[yi]).mean())/1000
        dy=np.deg2rad(np.diff(lat[yi]).mean())*6.371e6/1000
        power=(np.abs(np.fft.fft2(residual*win,axes=(-2,-1),norm='forward'))**2).mean(axis=0)*h*ww*dx*dy/(win**2).mean()
        ky,kx=np.meshgrid(np.fft.fftfreq(h,dy),np.fft.fftfreq(ww,dx),indexing='ij')
        radial=np.hypot(kx,ky); edges=np.linspace(0,min(1/(2*dx),1/(2*dy)),max(2,min(h,ww)//4)+1)
        k=(edges[:-1]+edges[1:])/2
        spectrum=np.array([power[(radial>=lo)&(radial<hi)].mean() if np.any((radial>=lo)&(radial<hi)) else np.nan for lo,hi in zip(edges[:-1],edges[1:])])*k
        out[f'{name}_k']=k;out[f'{name}_spatial']=spectrum
        # Pooled covariance, overlap-normalized, not a significance estimate.
        z=temporal-temporal.mean(axis=0)
        variance=weighted_mean(np.mean(z*z,axis=0),w)
        out[f'{name}_acf']=np.array([weighted_mean(np.mean(z[:len(z)-lag]*z[lag:],axis=0),w)/variance if variance>0 else np.nan for lag in range(61)])
    return out


def reduce_model(key, output):
    if not ready(key):
        raise RuntimeError(f'{key}: final inference is not complete')
    path=RUNS[key][1]; destination=output/key; destination.mkdir(parents=True,exist_ok=True)
    truth=xr.open_zarr(BASE/'datasets/onedeg/OM4.zarr')
    pred=xr.open_zarr(path).rename({'lat':'y','lon':'x'})
    reference=xr.open_zarr(RUNS['samudra2'][1])
    np.testing.assert_array_equal(pred.time.values,reference.time.values)
    for dim in ['y','x']:
        np.testing.assert_allclose(pred[dim].values,truth[dim].values,rtol=0,atol=1e-9)
    if pred.sizes['time'] != 598:
        raise ValueError('Expected 598 predicted frames, excluding the two initial states')
    truth=truth.sel(time=pred.time)
    times=pred.time.values; days=np.array([t.dayofyr for t in times]); n=len(times)
    lat,lon=truth.y.values,truth.x.values; area=gaussian_area(lat,lon); regions=region_weights(area,lat,lon)
    masks=[truth[f'mask_{i}'].values.astype(bool) for i in range(19)]
    shape=(n,len(lat),len(lon)); saved={'lat':lat,'lon':lon,'days':np.arange(n)*5,'dates':np.array([str(t) for t in times])}
    rows=[]; index_metrics={}; finite=[]

    def get(ds,var,level):
        name=f'{var}_{level}' if var != 'zos' else var
        x=ds[name].transpose('time','y','x').values.astype(np.float64)
        mask=masks[level]
        if not np.isfinite(x[:,mask]).all():
            raise ValueError(f'{key}: nonfinite wet values in {name}')
        return np.where(mask,x,0)

    for var in ['thetao','so','uo','vo','ke','zos']:
        print(f'{key}: {var}',flush=True)
        layers=LAYERS if var!='zos' else [(0,0,'Surface')]
        for layer,(lo,hi,title) in enumerate(layers):
            levels=np.flatnonzero((DEPTH>=lo)&(DEPTH<=hi)) if var!='zos' else [0]
            a=np.zeros(shape); b=np.zeros(shape); thickness=np.zeros(shape[1:])
            for level in levels:
                weight=(DZ[level] if var!='zos' else 1)*masks[level]
                if var=='ke':
                    ta,tb=get(truth,'uo',level),get(truth,'vo',level)
                    pa,pb=get(pred,'uo',level),get(pred,'vo',level)
                    av,bv=(ta*ta+tb*tb)/2,(pa*pa+pb*pb)/2
                else:
                    av,bv=get(truth,var,level),get(pred,var,level)
                finite.append(f'{var}_{level}')
                a+=av*weight; b+=bv*weight; thickness+=weight
                if var in ['thetao','so']:
                    # Per-level zonal bias and final snapshot; actual coordinate depths.
                    saved[f'{var}_zonal_{level}']=weighted_mean((bv-av).mean(axis=0)[:,None,:],area[:,None,:]*masks[level][:,None,:])
                if var=='thetao' and level in [0,10,14,15]:
                    aa,bb=seasonal_anomaly(av,days),seasonal_anomaly(bv,days)
                    saved[f'thetao_snapshot_truth_{level}']=np.where(masks[level],aa[580],np.nan)
                    saved[f'thetao_snapshot_pred_{level}']=np.where(masks[level],bb[580],np.nan)
                if layer==0 and level==0 and var in ['thetao','uo','vo','zos']:
                    for who,field in [('truth',av),('pred',bv)]:
                        for name,value in spectral_diagnostics(field,lat,lon,masks[0],days).items():
                            saved[f'{var}_{who}_{name}']=value
                    if var=='thetao':
                        for who,field in [('truth',av),('pred',bv)]:
                            series=weighted_mean(seasonal_anomaly(field,days),regions['Niño 3.4']*masks[0])
                            saved[f'nino34_{who}']=pd.Series(series).rolling(30).mean().values[30:]
                        index_metrics['nino34']=time_metrics(saved['nino34_truth'],saved['nino34_pred'])
                    if var=='uo':
                        saved['surface_u_snapshot_truth']=np.where(masks[0],seasonal_anomaly(av,days)[580],np.nan)
                        saved['surface_u_snapshot_pred']=np.where(masks[0],seasonal_anomaly(bv,days)[580],np.nan)
            wet=thickness>0; safe=np.where(wet,thickness,1)
            if var=='thetao':
                for who,field in [('truth',a),('pred',b)]:
                    saved[f'ohc_{who}_{layer}']=np.sum(field*area,axis=(-2,-1))*1035*3850/1e21
            a/=safe; b/=safe
            prefix=f'{var}_{layer}'
            weights=area*thickness
            for who,field in [('truth',a),('pred',b)]:
                series=weighted_mean(field,weights)
                saved[f'{prefix}_{who}_mean']=series
                saved[f'{prefix}_{who}_detrended']=detrend(series)
            index_metrics[prefix]=time_metrics(saved[f'{prefix}_truth_detrended'],saved[f'{prefix}_pred_detrended'])
            rawdiff=b-a
            saved[f'{prefix}_rmse_time']=np.sqrt(weighted_mean(rawdiff**2,area*wet))
            saved[f'{prefix}_bias']=np.where(wet,rawdiff.mean(axis=0),np.nan)
            aa,bb=seasonal_anomaly(a,days),seasonal_anomaly(b,days)
            va,vb=aa.var(axis=0),bb.var(axis=0)
            difference=bb-aa
            direct=np.sqrt(np.mean(difference**2,axis=0))
            residual=np.sqrt(np.mean(detrend(difference,axis=0)**2,axis=0))
            saved[f'{prefix}_truth_variance']=np.where(wet,va,np.nan)
            saved[f'{prefix}_pred_variance']=np.where(wet,vb,np.nan)
            for region,rw in regions.items():
                w=rw*wet
                rows.append({'model':key,'variable':var,'layer':title,'region':region,
                             'raw_rmse':float(np.sqrt(weighted_mean(np.mean(rawdiff**2,axis=0),w))),
                             'raw_bias':float(weighted_mean(rawdiff.mean(axis=0),w)),
                             'direct_rmse':float(weighted_mean(direct,w)),
                             'detrend_rmse':float(weighted_mean(residual,w)),
                             'variance_rmse':float(np.sqrt(weighted_mean((vb-va)**2,w))),
                             'variance_corr':correlation(va,vb,w),
                             'variance_ratio':float(weighted_mean(vb,w)/weighted_mean(va,w))})
            del a,b,aa,bb,rawdiff,difference
    # Snapshot association is a diagnostic, not proof of causal leakage.
    index_metrics['deep_T_surface_u_snapshot_corr']={who:correlation(saved[f'thetao_snapshot_{who}_15'],saved[f'surface_u_snapshot_{who}'],area*masks[15]) for who in ['truth','pred']}
    np.savez_compressed(destination/'diagnostics.npz',**saved)
    pd.DataFrame(rows).to_csv(destination/'metrics.csv',index=False)
    (destination/'indices.json').write_text(json.dumps(index_metrics,indent=2,allow_nan=False))
    completion={'model':key,'source':str(path),'n_times':n,'first_prediction':str(times[0]),'last_prediction':str(times[-1]),
                'snapshot':str(times[580]),'finite_wet_fields':len(set(finite)),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    temporary=destination/'complete.tmp'; temporary.write_text(json.dumps(completion,indent=2));temporary.replace(destination/'complete.json')


def render(output):
    keys=[key for key in RUNS if (output/key/'complete.json').exists()]
    if not keys:
        raise RuntimeError('No completed diagnostic reductions')
    data={k:np.load(output/k/'diagnostics.npz') for k in keys}
    truth=data[keys[0]]; figures=output/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    dates=pd.to_datetime([s[:10] for s in truth['dates']]); captions=[]

    def save(fig,name,caption):
        fig.savefig(figures/f'{name}.png',dpi=170,bbox_inches='tight')
        fig.savefig(figures/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig); captions.append({'file':f'{name}.png','caption':caption})

    def legend(fig,ax):
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)

    for var in ['thetao','so']:
        fig,axes=plt.subplots(3,2,figsize=(11,9),layout='constrained')
        for layer,(_,_,label) in enumerate(LAYERS):
            for col,kind in enumerate(['mean','detrended']):
                ax=axes[layer,col];series=truth[f'{var}_{layer}_truth_{kind}']
                ax.plot(dates,series,color=COLORS['truth'],ls='--',label='OM4 truth')
                for key in keys:
                    ax.plot(dates,data[key][f'{var}_{layer}_pred_{kind}'],color=COLORS[key],label=RUNS[key][0],lw=1)
                ax.set(title=f'{label}: {kind}',ylabel=UNITS[var]);ax.grid(alpha=.15)
        legend(fig,axes[0,0]);save(fig,f'{var}_global_timeseries',f'Volume-weighted global {var}; raw means and separately linearly detrended series. Layer membership uses level centers as in Appendix A; no significance bands from this single rollout.')
    fig,axes=plt.subplots(3,2,figsize=(11,9),layout='constrained')
    for ax,(var,layer) in zip(axes.flat,[('thetao',0),('so',0),('thetao',2),('so',2),('uo',0),('zos',0)]):
        for key in keys:
            ax.plot(dates,data[key][f'{var}_{layer}_rmse_time'],color=COLORS[key],label=RUNS[key][0],lw=1)
        ax.set(title=f'{var}: {LAYERS[layer][2] if var!="zos" else "surface"}',ylabel=f'RMSE ({UNITS[var]})');ax.grid(alpha=.15)
    legend(fig,axes[0,0]);save(fig,'rollout_error','Area-weighted RMSE of depth-averaged raw fields throughout a continuous free ocean rollout. This is not multi-initialization forecast lead-time skill.')
    metrics=pd.concat([pd.read_csv(output/k/'metrics.csv') for k in keys],ignore_index=True)
    metrics.to_csv(output/'all_metrics.csv',index=False)
    for variable in ['thetao','so','ke']:
        fig,axes=plt.subplots(len(keys)+1,3,figsize=(12,2.4*(len(keys)+1)),layout='constrained',squeeze=False)
        for layer,(_,_,label) in enumerate(LAYERS):
            maximum=max(np.nanpercentile(data[k][f'{variable}_{layer}_pred_variance'],98) for k in keys)
            maximum=max(maximum,np.nanpercentile(truth[f'{variable}_{layer}_truth_variance'],98))
            for row,key in enumerate(['truth']+keys):
                ax=axes[row,layer]
                values=truth[f'{variable}_{layer}_truth_variance'] if key=='truth' else data[key][f'{variable}_{layer}_pred_variance']
                im=ax.pcolormesh(truth['lon'],truth['lat'],values,vmin=0,vmax=maximum,cmap='viridis',rasterized=True,shading='auto')
                ax.set_title(f'{"OM4 truth" if key=="truth" else RUNS[key][0]} | {label}',fontsize=9)
                if layer==0:ax.set_ylabel('Latitude (°N)')
                if row==len(keys):ax.set_xlabel('Longitude (°E)')
            fig.colorbar(im,ax=axes[:,layer],shrink=.7,label=f'Variance ({UNITS[variable]})²')
        save(fig,f'{variable}_variance_maps',f'{variable}: temporal variance after removing each run’s own day-of-year climatology. Common color scale per depth across models, upper 2% saturated. KE means variance of depth-averaged kinetic energy, not mean EKE.')
    for var in ['thetao','so','ke']:
        fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
        for col,(_,_,layer_label) in enumerate(LAYERS):
            selected=metrics[(metrics.variable==var)&(metrics.layer==layer_label)]
            for row,metric in enumerate(['direct_rmse','variance_corr']):
                table=selected.pivot(index='model',columns='region',values=metric).reindex(index=keys,columns=list(REGIONS))
                ax=axes[row,col]
                im=ax.imshow(table.values,aspect='auto',cmap='viridis' if row==0 else 'RdBu_r',vmin=0 if row==0 else -1,vmax=None if row==0 else 1)
                ax.set_xticks(range(len(REGIONS)),list(REGIONS),rotation=45,ha='right')
                ax.set_yticks(range(len(keys)),[RUNS[k][0] for k in keys] if col==0 else [])
                ax.set_title(f'{layer_label}: {metric}')
                fig.colorbar(im,ax=ax,shrink=.65,label=UNITS[var] if row==0 else 'Correlation')
        save(fig,f'{var}_regional_scores',f'{var}: paper-style direct RMSE (area mean of per-cell temporal RMSE on seasonal anomalies) and variance-pattern correlation. RMSE color scales vary by layer and are explicitly labeled; correlation uses −1 to 1. Row order is identical in every panel. Exact values are in all_metrics.csv. All seven regions use common wet masks.')
    fig,axes=plt.subplots(len(keys)+1,3,figsize=(12,2.4*(len(keys)+1)),layout='constrained',squeeze=False)
    for col,level in enumerate([0,10,14]):
        limit=max(np.nanpercentile(abs(data[k][f'thetao_snapshot_pred_{level}']),98) for k in keys)
        limit=max(limit,np.nanpercentile(abs(truth[f'thetao_snapshot_truth_{level}']),98))
        for row,key in enumerate(['truth']+keys):
            ax=axes[row,col];values=truth[f'thetao_snapshot_truth_{level}'] if key=='truth' else data[key][f'thetao_snapshot_pred_{level}']
            im=ax.pcolormesh(truth['lon'],truth['lat'],values,shading='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,rasterized=True)
            ax.set_title(f'{"OM4 truth" if key=="truth" else RUNS[key][0]} | {DEPTH[level]:g} m',fontsize=9)
        fig.colorbar(im,ax=axes[:,col],shrink=.7,label='Temperature anomaly (°C)')
    save(fig,'temperature_snapshot',f'Deseasonalized temperature at predicted frame 580 ({truth["dates"][580][:10]}), with actual model levels 2.5, 775 and 2400 m. Common scales within each depth; top 2% absolute anomalies saturated.')
    fig,axes=plt.subplots(len(keys)+1,2,figsize=(10,2.4*(len(keys)+1)),layout='constrained',squeeze=False)
    for col,(stem,unit) in enumerate([('thetao_snapshot','°C'),('surface_u_snapshot','m/s')]):
        def imprint(source,who):return source[f'{stem}_{who}_15'] if col==0 else source[f'{stem}_{who}']
        limit=max(np.nanpercentile(abs(imprint(data[k],'pred')),98) for k in keys)
        limit=max(limit,np.nanpercentile(abs(imprint(truth,'truth')),98))
        for row,key in enumerate(['truth']+keys):
            ax=axes[row,col];values=imprint(truth,'truth') if key=='truth' else imprint(data[key],'pred')
            im=ax.pcolormesh(truth['lon'],truth['lat'],values,shading='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,rasterized=True)
            ax.set_title(f'{"OM4 truth" if key=="truth" else RUNS[key][0]} | {"T at 3100 m" if col==0 else "u at 2.5 m"}',fontsize=9)
        fig.colorbar(im,ax=axes[:,col],shrink=.7,label=unit)
    save(fig,'imprinting_diagnostic','Deep temperature and surface u seasonal anomalies at frame 580. Similar patterns can flag imprinting, but their correlation alone does not establish causal leakage. Common scales per variable; upper 2% saturated.')
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for layer,ax in enumerate(axes):
        for key in keys:
            bias=data[key][f'ohc_pred_{layer}']-data[key][f'ohc_truth_{layer}']
            ax.plot(dates,bias,color=COLORS[key],label=RUNS[key][0],lw=1)
        ax.axhline(0,color='k',lw=.5);ax.set(title=LAYERS[layer][2],ylabel='Integrated OHC bias (ZJ)')
    legend(fig,axes[0]);save(fig,'ohc_drift','Global integrated OHC prediction minus truth, with cell areas and layer thicknesses, density 1035 kg/m³ and heat capacity 3850 J/(kg K). Units are ZJ, not ZJ/m².')
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    ax.plot(dates[30:],truth['nino34_truth'],color=COLORS['truth'],ls='--',label='OM4 truth')
    for key in keys:ax.plot(dates[30:],data[key]['nino34_pred'],color=COLORS[key],label=RUNS[key][0],lw=1)
    ax.set(ylabel='Niño 3.4 anomaly (°C)',title='Niño 3.4: 150-day running mean');legend(fig,ax)
    save(fig,'nino34','Niño 3.4, 190–240°E and 5°S–5°N; own test-period day-of-year climatology, trailing 30-frame mean, first 30 frames excluded, following viz_jupyter.')
    for var in ['thetao','zos','velocity']:
        for kind in ['temporal','spatial','acf']:
            if var=='zos' and kind=='spatial':continue
            fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
            for ax,region in zip(axes.flat,list(REGIONS)[1:]):
                def values(source,who):
                    if var=='velocity':return (source[f'uo_{who}_{region}_{kind}']+source[f'vo_{who}_{region}_{kind}'])/2
                    return source[f'{var}_{who}_{region}_{kind}']
                field='uo' if var=='velocity' else var
                x=np.arange(61)*5 if kind=='acf' else truth[f'{field}_truth_{region}_{"f" if kind=="temporal" else "k"}']
                plot=ax.plot if kind=='acf' else ax.loglog
                plot(x,values(truth,'truth'),color=COLORS['truth'],ls='--',label='OM4 truth')
                for key in keys:plot(x,values(data[key],'pred'),color=COLORS[key],label=RUNS[key][0],lw=1)
                power_unit={'thetao':'°C²','zos':'m²','velocity':'m²/s²'}[var]
                ax.set(title=region,xlabel={'acf':'Lag (days)','temporal':'Frequency (cycles/year)','spatial':'Wavenumber (cycles/km)'}[kind],ylabel={'acf':'Autocorrelation','temporal':f'PSD ({power_unit} / (cycles/year))','spatial':f'k × PSD ({power_unit} × km)'}[kind]);ax.grid(alpha=.15)
            legend(fig,axes[0,0]);save(fig,f'{var}_{kind}',f'Surface {var}, {kind}, six paper regions. Spatial: time-mean anomalies, plane removal, Hann window, strict Nyquist cutoff. Temporal: own seasonal anomalies and one-sided density. Velocity spectra are half the sum of u and v component spectra; velocity ACF is the mean of component ACFs. Land uses a shared zero mask for FFT only; regional averages exclude land.')
    for var in ['thetao','so']:
        fig,axes=plt.subplots(1,len(keys),figsize=(3.4*len(keys),4),layout='constrained',squeeze=False)
        arrays=[np.stack([data[k][f'{var}_zonal_{i}'] for i in range(19)]) for k in keys]
        maximum=max(np.nanpercentile(abs(a),98) for a in arrays)
        for ax,key,array in zip(axes.flat,keys,arrays):
            im=ax.pcolormesh(truth['lat'],DEPTH,array,cmap='RdBu_r',vmin=-maximum,vmax=maximum,shading='auto',rasterized=True)
            ax.invert_yaxis();ax.set(title=RUNS[key][0],xlabel='Latitude (°N)',ylabel='Depth (m)')
        fig.colorbar(im,ax=axes.ravel().tolist(),label=f'Mean bias ({UNITS[var]})')
        save(fig,f'{var}_zonal_bias',f'Global zonal mean {var} bias over the full rollout; wet-mask weighting along each latitude, actual model depth coordinates. Symmetric common color scale; upper 2% absolute bias saturated.')
    summary={'complete_models':keys,'missing_models':[k for k in RUNS if k not in keys], 'figures':captions,'status':'complete_pending_scientific_review' if len(keys)==len(RUNS) else 'partial_pending_inference'}
    (output/'figure_manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    # A small companion notebook reads reviewed reductions; it does not redo I/O.
    import nbformat
    notebook=nbformat.v4.new_notebook()
    notebook.cells=[nbformat.v4.new_markdown_cell('# Samudra 2 revision baseline comparison\n\n## Status\n'+summary['status']+'\n\nMissing: '+', '.join(summary['missing_models'])+'\n\nThis notebook displays computed diagnostics. Scientific interpretation and Google Doc publication require review. No synthetic results are included.'),nbformat.v4.new_code_cell('from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\nroot = Path('+repr(str(output.resolve()))+')\nmetrics = pd.read_csv(root / "all_metrics.csv")\ndisplay(metrics.query("region == \'Global\'"))')]
    for item in captions:
        notebook.cells.extend([nbformat.v4.new_markdown_cell(item['caption']),nbformat.v4.new_code_cell('display(Image(filename=str(root / "figures" / '+repr(item['file'])+')))')])
    notebook.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.write(notebook,output/'comparison.ipynb')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',nargs='*',choices=list(RUNS));parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not args.render_only:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Bulk evaluation must run in a compute allocation')
        for key in args.models or list(RUNS):
            if not (args.output/key/'complete.json').exists():reduce_model(key,args.output)
    render(args.output)


if __name__=='__main__':
    main()
