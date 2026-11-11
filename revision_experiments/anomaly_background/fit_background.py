"""Training-only, fixed six-feature spatial background. Never reads held-out fields."""
from pathlib import Path
import argparse, datetime, hashlib, json, resource, time
import numpy as np
import cftime
import zarr

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/datasets/onedeg')
CHANNELS = [f'{v}_{k}' for v in ('uo','vo','thetao','so') for k in range(19)] + ['zos']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def feature_values(dates, origin):
    """Actual Julian days for trend; calendar-year phase, with leap/fractional day."""
    units='days since 1958-01-01 00:00:00'
    days=np.asarray(cftime.date2num(dates,units,calendar='julian'),dtype=np.float64)
    phase=[]
    for d in dates:
        start=cftime.DatetimeJulian(d.year,1,1)
        end=cftime.DatetimeJulian(d.year+1,1,1)
        a=float(cftime.date2num(d,units,calendar='julian'))
        b=float(cftime.date2num(start,units,calendar='julian'))
        e=float(cftime.date2num(end,units,calendar='julian'))
        phase.append((a-b)/(e-b))
    p=2*np.pi*np.asarray(phase)
    return np.stack(((days-origin)/365.25,np.sin(p),np.cos(p),np.sin(2*p),np.cos(2*p)),axis=1)

class Moments:
    """Float64 Chan joint covariance: stable under large salinity intercepts."""
    def __init__(self,ncell):
        self.n=0; self.mx=np.zeros(5); self.my=np.zeros(ncell)
        self.xx=np.zeros((5,5)); self.xy=np.zeros((5,ncell)); self.yy=np.zeros(ncell)
    def update(self,x,y):
        assert x.shape==(len(y),5) and np.isfinite(x).all() and np.isfinite(y).all()
        n=len(y); mx=x.mean(0); my=y.mean(0); xc=x-mx; yc=y-my
        xx=xc.T@xc; xy=xc.T@yc; yy=np.einsum('ij,ij->j',yc,yc)
        if self.n:
            w=self.n*n/(self.n+n); dx=mx-self.mx; dy=my-self.my
            self.xx+=xx+w*np.outer(dx,dx); self.xy+=xy+w*np.outer(dx,dy); self.yy+=yy+w*dy*dy
            self.mx+=dx*n/(self.n+n); self.my+=dy*n/(self.n+n)
        else:
            self.mx=mx; self.my=my; self.xx=xx; self.xy=xy; self.yy=yy
        self.n+=n
    def finish(self):
        beta=np.linalg.solve(self.xx,self.xy)
        explained=np.einsum('ij,ij->j',beta,self.xy)
        sse=self.yy-explained
        tol=128*np.finfo(float).eps*np.maximum(self.yy,np.abs(explained))
        if (sse < -tol).any(): raise AssertionError('Substantive negative residual SSE')
        sse=np.maximum(sse,0)
        # Coefficients are intercept at global training feature center, then five slopes.
        return np.concatenate((self.my[None,:],beta),0),sse

def fixture():
    rng=np.random.default_rng(15)
    dates=[cftime.DatetimeJulian(y,m,3,12) for y in range(1975,2014) for m in range(1,13)]
    origin=float(cftime.date2num(dates[0],'days since 1958-01-01',calendar='julian'))
    x=feature_values(dates,origin); y=35+ x@rng.normal(size=(5,3))*.01+rng.normal(size=(len(x),3))*.0001
    acc=Moments(3)
    for i in range(0,len(x),25): acc.update(x[i:i+25],y[i:i+25])
    coeff,sse=acc.finish(); design=np.column_stack((np.ones(len(x)),x-acc.mx)); direct=np.linalg.lstsq(design,y,rcond=None)[0]
    assert np.allclose(coeff,direct,rtol=1e-9,atol=1e-11)
    err=((y-design@coeff)**2).sum(0)
    assert np.allclose(sse,err,rtol=1e-7,atol=1e-12)
    # Julian 1900 is leap; phase uses actual day fraction, not Gregorian or index/73.
    z=feature_values([cftime.DatetimeJulian(1900,2,28,12),cftime.DatetimeJulian(1900,2,29,12),cftime.DatetimeJulian(1900,3,1,12)],origin)
    assert np.allclose(np.diff(z[:,0]),1/365.25)
    return {'status':'PASS_SYNTHETIC_CHAN_VS_DIRECT_OLS_AND_JULIAN_LEAP','max_coefficient_error':float(abs(coeff-direct).max()),'max_sse_error':float(abs(sse-err).max())}

def run(mode,out,chunk):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists(): raise FileExistsError('Never overwrite a completed fit')
    started=time.monotonic(); g=zarr.open_consolidated(str(DATA/'OM4.zarr'),mode='r'); attrs=dict(g['time'].attrs)
    assert attrs['calendar']=='julian'
    rawtime=np.asarray(g['time'][:]); dates=np.asarray(cftime.num2date(rawtime,attrs['units'],calendar='julian'),dtype=object)
    idx=np.asarray([i for i,d in enumerate(dates) if '1975-01-03'<=str(d)<'2013-10-05'])
    assert len(idx)==2829 and np.all(np.diff(idx)==1)
    origin=float(cftime.date2num(dates[idx[0]],'days since 1958-01-01',calendar='julian'))
    x=feature_values(list(dates[idx]),origin); center=x.mean(0)
    selected=CHANNELS if mode=='full' else ['uo_0','thetao_0','so_18']
    stats=[]; coeffs=[]; maskhashes=[]; masks=[]; nbytes=0
    for name in selected:
        k=int(name.rsplit('_',1)[1]) if name!='zos' else 0
        a=g[f'mask_{k}']; wet=np.asarray(a[0] if a.ndim==3 else a[:],dtype=bool)
        assert wet.any(); masks.append(wet); maskhashes.append(hashlib.sha256(wet.tobytes()).hexdigest())
        acc=Moments(int(wet.sum())); source=g[name]; assert source.shape[1:]==wet.shape
        ranges=[(i,min(i+chunk,len(idx))) for i in range(0,len(idx),chunk)] if mode=='full' else [(i,min(i+8,len(idx))) for i in np.linspace(0,len(idx)-8,6,dtype=int)]
        direct_x=[]; direct_y=[]
        for lo,hi in ranges:
            v=np.asarray(source[int(idx[lo]):int(idx[hi-1])+1],dtype=np.float64)
            assert v.shape[0]==hi-lo
            y=v[:,wet]; assert np.isfinite(y).all(), f'Nonfinite training wet data: {name}'
            acc.update(x[lo:hi],y); direct_x.append(x[lo:hi]); direct_y.append(y[:,:3].copy())
            nbytes+=int(v.size*source.dtype.itemsize)
            progress={'status':'RUNNING','channel':name,'channels_complete':len(stats),'frames':acc.n,'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'logical_bytes_read':nbytes}
            (out/'progress.json.tmp').write_text(json.dumps(progress,indent=2)); (out/'progress.json.tmp').replace(out/'progress.json')
        c,sse=acc.finish()
        # Center representation is global training mean even in sparse probe mode.
        c[0]+= (center-acc.mx)@c[1:]
        dx=np.concatenate(direct_x); dy=np.concatenate(direct_y); des=np.column_stack((np.ones(len(dx)),dx-center))
        direct=np.linalg.lstsq(des,dy,rcond=None)[0]
        assert np.allclose(c[:,:3],direct,rtol=2e-8,atol=2e-10)
        direct_sse=((dy-des@c[:,:3])**2).sum(0)
        assert np.allclose(sse[:3],direct_sse,rtol=1e-5,atol=1e-13)
        tau=float(np.sqrt(sse.sum()/(acc.n*wet.sum())))
        assert np.isfinite(tau) and tau>0
        image=np.zeros((6,*wet.shape),dtype=np.float32); image[:,wet]=c.astype(np.float32)
        # Training residual scale uses the actually serialized float32 background.
        delta=image[:,wet].astype(np.float64)-c
        # Centered residuals are orthogonal to the OLS design: rounding adds this exact quadratic term.
        gram=np.zeros((6,6)); gram[0,0]=acc.n
        mx=acc.mx-center; gram[0,1:]=acc.n*mx; gram[1:,0]=acc.n*mx
        gram[1:,1:]=acc.xx+acc.n*np.outer(mx,mx)
        rounding_sse=float(np.einsum('ic,ij,jc->',delta,gram,delta))
        serialized_tau=float(np.sqrt((sse.sum()+rounding_sse)/(acc.n*wet.sum())))
        np.save(out/f'background_{name}.npy',image,allow_pickle=False)
        coeffs.append(str(out/f'background_{name}.npy'))
        stats.append({'channel':name,'states':acc.n,'wet_cells':int(wet.sum()),'residual_std_physical':serialized_tau,'ols_residual_std_physical':tau,'coefficient_rounding_added_sse':rounding_sse,'coefficient_sha256':sha(coeffs[-1]),'direct_first_three_cell_max_coeff_error':float(abs(c[:,:3]-direct).max()),'direct_first_three_cell_max_sse_error':float(abs(sse[:3]-direct_sse).max())})
        print(json.dumps(stats[-1]),flush=True)
    spec={'status':'COMPLETE_TRAIN_ONLY_BACKGROUND' if mode=='full' else 'PROBE_ONLY_NOT_TRAINING_ASSET','calendar':'julian','feature_names':['intercept','elapsed_years','annual_sin','annual_cos','semiannual_sin','semiannual_cos'],'fixed_harmonics':2,'origin_days_since_1958_01_01':origin,'feature_center':center.tolist(),'fit_start_inclusive':str(dates[idx[0]]),'fit_end_inclusive':str(dates[idx[-1]]),'train_end_exclusive':'2013-10-05','fit_training_states':2829,'channels':selected,'statistics':stats,'mask_sha256':maskhashes,'time_coordinate_sha256':hashlib.sha256(rawtime.tobytes()).hexdigest(),'source':str(DATA/'OM4.zarr'),'source_time_attrs':attrs,'source_sha256':sha(__file__),'logical_bytes_read':nbytes,'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'input_normalization_policy':'Original frozen state/forcing mean and std unchanged; their original fit range is not established. Only new background and residual scale are train-only.','extrapolation_policy':'Evaluate this fixed six-term function at target dates; no held-out truth fit; linear-trend extrapolation is not learned forced response.'}
    (out/'background_specification.json').write_text(json.dumps(spec,indent=2)+'\n')
    receipt={'status':spec['status'],'specification_sha256':sha(out/'background_specification.json'),'fixture':fixture(),**{k:spec[k] for k in ('logical_bytes_read','elapsed_seconds','peak_rss_bytes','source_sha256')}}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); (out/'progress.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fixture','probe','full'],required=True);p.add_argument('--output',type=Path);p.add_argument('--chunk',type=int,default=25);a=p.parse_args()
    assert 1<=a.chunk<=25
    if a.mode=='fixture': print(json.dumps(fixture()))
    else:
        assert a.output is not None
        run(a.mode,a.output,a.chunk)
