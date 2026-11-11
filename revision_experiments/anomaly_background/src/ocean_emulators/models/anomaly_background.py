"""Two independent anomaly heads, reconstructed against a train-only background."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ocean_emulators.constants import PROGNOSTIC_VARS

class AnomalyBackgroundTarget(nn.Module):
    mode = 'two_independent_anomaly_heads_train_only_six_term_background_v1'
    def __init__(self, statistics_path, input_mean, input_std):
        super().__init__()
        p=Path(statistics_path); spec=json.loads(p.read_text())
        self.statistics_sha256=hashlib.sha256(p.read_bytes()).hexdigest()
        if spec['status']!='COMPLETE_TRAIN_ONLY_BACKGROUND' or spec['fit_training_states']!=2829 or spec['calendar']!='julian':
            raise ValueError('Require completed train-only Julian background')
        if spec['channels']!=list(PROGNOSTIC_VARS['thermo_dynamic_all']) or spec['fixed_harmonics']!=2:
            raise ValueError('Background channel order or basis differs')
        mean=torch.as_tensor(input_mean,dtype=torch.float64).reshape(-1)
        std=torch.as_tensor(input_std,dtype=torch.float64).reshape(-1)
        tau=torch.as_tensor([s['residual_std_physical'] for s in spec['statistics']],dtype=torch.float64)
        if mean.shape!=(77,) or std.shape!=(77,) or tau.shape!=(77,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not torch.isfinite(tau).all() or (std<=0).any() or (tau<=0).any():
            raise ValueError('Invalid frozen normalization/anomaly scale')
        coefficients=[]; hashes=[]
        for name,s in zip(spec['channels'],spec['statistics'],strict=True):
            f=p.parent/f'background_{name}.npy'; h=hashlib.sha256(f.read_bytes()).hexdigest()
            if h!=s['coefficient_sha256']: raise ValueError('Background coefficient hash mismatch')
            a=np.load(f,allow_pickle=False)
            if a.ndim!=3 or a.shape[0]!=6 or not np.isfinite(a).all(): raise ValueError('Invalid coefficients')
            coefficients.append(a); hashes.append(h)
        physical=torch.from_numpy(np.stack(coefficients,axis=1)).to(torch.float64)
        normalized=physical/std[None,:,None,None]
        normalized[0]=(physical[0]-mean[:,None,None])/std[:,None,None]
        self.register_buffer('normalized_coefficients',normalized.float())
        self.register_buffer('input_state_mean',mean)
        self.register_buffer('input_state_std',std)
        self.register_buffer('physical_anomaly_std',tau)
        self.register_buffer('state_anomaly_scale',tau/std)
        center=torch.as_tensor(spec['feature_center'],dtype=torch.float64)
        center[0]+=spec['origin_days_since_1958_01_01']/365.25
        self.register_buffer('absolute_feature_center',center)
        self.coefficient_sha256=hashes
        self.mask_sha256=spec['mask_sha256']
    def background(self, raw, target_time_features, wet):
        if raw.ndim!=4 or raw.shape[1]!=154: raise ValueError('Expected B154HW heads')
        if target_time_features is None or tuple(target_time_features.shape)!=(raw.shape[0],2,5):
            raise ValueError('Missing/misaligned actual target-date metadata')
        if tuple(raw.shape[-2:])!=tuple(self.normalized_coefficients.shape[-2:]): raise ValueError('Grid differs')
        ft=target_time_features.to(device=raw.device,dtype=torch.float64)
        if not torch.isfinite(ft).all(): raise ValueError('Invalid date features')
        centered=ft-self.absolute_feature_center
        basis=torch.cat((torch.ones((*centered.shape[:2],1),device=raw.device,dtype=torch.float64),centered),dim=-1).to(raw.dtype)
        b=torch.einsum('btf,fchw->btchw',basis,self.normalized_coefficients.to(raw.dtype)).flatten(1,2)
        return torch.where(wet,b,0)
    def decode(self, raw, target_time_features, wet):
        b=self.background(raw,target_time_features,wet)
        q=self.state_anomaly_scale.to(raw).repeat(2)[None,:,None,None]
        return torch.where(wet,b+q*raw,0)
    def normalized_targets(self, label, target_time_features, wet):
        b=self.background(label,target_time_features,wet)
        q=self.state_anomaly_scale.to(label).repeat(2)[None,:,None,None]
        return torch.where(wet,(label-b)/q,0)
    def specification(self):
        return {'mode':self.mode,'statistics_sha256':self.statistics_sha256,'coefficient_sha256':self.coefficient_sha256,'input_state_mean':self.input_state_mean.tolist(),'input_state_std':self.input_state_std.tolist(),'physical_anomaly_std':self.physical_anomaly_std.tolist(),'state_anomaly_scale':self.state_anomaly_scale.tolist(),'mask_sha256':self.mask_sha256,'target_date_metadata':'actual Julian target dates, two independent heads','background_fit':'2829 original training states only, fixed intercept/trend/annual/semiannual','anomaly_scale':'train wet population residual RMS of serialized background','feedback':'reconstructed normalized full states; never accumulate anomaly heads'}
    def _load_from_state_dict(self,state_dict,prefix,*args,**kwargs):
        for name,value in self.named_buffers(recurse=False):
            key=prefix+name
            if key not in state_dict or not torch.equal(state_dict[key].cpu(),value.cpu()):
                raise ValueError(f'Immutable background/normalization buffer changed: {name}')
        return super()._load_from_state_dict(state_dict,prefix,*args,**kwargs)
