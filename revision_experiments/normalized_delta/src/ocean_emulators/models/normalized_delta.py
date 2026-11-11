"""Explicit normalized native-record increments; frozen input normalization.

Two heads are adjacent increments, with the second anchored to the first
prediction. Targets retain gradients through all model-produced anchors.
"""
import hashlib
import json
from pathlib import Path
import torch
from ocean_emulators.constants import PROGNOSTIC_VARS


class NormalizedDeltaTarget(torch.nn.Module):
    mode = 'two_head_sequential_model_anchored_normalized_delta_v1'

    def __init__(self, statistics_path):
        super().__init__()
        path = Path(statistics_path)
        self.statistics_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.statistics_path = str(path.resolve())
        data = json.loads(path.read_text())
        if data['channel_names'] != PROGNOSTIC_VARS['thermo_dynamic_all']:
            raise ValueError('Wrong named physical channel order')
        if data['increment_mean_subtracted'] is not False or data['training_states'] != 2829:
            raise ValueError('Unexpected increment statistics definition')
        tau = torch.tensor(data['physical_increment_std'], dtype=torch.float64)
        sigma = torch.tensor(data['input_state_std'], dtype=torch.float64)
        if tau.shape != (77,) or sigma.shape != (77,) or not torch.isfinite(tau).all() or not torch.isfinite(sigma).all() or not (tau > 0).all() or not (sigma > 0).all():
            raise ValueError('Invalid named positive normalization scales')
        self.register_buffer('physical_increment_std', tau)
        self.register_buffer('input_state_std', sigma)
        self.register_buffer('state_delta_scale', tau / sigma)

    def _parts(self, input_tensor, raw_heads):
        if input_tensor.ndim != 4 or input_tensor.shape[1] != 162 or raw_heads.shape[1] != 154 or raw_heads.shape[0] != input_tensor.shape[0]:
            raise ValueError('Expected2time-major77state+2x4forcing inputs and2x77heads')
        return input_tensor[:, 77:154], raw_heads[:, :77], raw_heads[:, 77:], self.state_delta_scale.to(input_tensor).view(1,77,1,1)

    def decode(self, input_tensor, raw_heads, wet):
        anchor, d1, d2, q = self._parts(input_tensor, raw_heads)
        z1 = torch.where(wet[:77], anchor + q*d1, 0.0)
        z2 = torch.where(wet[77:], z1 + q*d2, 0.0)
        return torch.cat((z1, z2), dim=1)

    def normalized_targets(self, input_tensor, raw_heads, label, wet):
        anchor, d1, _, q = self._parts(input_tensor, raw_heads)
        z1 = torch.where(wet[:77], anchor + q*d1, 0.0)
        # NO detach: gradients through preceding predicted anchors are essential.
        t1 = (label[:, :77] - anchor) / q
        t2 = (label[:, 77:] - z1) / q
        return torch.where(wet, torch.cat((t1,t2),dim=1), 0.0)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        for name in ('physical_increment_std','input_state_std','state_delta_scale'):
            key=prefix+name
            if key in state_dict and not torch.equal(state_dict[key].detach().cpu(),getattr(self,name).detach().cpu()):
                error_msgs.append('Immutable normalized-delta scale mismatch: '+key)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def specification(self):
        return {'mode':self.mode,'statistics_sha256':self.statistics_sha256,
                'physical_increment_std':self.physical_increment_std.detach().cpu(),
                'input_state_std':self.input_state_std.detach().cpu(),
                'state_delta_scale':self.state_delta_scale.detach().cpu(),
                'increment_mean_subtracted':False,'anchor_gradient_detached':False}
