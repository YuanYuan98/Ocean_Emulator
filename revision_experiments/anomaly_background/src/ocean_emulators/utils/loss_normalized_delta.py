"""Unweighted masked MSE on explicit normalized delta-head predictions/targets."""
import hashlib
from pathlib import Path
import torch
from ocean_emulators.utils.loss_controlled import FixedPhysicalLoss

class NormalizedDeltaLoss(FixedPhysicalLoss):
    def __init__(self, loss_fn, *, device, num_channels, statistics_path):
        if num_channels != 77:
            raise ValueError('Expected77 named physical variables per output head')
        super().__init__(loss_fn,torch.ones(num_channels,device=device))
        self.statistics_sha256=hashlib.sha256(Path(statistics_path).read_bytes()).hexdigest()

    def state_dict(self):
        return {**super().state_dict(),'objective_mode':'explicit_normalized_delta_targets_v1',
                'statistics_sha256':self.statistics_sha256}

    def load_state_dict(self,state):
        if state['objective_mode']!='explicit_normalized_delta_targets_v1' or state['statistics_sha256']!=self.statistics_sha256:
            raise ValueError('Resumed normalized-delta objective/statistics differ')
        super().load_state_dict(state)
