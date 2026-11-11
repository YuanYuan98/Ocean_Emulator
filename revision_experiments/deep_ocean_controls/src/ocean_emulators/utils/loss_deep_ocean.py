"""Original dynamic weighting applied only to the ten supervised deep T/S fields."""
import torch
from ocean_emulators.utils.loss import DynamicLoss

DEEP_INDICES = (52, 53, 54, 55, 56, 71, 72, 73, 74, 75)


class DeepDynamicLoss:
    objective_mode = 'deep_ts_dynamic'
    normalization = 'mean_over_2x10_active_channels_sum_over_4_calls'

    def __init__(self, base, *, device, num_channels, limit=100.0):
        if num_channels != 77 or limit != 100.0:
            raise ValueError('Fixed deep-control contract requires 77 layout and limit100')
        self.deep_indices = DEEP_INDICES
        self.indices = DEEP_INDICES + tuple(i + 77 for i in DEEP_INDICES)
        self.inner = DynamicLoss(base, limit=limit, device=device, num_channels=10)
        self._device = device
        self.limit = limit

    def _selected(self, pred, target, wet):
        if pred.shape != target.shape or pred.shape[1] != 154 or wet.shape[0] != 154:
            raise ValueError('Deep loss requires both time-major output slots')
        return pred[:, self.indices], target[:, self.indices], wet[list(self.indices)]

    def __call__(self, pred, target, wet):
        p, t, w = self._selected(pred, target, wet)
        selected = self.inner(p, t, w)
        # The unchanged Stepper averages 154 slots. This makes its scalar equal
        # to the mean of exactly 20 supervised slots, not 154 diluted slots.
        full = selected.new_zeros(154)
        full[list(self.indices)] = selected * (77.0 / 10.0)
        return full

    def update(self, pred, target, wet):
        p, t, w = self._selected(pred, target, wet)
        self.inner.update(p, t, w)

    def loss_scale_per_channel(self):
        # Inactive unit entries avoid 0/0 in generic objective logging; these
        # entries are never used in the loss or the dynamic update.
        full = torch.ones(77, device=self._device)
        full[list(self.deep_indices)] = self.inner.loss_scale_per_channel() * (77.0 / 10.0)
        return full

    def state_dict(self):
        return {'objective_mode': self.objective_mode,
                'deep_indices': list(self.deep_indices), 'limit': self.limit,
                'normalization': self.normalization,
                **self.inner.state_dict()}

    def load_state_dict(self, state):
        if (state['objective_mode'] != self.objective_mode
                or state['deep_indices'] != list(self.deep_indices)
                or state['limit'] != self.limit
                or state['normalization'] != self.normalization
                or state['per_channel_scale'].shape != (10,)):
            raise ValueError('Deep loss checkpoint contract mismatch')
        self.inner.load_state_dict(state)
