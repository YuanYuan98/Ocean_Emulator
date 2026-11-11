"""Proposed physical-channel loss controls; isolated, not approved for training.

All methods preserve the released masked full-grid MSE, two time-major output
slots, and Trainer's post-optimizer first-forward recomputation. The fixed loss
has a no-op update so that this extra forward and its BatchNorm effects remain.
"""

import json
from pathlib import Path

import torch

from ocean_emulators.constants import PROGNOSTIC_VARS
from ocean_emulators.utils.loss import LossFnWithMask


def checked_vector(value, size):
    if value.shape != (size,) or not torch.isfinite(value).all() or not (value > 0).all():
        raise ValueError("Expected a finite positive physical-channel weight vector")


class FixedPhysicalLoss:
    def __init__(self, loss_fn: LossFnWithMask, weights: torch.Tensor):
        checked_vector(weights, weights.numel())
        self.loss_fn = loss_fn
        self.weights = weights.detach().clone()

    @classmethod
    def from_file(cls, loss_fn, path, device, num_channels):
        payload = json.loads(Path(path).read_text())
        if payload["layout"] != "two_time_major_slots_same_physical_channel_weights":
            raise ValueError("Ambiguous channel layout")
        if (payload["channel_names"] != PROGNOSTIC_VARS["thermo_dynamic_all"]
                or len(payload["channel_names"]) != num_channels):
            raise ValueError("Wrong physical-channel names or order")
        weights = torch.tensor(payload["channel_weights"], dtype=torch.float32, device=device)
        checked_vector(weights, num_channels)
        return cls(loss_fn, weights)

    def _errors(self, pred, target, wet):
        errors = self.loss_fn(pred, target, wet)
        if errors.shape != (2 * self.weights.numel(),):
            raise ValueError("Expected exactly two time-major output slots")
        if not torch.isfinite(errors).all() or (errors < 0).any():
            raise ValueError("Invalid raw channel MSE")
        return errors.reshape(2, self.weights.numel())

    def __call__(self, pred, target, wet):
        return (self._errors(pred, target, wet) * self.weights).reshape(-1)

    def update(self, pred, target, wet):
        """No weight update; Trainer still executes its original extra forward."""

    def loss_scale_per_channel(self):
        return self.weights

    def state_dict(self):
        return {"physical_channel_weights": self.weights.detach().cpu().clone()}

    def load_state_dict(self, state):
        saved = state["physical_channel_weights"].to(self.weights)
        checked_vector(saved, self.weights.numel())
        if not torch.equal(saved, self.weights):
            raise ValueError("Resumed fixed weights differ from configured weights")


class HuPhysicalLoss(FixedPhysicalLoss):
    """Declared ocean adaptation of Hu et al. Appendix C.1/C.2.

Mean raw errors across output slots/ranks, then EMA the errors (m=.9).
Weights = .95 * min(EMA_errors)/EMA_errors + .05, hence in [.05,1].
Unit initial error estimates/weights and epsilon=1e-8 are declared adaptation
choices, not claimed as recovered original Hu initialization. There is no
extra Samudra cap or normalization of the absolute weight scale.
"""

    def __init__(self, loss_fn, *, device, num_channels):
        super().__init__(loss_fn, torch.ones(num_channels, device=device))
        self.ema_errors = torch.ones(num_channels, device=device)
        self.updates = 0

    @torch.no_grad()
    def update(self, pred, target, wet):
        from ocean_emulators.utils.distributed import all_reduce_mean, get_world_size

        errors = self._errors(pred, target, wet).mean(dim=0)
        if get_world_size() > 1:
            all_reduce_mean(errors)
        self.ema_errors.mul_(0.9).add_(errors, alpha=0.1)
        safe = self.ema_errors.clamp_min(1e-8)
        self.weights.copy_(0.95 * safe.min() / safe + 0.05)
        self.updates += 1

    def state_dict(self):
        return {**super().state_dict(),
                "ema_errors": self.ema_errors.detach().cpu().clone(),
                "updates": self.updates,
                "definition": "Hu_C1_C2_m0.9_gamma0.05_alpha_min_unit_initial_v1"}

    def load_state_dict(self, state):
        if state["definition"] != "Hu_C1_C2_m0.9_gamma0.05_alpha_min_unit_initial_v1":
            raise ValueError("Resumed Hu definition differs")
        errors = state["ema_errors"].to(self.ema_errors)
        weights = state["physical_channel_weights"].to(self.weights)
        checked_vector(errors.clamp_min(1e-8), self.ema_errors.numel())
        if (errors < 0).any() or not isinstance(state["updates"], int) or state["updates"] < 0:
            raise ValueError("Invalid Hu state")
        safe = errors.clamp_min(1e-8)
        expected = 0.95 * safe.min() / safe + 0.05
        checked_vector(weights, self.weights.numel())
        if not torch.equal(expected, weights):
            raise ValueError("Saved weights inconsistent with stored Hu error estimates")
        self.ema_errors.copy_(errors)
        self.weights.copy_(weights)
        self.updates = state["updates"]
