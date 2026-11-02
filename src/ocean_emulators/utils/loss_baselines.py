"""Loss-only revision baselines; preserve the original training/checkpoint path."""

import json
from pathlib import Path

import torch

from ocean_emulators.utils.loss import LossFn


def _checked_weights(weights: torch.Tensor, size: int) -> None:
    if weights.shape != (size,):
        raise ValueError(
            f"Expected {size} flattened output weights, got {weights.shape}"
        )
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Loss weights must be finite and strictly positive")


class FixedOutputWeightedLoss:
    """Apply weights in the actual flattened (time, variable) output order.

    Frozen dynamic weights are expanded with the original grouping before loading.
    Tendency weights repeat physical-channel weights for each output time instead.
    The no-op update deliberately preserves Trainer's post-optimizer forward pass,
    including its BatchNorm updates, when comparing against DynamicLoss.
    """

    def __init__(self, loss_fn: LossFn, weights: torch.Tensor):
        _checked_weights(weights, weights.numel())
        self.loss_fn = loss_fn
        self.weights = weights.detach().clone()

    @classmethod
    def from_file(
        cls, loss_fn: LossFn, path: Path, device: torch.device, num_outputs: int
    ) -> "FixedOutputWeightedLoss":
        payload = json.loads(path.read_text())
        weights = torch.tensor(
            payload["output_weights"], device=device, dtype=torch.float32
        )
        _checked_weights(weights, num_outputs)
        return cls(loss_fn, weights)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = self.loss_fn(pred, target)
        if errors.shape != self.weights.shape:
            raise ValueError(
                f"Loss/weight shape mismatch: {errors.shape}, {self.weights.shape}"
            )
        return errors * self.weights

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Keep the weights fixed while retaining the legacy extra forward pass."""

    def loss_scale_per_channel(self) -> torch.Tensor:
        return self.weights

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"output_weights": self.weights.detach().cpu().clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        saved = state["output_weights"].to(self.weights)
        _checked_weights(saved, self.weights.numel())
        if not torch.equal(saved, self.weights):
            raise ValueError(
                "Resumed fixed weights differ from the configured weight artifact"
            )


class AdaLoss(FixedOutputWeightedLoss):
    """Hu et al. inverse-EMA-loss rule applied to the existing loss groups.

    Keep the original DynamicLoss grouping and update cadence for a loss-only
    comparison. Unlike DynamicLoss, smooth errors before inversion and do not cap
    the weight ratio. Unit initial errors give unit initial weights.
    """

    def __init__(
        self,
        loss_fn: LossFn,
        *,
        num_channels: int,
        num_outputs: int,
        device: torch.device,
        ema_window: int = 25,
        epsilon: float = 1e-8,
    ):
        if num_outputs % num_channels or ema_window < 1 or epsilon <= 0:
            raise ValueError("Invalid AdaLoss grouping or smoothing configuration")
        super().__init__(loss_fn, torch.ones(num_outputs, device=device))
        self.ema_losses = torch.ones(num_channels, device=device)
        self.ema_window = ema_window
        self.epsilon = epsilon

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        from ocean_emulators.utils.distributed import all_reduce_mean

        errors = self.loss_fn(pred, target)
        if errors.shape != self.weights.shape:
            raise ValueError("AdaLoss output shape changed")
        errors = errors.reshape(self.ema_losses.numel(), -1).mean(dim=1)
        all_reduce_mean(errors)
        if not torch.isfinite(errors).all() or (errors < 0).any():
            raise ValueError("AdaLoss received invalid errors")
        self.ema_losses.mul_(1 - 1 / self.ema_window).add_(errors / self.ema_window)
        group_weights = self.ema_losses.clamp_min(self.epsilon).reciprocal()
        self.weights.copy_(
            group_weights.repeat_interleave(
                self.weights.numel() // group_weights.numel()
            )
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            **super().state_dict(),
            "ema_losses": self.ema_losses.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        weights = state["output_weights"].to(self.weights)
        errors = state["ema_losses"].to(self.ema_losses)
        _checked_weights(weights, self.weights.numel())
        if (
            errors.shape != self.ema_losses.shape
            or not torch.isfinite(errors).all()
            or (errors < 0).any()
        ):
            raise ValueError("Invalid saved AdaLoss error state")
        self.weights.copy_(weights)
        self.ema_losses.copy_(errors)
