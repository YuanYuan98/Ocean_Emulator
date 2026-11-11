# TODO: Need to return step-wise losses for logging

import logging

import torch

logger = logging.getLogger(__name__)

from ocean_emulators.datasets import InferenceDataset, TrainData
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.output import ModelInferenceOutput


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        wet,
        hist,
        pred_residuals,
        last_kernel_size,
        pad,
        static_data,
        gradient_detach_interval: int,
    ) -> None:
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wet = wet.bool()
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad = pad
        self.pred_residuals = pred_residuals
        self.hist = hist
        self.static_data = static_data
        self.gradient_detach_interval = gradient_detach_interval
        self.anomaly_background_target = None

    def forward_once(self, fts):
        raise NotImplementedError()

    def configure_anomaly_background(self, statistics_path):
        if self.pred_residuals or self.hist != 1 or self.in_channels != 162 or self.out_channels != 154:
            raise ValueError('Anomaly mode requires explicit two independent heads, hist1, legacy residual false')
        from ocean_emulators.models.corrector import Correctors
        corrector = getattr(self, 'corrector', None)
        if corrector is not None and not (isinstance(corrector, Correctors) and len(corrector.correctors) == 0):
            raise ValueError('Active full-state correctors cannot act on raw anomaly heads')
        from ocean_emulators.models.anomaly_background import AnomalyBackgroundTarget
        from ocean_emulators.utils.data import Normalize
        normalizer = Normalize.get_instance()
        self.anomaly_background_target = AnomalyBackgroundTarget(
            statistics_path, normalizer._prognostic_mean_np, normalizer._prognostic_std_np)
        if self.wet.shape[0] != 154 or not torch.equal(self.wet[:77], self.wet[77:]):
            raise ValueError('Require identical named wet masks in both time slots')
        import hashlib
        actual_mask_hashes = [hashlib.sha256(m.detach().cpu().numpy().tobytes()).hexdigest() for m in self.wet[:77]]
        if actual_mask_hashes != self.anomaly_background_target.mask_sha256:
            raise ValueError('Training masks differ from background fitting masks')

    def decode_prediction(self, input_tensor, decodings, target_time_features=None):
        if self.anomaly_background_target is not None:
            return self.anomaly_background_target.decode(decodings, target_time_features, self.wet)
        if self.pred_residuals:
            return input_tensor[:, :self.out_channels] + decodings
        return decodings

    def forward(
        self,
        train_data: TrainData,
        loss_fn=None,
    ) -> torch.Tensor | list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        loss = torch.tensor(torch.nan)
        for step in range(len(train_data)):
            if step == 0:
                input_tensor = train_data.get_initial_input()
            else:
                prev_output = outputs[-1]
                if (
                    self.gradient_detach_interval > 0
                    and step % self.gradient_detach_interval == 0
                ):
                    prev_output = prev_output.detach()
                input_tensor = train_data.merge_prognostic_and_boundary(
                    prognostic=prev_output, step=step
                )

            decodings = self.forward_once(input_tensor)
            pred = self.decode_prediction(input_tensor, decodings, train_data.get_target_time_features(step))
            if loss_fn is not None:
                if self.anomaly_background_target is not None:
                    target = self.anomaly_background_target.normalized_targets(train_data.get_label(step), train_data.get_target_time_features(step), self.wet)
                    step_loss = loss_fn(decodings, target)
                else:
                    step_loss = loss_fn(pred, train_data.get_label(step))
                loss = step_loss if step == 0 else loss + step_loss

            outputs.append(pred)

        if loss_fn is None:
            return outputs
        else:
            return loss

    def inference(
        self,
        dataset: InferenceDataset,
        initial_prognostic: torch.Tensor,
        steps_completed=0,
        num_steps=None,
        epoch=None,
    ) -> ModelInferenceOutput:
        out_shape = (num_steps, *dataset[0][1].shape[1:])

        pred_tensor = torch.zeros(out_shape, device=get_device())
        initial_prognostic = initial_prognostic.to(get_device())
        target_time = dataset.get_target_time(steps_completed, num_steps)

        for step in range(num_steps):
            logger.info(
                f"Inference [epoch {epoch}]: Rollout step {steps_completed + step} "
                f"of {steps_completed + num_steps - 1}."
            )
            if step == 0:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=initial_prognostic,
                    step=steps_completed,
                )
            else:
                input_tensor = dataset.merge_prognostic_and_boundary(
                    prognostic=pred_tensor[step - 1].unsqueeze(0),
                    step=steps_completed + step,
                )

            decodings = self.forward_once(input_tensor)
            pred = self.decode_prediction(input_tensor, decodings, dataset.get_target_time_features(steps_completed + step))

            pred_tensor[step] = pred

        target_tensor = dataset.inference_target(
            slice(steps_completed, steps_completed + num_steps)
        ).to(device=get_device())

        IO = ModelInferenceOutput(pred_tensor, target_tensor, target_time)
        return IO
