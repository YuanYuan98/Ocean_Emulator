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
        self.normalized_delta_target = None
        self.deep_ocean_control = None

    def forward_once(self, fts):
        raise NotImplementedError()

    def configure_deep_ocean_control(self, mode, contract_path):
        if self.pred_residuals or self.hist != 1 or self.in_channels != 162 or self.out_channels != 154:
            raise ValueError('Deep controls require the original full-state 162/154 layout')
        if self.normalized_delta_target is not None:
            raise ValueError('Deep controls do not use a delta/background target')
        from ocean_emulators.models.deep_ocean_control import DeepOceanControl
        self.deep_ocean_control = DeepOceanControl(mode, contract_path)

    def configure_normalized_delta(self, statistics_path):
        if self.pred_residuals or self.hist != 1 or self.in_channels != 162 or self.out_channels != 154:
            raise ValueError('Delta mode requires explicit2head hist1 and legacy residual flag false')
        from ocean_emulators.models.corrector import Correctors
        corrector=getattr(self,'corrector',None)
        if corrector is not None and not (isinstance(corrector,Correctors) and len(corrector.correctors)==0):
            raise ValueError('Active full-state correctors cannot act on raw delta heads')
        from ocean_emulators.models.normalized_delta import NormalizedDeltaTarget
        from ocean_emulators.utils.data import Normalize
        transform = NormalizedDeltaTarget(statistics_path)
        actual = torch.as_tensor(Normalize.get_instance()._prognostic_std_np, dtype=torch.float64)
        if not torch.equal(actual, transform.input_state_std):
            raise ValueError('Actual frozen input normalization differs from target specification')
        if self.wet.shape[0] != 154 or not torch.equal(self.wet[:77],self.wet[77:]):
            raise ValueError('Require identical named wet masks in both time slots')
        self.normalized_delta_target = transform

    def decode_prediction(self, input_tensor, decodings):
        if self.normalized_delta_target is not None:
            return self.normalized_delta_target.decode(input_tensor, decodings, self.wet)
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
            if self.deep_ocean_control is not None:
                previous = outputs[-1] if step else None
                if previous is not None and self.gradient_detach_interval > 0 and step % self.gradient_detach_interval == 0:
                    previous = previous.detach()
                input_tensor = self.deep_ocean_control.prepare_input(train_data.get_input(step), previous)
            elif step == 0:
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
            pred = self.decode_prediction(input_tensor, decodings)
            if loss_fn is not None:
                if self.normalized_delta_target is not None:
                    target = self.normalized_delta_target.normalized_targets(input_tensor, decodings, train_data.get_label(step), self.wet)
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
        if self.deep_ocean_control is not None:
            raise RuntimeError('Deep controls require matched truth-history evaluation or explicitly truth-completed deep AR; unrestricted full-field inference is invalid')
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
            pred = self.decode_prediction(input_tensor, decodings)

            pred_tensor[step] = pred

        target_tensor = dataset.inference_target(
            slice(steps_completed, steps_completed + num_steps)
        ).to(device=get_device())

        IO = ModelInferenceOutput(pred_tensor, target_tensor, target_time)
        return IO
