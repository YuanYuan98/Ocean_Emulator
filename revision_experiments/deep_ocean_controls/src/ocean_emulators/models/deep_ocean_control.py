"""Two fixed deep-temperature/salinity input controls; no learnable parameters."""
import hashlib
import json
from pathlib import Path
import torch
from ocean_emulators.constants import DEPTH_LEVELS, PROGNOSTIC_VARS

MODES = ('all_ocean_input', 'deep_ocean_input')


class DeepOceanControl:
    def __init__(self, mode, contract_path):
        if mode not in MODES:
            raise ValueError(f'Unknown deep-ocean control: {mode}')
        path = Path(contract_path)
        self.contract_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.contract = json.loads(path.read_text())
        names = PROGNOSTIC_VARS['thermo_dynamic_all']
        selected = [i for i, name in enumerate(names)
                    if name.startswith(('thetao_', 'so_'))
                    and DEPTH_LEVELS[int(name.rsplit('_', 1)[1])] >= 2000]
        if selected != self.contract['deep_indices'] or [names[i] for i in selected] != self.contract['deep_names']:
            raise ValueError('Deep selection differs from the explicit named channel contract')
        if len(names) != 77 or len(selected) != 10:
            raise ValueError('Unexpected prognostic layout')
        self.mode = mode
        self.deep_indices = tuple(selected)
        self.deep_output_indices = tuple(selected + [i + 77 for i in selected])

    def specification(self):
        return {'mode': self.mode, 'deep_indices': list(self.deep_indices),
                'contract_sha256': self.contract_sha256}

    def prepare_input(self, true_input, previous_prediction=None):
        """Use current input's forcing; never consume an unsupervised prediction.

        Input layout is [two time-major 77-state slots, two 4-forcing slots].
        The caller supplies true_input for this exact model-call date. Replacing
        only deep slots preserves chronological nondeep history for arm A.
        """
        if true_input.ndim != 4 or true_input.shape[1] != 162:
            raise ValueError('Expected two-history 162-channel input')
        result = true_input.clone()
        if self.mode == 'deep_ocean_input':
            result[:, :154] = 0
            result[:, self.deep_output_indices] = true_input[:, self.deep_output_indices]
        if previous_prediction is not None:
            if previous_prediction.shape != true_input[:, :154].shape:
                raise ValueError('Previous prediction must contain two 77-channel output slots')
            result[:, self.deep_output_indices] = previous_prediction[:, self.deep_output_indices]
        return result
