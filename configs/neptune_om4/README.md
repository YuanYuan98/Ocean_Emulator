# Neptune architecture on the Samudra 2 OM4 protocol

This is an independent, paper-based implementation, not an official Neptune
release or a claim to reproduce the original ORAS5 scores. Source:
[Donno et al., Neptune, arXiv:2609.08606v1](https://arxiv.org/html/2609.08606v1),
Section 2.2.2 and Figure 1. The linked Appendix A contains additional evaluations,
not a complete architecture or hyperparameter specification.

## Architecture and implementation choices

The paper specifies a 3×3 input convolution with 256 latent channels; latitude,
longitude, day-of-year and day-length Fourier embeddings; two encoder blocks,
two SFNO blocks and two decoder blocks; concatenated encoder/decoder skips;
two Conv–conditional LayerNorm–SiLU–dropout sequences per convolution block;
and date plus Gaussian noise conditioning. Figure 1 determines the placement of
the SFNO linear/soft-gating skips, conditional norms and GeLU activations.
The paper reports 2×2 spectral modes and approximately 13 million parameters.

Undocumented choices are explicit configuration parameters or documented here:

Parameter precedence: use values explicitly specified by Neptune first. For
unspecified settings with a corresponding Samudra 2 parameter, follow the
matched Samudra 2 setup. Our expansion factor 2 follows its ConvNeXt expansion
ratio (not an assertion that their MLP architectures are identical); dropout 0
matches its absence of dropout; spatial kernel size 3 and zonal padding also
match the reference. Samudra 2's stage widths `[280, 380, 480, 520]` belong to
its four-stage, downsampling U-Net and have no unique mapping to Neptune's
encoder/SFNO/decoder blocks. The common input latent width **256 is explicit in
the paper**. Block outputs retain width 256; decoder intermediate convolutions
retain the 512 channels formed by concatenation before projecting back to 256.
This decoder width is our assumption, chosen to respect the rounded 13M budget.
Samudra 2 has no corresponding CLN condition/noise embeddings or Fourier band
counts: condition width 256, noise dimension 32, spatial bands 8, temporal bands
4 and day-length bands 4 remain implementation choices, rather than values
inherited from Samudra 2 or specified by Neptune.

| Detail | This implementation |
| --- | --- |
| Encoder/decoder widths and sampling | Encoder 256→256→256; decoder 512→512→256 after skip concatenation; full resolution throughout. The diagram does not specify pooling or intermediate widths. |
| Kernels and padding | 3×3 spatial convolutions; periodic longitude, zero latitude padding, stride 1. Output kernel configurable. |
| Conditional LayerNorm | Normalize channels independently at each grid cell, epsilon 1e-5. A learned projection supplies `(1 + scale)` and shift from the shared condition embedding. |
| Condition | 4 powers-of-two sine/cosine time bands + 32 independent standard normal variates; Linear–SiLU–Linear embedding of width 256. |
| Noise scope | One vector per batch item and model call, shared by all conditional norms in that call. Fresh noise at the next call in training. DDP ranks select disjoint draws even with identical seeds. Checkpoint recomputation reuses the same vector. |
| Pointwise SFNO MLP | Channel expansion factor 2, GeLU between two 1×1 convolutions. Soft gate is a learned per-channel multiplier initialized to one. |
| Dropout | 0.0 by default; configurable. No undocumented regularization added to the baseline. |
| Position encoding | Latitude/longitude: 8 bands each; fractional year: 4 bands; day length: 4 bands. 48 sine/cosine features projected to 256 and added after input lifting. Annual frequencies are 1, 2, 4, 8 cycles/year. |
| Day length | Geometric sunrise/sunset approximation with 23.44° obliquity, equinox phase 80/365.2425, clipped polar day/night. Encode the bounded angle `pi * hours / 24` so 0 and 24 hours remain distinguishable; no twilight/refraction correction. |
| Initialization | PyTorch defaults for convolutions/linear layers; spectral real/imaginary components have standard deviation `1/sqrt(2*C)`. |
| Precision | FP32 default, optional BF16 convolution autocast. SHT and spectral contraction always use FP32/complex64. |

Spherical transforms use `torch-harmonics==0.7.4`, pinned for the repository's
PyTorch 2.8 environment. They operate directly on the Gaussian OM4 grid, with
latitude reversed internally when input coordinates run south to north. The
constructor validates the grid; it will reject incompatible coordinates instead
of silently treating them as a regular grid. A pole-inclusive equiangular grid
is also supported by configuration. There is no planar FFT substitute.

Spectral weights are dense complex channel matrices per spherical degree,
shared across azimuthal orders, following the structure of
[NeuralOperator's SphericalConv](https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/spherical_convolution.py).
Real/imaginary parameter pairs permit ordinary DDP synchronization. `modes: [2, 2]`
explicitly means retained coefficient counts `lmax=2, mmax=2`, including zero.
Only the triangle `m <= l` contains valid coefficients. This convention is an
assumption: NeuralOperator releases have used a half-spectrum conversion of the
second `n_modes` entry, and the paper does not pin its library version. Set
`modes: [2, 1]` to test the alternative effective truncation. Do not equate the
two conventions without checking the original implementation.

The supplied OM4 model has **13,281,946 trainable parameters**, with 162 input
and 154 output channels. `decoder_mid_channels: 512` preserves concatenated
features through the first decoder convolution, then reduces them to 256 in
the second. Both conditional norms match their respective channel counts.
The input embedding, block count, spectral modes and SFNO widths are unchanged.
With 130 input and 120 output channels corresponding to the paper's two-time
60-ocean/5-forcing setup, this same assumed architecture has 13,129,848
parameters. The OM4 I/O adaptation adds 152,098 parameters.

These counts match the paper's rounded 13M scale but do not establish the
authors' unknown intermediate widths. They replace the initial 9,485,466
parameter candidate, which immediately compressed concatenated decoder features
to 256. Its training job and renewal controller were stopped and its campaign
deleted at the user's request. The revised 13M model was subsequently launched
after explicit authorization, as recorded below. Omitting `decoder_mid_channels` preserves the earlier narrow-block
behavior for configurations that explicitly need it.

The architecture/calendar suite checks the exact OM4 parameter budget,
four-call backward propagation, checkpoint recomputation, two-rank DDP,
independent spatial/seasonal encoding, polar day-length separation and
reproducible fixed-noise inference across batch partitions. A separate
full-resolution trainer preflight verifies memory and optimizer/loss integration;
neither check establishes forecast skill. The deleted narrow model's earlier
memory measurements do not apply to the wider model.

For this 8/4/4-band, fixed-member configuration, 24 architecture/calendar tests
pass. Full-resolution validation on two A100 80GB GPUs passes at local batch
16, accumulation 1 (global batch 32): four autoregressive calls, two optimizer
updates, dynamic-loss updates and validation. The check is an isolated numerical
preflight, not the start of a 70-epoch run. Its local audit record, including
source hashes and measured memory, is `.LOCAL/neptune_validation_20261014/verification.json`.

## Matched OM4 experiment

### Hyperparameter assessment against related models

The current width/MLP/dropout choices are defensible starting points, not
validated optima. Relevant primary-source comparisons:

- [torch-harmonics v0.7.4 SFNO implementation](https://github.com/NVIDIA/torch-harmonics/blob/v0.7.4/torch_harmonics/examples/models/sfno.py)
  defaults to embedding width 256, MLP ratio 2 and dropout 0. This is an
  operator implementation reference, not an ocean-trained Neptune configuration.
- [MedFormer, architecture section](https://link.springer.com/article/10.1007/s12145-026-02169-9)
  uses patch embedding 384, doubles channels after downsampling, and has about
  400M parameters. Its Swin U-Net supports seasonal conditioning but its widths
  cannot be transferred directly to our full-resolution convolution/SFNO blocks.
- [OceanNet, Section 2.3.1](https://os.copernicus.org/articles/21/1065/2025/)
  reports four Fourier layers, 64 retained modes and about 80M parameters for
  regional SSH. This does not establish that Neptune's explicitly documented
  2-by-2 spherical truncation is wrong: its convolution/skip paths retain local
  information, and spherical and planar mode conventions differ.
- [GLONET, Sections 2–3](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000686)
  combines global FNO and local CNN representations, activation checkpointing,
  and backpropagation through a four-day rollout. These support the broad design
  choices, but do not determine our CLN/noise dimensions.
- [Sun et al., ocean FNO hyperparameter study](https://arxiv.org/html/2404.05768v1)
  searches 2–64 latent channels, 2–16 FNO blocks and 2–32 modes on a much smaller
  idealized surface-ocean problem. These are search ranges, not recommended
  defaults for global OM4; validation rollout skill is needed to select settings.

Keep hidden width 256, condition width 256, MLP ratio 2 and dropout 0 for the
current baseline. The condition width is an assumption; none of these sources
establishes that 256 is optimal for Neptune CLN. The decoder intermediate width
512 is a capacity-matching assumption, not a newly discovered paper setting.

**Spatial and temporal Fourier bands are now separate.** The former eight
dyadic annual bands contained 64 and 128 cycles/year, with periods about 5.7
and 2.9 days. OM4 samples are five days apart; successive two-output model
calls advance ten days. The new annual encoding uses four bands (1, 2, 4, 8
cycles/year), preserving eight spatial bands. This is a conservative seasonal
prior, not demonstrated superiority over the earlier encoding. Day length has
its own four-band setting because its spatial and seasonal dependence is
nonlinear. Its angle spans half a circle: encoding `2*pi*hours/24` would alias
polar night and polar day. The old shared `fourier_bands` field is removed;
the new feature/condition projections require fresh training.

Noise dimension 32 remains weakly constrained. Increasing it to 128/256 has no
identified supporting evidence. Gaussian training noise with pointwise MSE
does not itself ensure calibrated ensemble spread, and evaluating at zero noise
is not equivalent to averaging stochastic predictions. The default inference
now selects a reproducible Gaussian member instead of the all-zero vector.
This defines a coherent member, not a proven accuracy improvement. Compare
fixed members, zero-noise and member averages on identical validation initial
conditions after training, including spread and rollout drift. Initial-weight
smoke checks cannot establish this comparison. Do not select settings on the
test set.

The matched learning rate, loss, global batch and 70-epoch final-weight protocol
remain fixed for the current architecture comparison. This answers performance
under Samudra's protocol, not the best attainable performance of a separately
tuned Neptune model. Neither published hyperparameters nor successful numerical
smoke tests establish convergence or long-rollout stability.

The supplied train/eval configurations retain the existing Samudra 2 protocol:

- Original OM4 Gaussian 180×360 grid, five-day means and actual Julian timestamps.
- 77 prognostic variables/depth channels and the same four forcing channels.
  Two input states give 162 input channels; two output states give 154 channels.
- Same normalization, wet masks, data splits and prescribed forcing.
- Four autoregressive training calls, with gradients through the entire rollout.
- Full-state targets, dynamic MSE with weight-ratio limit 20; Adam at 6e-4 and
  the existing cosine schedule; seed 15; 70 epochs.
- Global batch 32: the supplied config uses local batch 16 on two GPUs.
  On four GPUs, override the local batch to 8.
- Final ordinary model weights, retaining the reference trainer's legacy
  `ema_ckpt.pt` export behavior. That filename does not establish that these are
  EMA parameters. Resume training from the complete `ckpt.pt` checkpoint.

These are deliberate adaptations, rather than the paper's residual/MAE training
on daily ORAS5/ERA5 data. The implementation additionally supports
`pred_residuals: true`: both outputs are consecutive increments accumulated from
the latest input state. The matched OM4 configs keep it false.

Dates come from the **latest input timestamp** for each model call, including
batch shuffling, training stride and inference chunk offsets. No future label
date or state is used as an input. Non-date synthetic datasets remain usable by
legacy models, but Neptune requires calendar metadata.

Evaluation defaults to `inference_noise: fixed`, `inference_noise_seed: 15`.
A local CPU generator draws one N(0,I) vector, reused across calls and initial
conditions. All batch partitions and ranks therefore select the same member.
This draw consumes no global RNG state and does not change training's fresh,
per-item noise. The vector is regenerated from configuration, so another seed
selects another member without changing model weights. Save the resolved config
alongside results. This is **one member, not an ensemble mean**.

`inference_noise: zero` remains available for a zero-noise comparison;
`sample` draws new noise on every call, also in validation. Explicit
`forward_once(inputs, phase, noise=...)` overrides any default for controlled
member experiments. The generic evaluator does not aggregate ensembles or
compute CRPS.

## Running

The formal 13M OM4 campaign was launched on 2026-10-14 UTC in
`/pscratch/sd/y/yy6080/workspace/Ocean_Emulator_Workspace/experiments/neptune_13m_om4_20261014`.
Its initial Slurm job is `59761779`; a persistent controller renews four-hour
shared-interactive allocations for two A100 80GB GPUs. The campaign contains
frozen source, runtime scripts, resolved train/eval configs and a SHA256 manifest.
It starts from fresh weights with batch 16 per GPU and finishes at epoch 70,
then evaluates the final ordinary-weight export. Initial training batches passed
with finite losses; completion is tracked in the campaign's controller/log files.

Install in a separate environment if other jobs are using the current one:

```bash
UV_PROJECT_ENVIRONMENT=.venv-neptune uv sync --dev --extra neptune
source .venv-neptune/bin/activate
```

Inside a two-GPU allocation, using the existing trainer CLI:

```bash
torchrun --standalone --nproc_per_node=2 -m ocean_emulators.train \
  configs/neptune_om4/train.yaml \
  --experiment.data_root=/path/to/onedeg
```

The YAML files use `.LOCAL/train` and `.LOCAL/inference` for outputs. Update
`ckpt_path` when changing the training output directory. `model.yaml` is a
standalone model fragment; the train/eval files contain resolved copies so they
also work with schema validation. Keep the three model definitions identical.

```bash
python -m ocean_emulators.eval configs/neptune_om4/eval.yaml \
  --experiment.data_root=/path/to/onedeg
python -m pytest tests/test_neptune.py tests/test_calendar_conditioning.py -q
```

To check a real OM4 example at full resolution without training, use a resolved
train config containing its `experiment.data_root`:

```bash
python scripts/verify_neptune.py /path/to/resolved_train.yaml \
  --backward --output=/path/to/verification.json
```

This defaults to CPU; `--device=cuda` is available inside a GPU allocation.
The smoke loss measures only numerical validity at random initialization.
It is not a forecast skill score or a training-throughput benchmark.

`scripts/neptune_interactive.sh CAMPAIGN PYTHON` requests two 80-GB GPUs in
four-hour `shared_interactive` allocations. The campaign needs resolved configs,
a frozen `source/ocean_emulators` tree, `deps` containing torch-harmonics, and
the runner scripts in `runtime`. It first exercises two real distributed
training batches, dynamic loss updates and validation in a separate output
directory. Only a CUDA OOM triggers a pre-training fallback to local batch 8
and accumulation 2; other failures stop the controller.

The resumable runner in `scripts/revision_loss_segment.py` preserves the
70-epoch scheduler and per-rank RNG state between allocations, stopping at
complete epoch boundaries. The controller automatically requests the next
allocation and runs final-checkpoint evaluation after training completes.
Use a fresh Neptune source snapshot; the already-running loss revision
snapshots predate this architecture.
