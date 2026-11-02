"""Paper-based Neptune architecture (Donno et al., arXiv:2609.08606v1).

See configs/neptune_om4/README.md for specified details, assumptions and the
deliberate OM4 protocol adaptations. This is not the authors' implementation.
"""

import math

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ocean_emulators.models.base import BaseModel


class ZonalConv(nn.Module):
    """Periodic longitude and zero latitude padding, preserving grid size."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        p = self.padding
        if p:
            x = F.pad(x, (p, p, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, p, p))
        return self.conv(x)


class ConditionalLayerNorm(nn.Module):
    """Channel layer norm at each grid cell, with conditional scale and shift."""

    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.affine = nn.Linear(condition_dim, 2 * channels)

    def forward(self, x, condition):
        scale, shift = self.affine(condition).chunk(2, dim=-1)
        x = self.norm(x.movedim(1, -1)).movedim(-1, 1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, channels, condition_dim, dropout, mid_channels=None
    ):
        super().__init__()
        mid_channels = channels if mid_channels is None else mid_channels
        self.convs = nn.ModuleList(
            [ZonalConv(in_channels, mid_channels), ZonalConv(mid_channels, channels)]
        )
        self.norms = nn.ModuleList(
            [ConditionalLayerNorm(c, condition_dim) for c in (mid_channels, channels)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, condition):
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = self.dropout(F.silu(norm(conv(x), condition)))
        return x


def spherical_grid(lat: torch.Tensor, lon: torch.Tensor, grid: str) -> bool:
    """Validate transform coordinates and return whether latitude needs flipping."""
    latitude = lat.detach().cpu().double().numpy()
    longitude = lon.detach().cpu().double().numpy()
    if (
        latitude.ndim != 1
        or longitude.ndim != 1
        or min(len(latitude), len(longitude)) < 2
    ):
        raise ValueError(
            "Neptune requires 1D global latitude and longitude coordinates."
        )
    if grid == "legendre-gauss":
        expected = np.rad2deg(
            np.arcsin(np.polynomial.legendre.leggauss(len(latitude))[0])
        )
    elif grid == "equiangular":
        expected = np.linspace(-90, 90, len(latitude))
    else:
        raise ValueError(f"Unsupported spherical grid: {grid}")
    ascending = bool(latitude[-1] > latitude[0])
    if not np.allclose(
        latitude if ascending else latitude[::-1], expected, atol=2e-4, rtol=0
    ):
        raise ValueError(
            f"Latitude coordinates do not match the {grid} SHT grid; no implicit regridding is performed."
        )
    delta = np.diff(np.r_[longitude, longitude[0] + 360])
    if not np.allclose(delta, 360 / len(longitude), atol=2e-4, rtol=0):
        raise ValueError(
            "Longitude must be ascending, uniformly spaced and cover one period without duplicate endpoints."
        )
    return ascending


class SphericalConvolution(nn.Module):
    """Dense SHT convolution with degree-dependent, order-independent weights.

    modes=(L, M) means the actual retained coefficient array [0:L, 0:M],
    including degree/order zero. Parameters use real pairs for DDP compatibility.
    """

    def __init__(self, channels, nlat, nlon, modes, grid, flip_latitude):
        super().__init__()
        try:
            from torch_harmonics import (  # type: ignore[import-untyped,import-not-found]
                InverseRealSHT,
                RealSHT,
            )
        except ImportError as exc:
            raise ImportError(
                "Neptune requires the optional dependency: uv sync --extra neptune"
            ) from exc
        lmax, mmax = modes
        if not (1 <= mmax <= lmax <= nlat and mmax <= nlon // 2 + 1):
            raise ValueError(
                "SHT modes must satisfy 1 <= M <= L <= nlat and M <= nlon//2+1."
            )
        self.sht = RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
        self.isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid)
        self.flip_latitude = flip_latitude
        self.weight = nn.Parameter(
            torch.randn(channels, channels, lmax, 2) / math.sqrt(2 * channels)
        )
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        dtype = x.dtype
        # Complex transforms/contractions run in FP32 even when convolutions use AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            if self.flip_latitude:
                x = x.flip(-2)
            spectrum = self.sht(x)
            weight = torch.view_as_complex(self.weight.float().contiguous())
            spectrum = torch.einsum("bilm,iol->bolm", spectrum, weight)
            x = self.isht(spectrum)
            if self.flip_latitude:
                x = x.flip(-2)
            return (x + self.bias).to(dtype)


class SFNOBlock(nn.Module):
    """The two residual branches and CLN placements follow paper Figure 1."""

    def __init__(
        self, channels, condition_dim, nlat, nlon, modes, grid, flip_latitude, mlp_ratio
    ):
        super().__init__()
        self.spectral = SphericalConvolution(
            channels, nlat, nlon, modes, grid, flip_latitude
        )
        self.norm1 = ConditionalLayerNorm(channels, condition_dim)
        self.linear = nn.Conv2d(channels, channels, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(channels * mlp_ratio, channels, 1),
        )
        self.gate = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.norm2 = ConditionalLayerNorm(channels, condition_dim)

    def forward(self, x, condition):
        y = F.gelu(self.norm1(self.spectral(x), condition) + self.linear(x))
        return F.gelu(self.norm2(self.mlp(y) + self.gate * x, condition))


class FourierPosition(nn.Module):
    """Latitude, longitude, fractional year and astronomical day length."""

    latitude: torch.Tensor
    longitude: torch.Tensor
    spatial_frequencies: torch.Tensor
    temporal_frequencies: torch.Tensor
    day_length_frequencies: torch.Tensor

    def __init__(
        self, lat, lon, channels, spatial_bands, temporal_bands, day_length_bands
    ):
        super().__init__()
        if min(spatial_bands, temporal_bands, day_length_bands) < 1:
            raise ValueError("Fourier band counts must be positive.")
        self.register_buffer("latitude", torch.deg2rad(lat.float()), persistent=False)
        self.register_buffer("longitude", torch.deg2rad(lon.float()), persistent=False)
        for name, bands in (
            ("spatial_frequencies", spatial_bands),
            ("temporal_frequencies", temporal_bands),
            ("day_length_frequencies", day_length_bands),
        ):
            self.register_buffer(name, 2.0 ** torch.arange(bands), persistent=False)
        features = 4 * spatial_bands + 2 * temporal_bands + 2 * day_length_bands
        self.projection = nn.Conv2d(features, channels, 1)

    @staticmethod
    def _encode(field, frequencies):
        angles = field[:, None] * frequencies[None, :, None, None]
        return torch.cat((angles.sin(), angles.cos()), dim=1)

    def day_length(self, phase):
        # Geometric sunrise/sunset approximation; polar night/day clipped to 0/24 h.
        declination = math.radians(23.44) * torch.sin(
            2 * math.pi * (phase - 80 / 365.2425)
        )
        tan_latitude = self.latitude.sin() / self.latitude.cos().clamp_min(1e-7)
        cosine = -tan_latitude[None, :] * torch.tan(declination)[:, None]
        return 24 / math.pi * torch.acos(cosine.clamp(-1, 1))

    def forward(self, phase):
        batch, nlat, nlon = (
            phase.shape[0],
            self.latitude.numel(),
            self.longitude.numel(),
        )
        shape = (batch, nlat, nlon)
        features = torch.cat(
            [
                self._encode(
                    self.latitude[None, :, None].expand(shape), self.spatial_frequencies
                ),
                self._encode(
                    self.longitude[None, None, :].expand(shape),
                    self.spatial_frequencies,
                ),
                self._encode(
                    (2 * math.pi * phase)[:, None, None].expand(shape),
                    self.temporal_frequencies,
                ),
                # Day length is bounded, not periodic: 0 h and 24 h must differ.
                self._encode(
                    (math.pi * self.day_length(phase) / 24)[:, :, None].expand(shape),
                    self.day_length_frequencies,
                ),
            ],
            dim=1,
        )
        return self.projection(features)


class Neptune(BaseModel):
    requires_calendar = True
    time_frequencies: torch.Tensor
    fixed_noise: torch.Tensor

    def __init__(
        self,
        in_channels,
        out_channels,
        hist,
        wet,
        lat,
        lon,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        static_data=None,
        gradient_detach_interval=0,
        hidden_channels=256,
        decoder_mid_channels=None,
        modes=(2, 2),
        grid="legendre-gauss",
        condition_dim=256,
        noise_dim=32,
        spatial_fourier_bands=8,
        temporal_fourier_bands=4,
        day_length_fourier_bands=4,
        dropout=0.0,
        mlp_ratio=2,
        inference_noise="fixed",
        inference_noise_seed=15,
        checkpointing=None,
        use_bfloat16=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            wet,
            hist,
            pred_residuals,
            last_kernel_size,
            pad,
            static_data,
            gradient_detach_interval,
        )
        if hist != 1 or out_channels % 2 or in_channels < out_channels:
            raise ValueError(
                "Neptune expects two input states, two output states and optional forcing (hist=1)."
            )
        if pad != "circular" or static_data is not None and len(static_data.data_vars):
            raise ValueError(
                "Neptune supports circular zonal padding and no extra static variables."
            )
        if (
            last_kernel_size < 1
            or decoder_mid_channels is not None
            and decoder_mid_channels < 1
            or min(
                hidden_channels,
                condition_dim,
                noise_dim,
                spatial_fourier_bands,
                temporal_fourier_bands,
                day_length_fourier_bands,
                mlp_ratio,
            )
            < 1
        ):
            raise ValueError("Kernel sizes and model dimensions must be positive.")
        if inference_noise not in ("zero", "sample", "fixed") or checkpointing not in (
            None,
            "all",
        ):
            raise ValueError(
                "Use inference_noise='zero'/'sample'/'fixed' and checkpointing=None/'all'."
            )
        flip_latitude = spherical_grid(lat, lon, grid)
        if wet.shape[-2:] != (len(lat), len(lon)) or wet.shape[-3] != out_channels:
            raise ValueError(
                "Wet mask must have one mask per output channel on the model grid."
            )
        # BaseModel's legacy wet attribute is not a buffer; Neptune needs .to() support.
        del self.wet
        self.register_buffer("wet", wet.bool(), persistent=False)
        self.noise_dim = noise_dim
        self.inference_noise = inference_noise
        # One reproducible Gaussian member, independent of global RNG, batching
        # and DDP rank. Regenerated from config rather than loaded from weights,
        # so changing the evaluation seed selects another coherent member.
        generator = torch.Generator(device="cpu").manual_seed(inference_noise_seed)
        self.register_buffer(
            "fixed_noise", torch.randn(noise_dim, generator=generator), persistent=False
        )
        self.checkpointing = checkpointing
        self.use_bfloat16 = use_bfloat16
        self.register_buffer(
            "time_frequencies",
            2.0 ** torch.arange(temporal_fourier_bands),
            persistent=False,
        )
        self.condition = nn.Sequential(
            nn.Linear(2 * temporal_fourier_bands + noise_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.lift = ZonalConv(in_channels, hidden_channels)
        self.position = FourierPosition(
            lat,
            lon,
            hidden_channels,
            spatial_fourier_bands,
            temporal_fourier_bands,
            day_length_fourier_bands,
        )
        self.encoders = nn.ModuleList(
            [
                ConvBlock(hidden_channels, hidden_channels, condition_dim, dropout)
                for _ in range(2)
            ]
        )
        self.sfno = nn.ModuleList(
            [
                SFNOBlock(
                    hidden_channels,
                    condition_dim,
                    len(lat),
                    len(lon),
                    modes,
                    grid,
                    flip_latitude,
                    mlp_ratio,
                )
                for _ in range(2)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                ConvBlock(
                    2 * hidden_channels,
                    hidden_channels,
                    condition_dim,
                    dropout,
                    mid_channels=decoder_mid_channels,
                )
                for _ in range(2)
            ]
        )
        self.project = ZonalConv(hidden_channels, out_channels, last_kernel_size)

    def _block(self, block, x, condition):
        if self.checkpointing == "all" and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, condition, use_reentrant=False)
        return block(x, condition)

    def forward_step(self, fts, phase=None):
        return self.forward_once(fts, phase)

    def forward_once(self, fts, phase=None, noise=None):
        """Decode two states (or increments); explicit noise enables ensemble members.

        phase is the fractional year of the latest input state, one value per
        batch item. Training samples N(0,I) once per item and model call. All
        blocks in that call share the condition; checkpoint replay reuses it.
        """
        if phase is None:
            raise ValueError("Neptune requires the actual last-input calendar phase.")
        phase = torch.as_tensor(phase, device=fts.device, dtype=torch.float32)
        if phase.shape != (fts.shape[0],):
            raise ValueError("Calendar phase must have shape (batch,).")
        if fts.ndim != 4 or fts.shape[1:] != (self.in_channels, *self.wet.shape[-2:]):
            raise ValueError(
                "Input channels or grid do not match the Neptune configuration."
            )
        if noise is None:
            shape = (fts.shape[0], self.noise_dim)
            if self.training or self.inference_noise == "sample":
                # Trainer seeds every rank identically. Select disjoint draws
                # without a separate RNG, so the existing resume RNG stays valid.
                distributed = dist.is_available() and dist.is_initialized()
                ranks = dist.get_world_size() if distributed else 1
                rank = dist.get_rank() if distributed else 0
                noise = torch.randn((ranks, *shape), device=fts.device)[rank]
            elif self.inference_noise == "fixed":
                noise = self.fixed_noise.expand(shape)
            else:
                noise = torch.zeros(shape, device=fts.device)
        else:
            noise = noise.to(device=fts.device, dtype=torch.float32)
        if noise.shape != (fts.shape[0], self.noise_dim):
            raise ValueError("Noise must have shape (batch, noise_dim).")
        angles = 2 * math.pi * phase[:, None] * self.time_frequencies[None]
        with torch.autocast(
            device_type=fts.device.type, dtype=torch.bfloat16, enabled=self.use_bfloat16
        ):
            condition = self.condition(
                torch.cat((angles.sin(), angles.cos(), noise), dim=-1)
            )
            x = self.lift(fts) + self.position(phase)
            skips = []
            for block in self.encoders:
                x = self._block(block, x, condition)
                skips.append(x)
            for block in self.sfno:
                x = self._block(block, x, condition)
            for block, skip in zip(self.decoders, reversed(skips), strict=True):
                x = self._block(block, torch.cat((x, skip), dim=1), condition)
            x = self.project(x)
        return torch.where(self.wet, x.float(), 0)

    def decode_prediction(self, input_tensor, decodings):
        if not self.pred_residuals:
            return decodings
        # Paper residuals are consecutive increments, not residuals to paired histories.
        channels = self.out_channels // 2
        increments = decodings.unflatten(1, (2, channels))
        last_state = input_tensor[:, channels : 2 * channels].unsqueeze(1)
        states = (last_state + increments.cumsum(dim=1)).flatten(1, 2)
        return torch.where(self.wet, states, 0)
