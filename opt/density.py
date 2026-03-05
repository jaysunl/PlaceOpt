"""
opt.density — Differentiable placement density map.

Implements a RUDY-style (rectangle uniform density) placement density
computation as a custom ``torch.autograd.Function``.  The density map is
used as a penalty in the optimization loss to discourage cell overlap.

The forward pass accumulates each cell's area into the grid bins it
overlaps (weighted overlap area / bin area).  Soft buffers contribute
proportionally to their insertion probability ``b``.

Backward pass
-------------
Gradients are computed via an inner autograd graph (``torch.enable_grad``
in the backward method) rather than hand-written formulae, which avoids
maintaining a separate analytical backward while still being efficient.

CUDA extension
--------------
If the optional ``density_cuda_ext`` native extension is present it will be
used automatically for the forward and backward passes on CUDA tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Optional

from placeopt.timing.constants import GRID_SIZE

# Try to load the optional CUDA extension for faster density computation.
try:
    from placeopt.opt import density_cuda_ext as _cuda_ext  # type: ignore
    _HAS_CUDA_EXT = True
except Exception:
    _cuda_ext = None
    _HAS_CUDA_EXT = False


# ---------------------------------------------------------------------------
# Overlap accumulation (Python fallback)
# ---------------------------------------------------------------------------

def _accumulate_density(
    density: torch.Tensor,            # [Gx, Gy]  mutable output
    grid_x0: torch.Tensor, grid_y0: torch.Tensor,
    grid_x1: torch.Tensor, grid_y1: torch.Tensor,
    rect_xy: torch.Tensor,            # [N, 2]  lower-left corners
    rect_wh: torch.Tensor,            # [N, 2]  widths/heights
    weight: Optional[torch.Tensor],   # [N] or scalar
    chunk: int = 256,
) -> None:
    """
    Add weighted cell contributions to the density grid in-place.

    Each cell [x0, y0, x0+w, y0+h] contributes its overlap area with every
    grid bin, optionally scaled by ``weight``.
    """
    if rect_xy.numel() == 0:
        return

    w_is_t = torch.is_tensor(weight)
    if weight is None:
        weight = 1.0

    for start in range(0, rect_xy.shape[0], chunk):
        end  = min(start + chunk, rect_xy.shape[0])
        bx0  = rect_xy[start:end, 0]
        by0  = rect_xy[start:end, 1]
        bx1  = bx0 + rect_wh[start:end, 0]
        by1  = by0 + rect_wh[start:end, 1]

        # Overlap in x: [Gx, chunk]
        x_ov = (torch.minimum(grid_x1.unsqueeze(0), bx1.unsqueeze(1))
                - torch.maximum(grid_x0.unsqueeze(0), bx0.unsqueeze(1))).clamp_min(0)
        # Overlap in y: [chunk, Gy]
        y_ov = (torch.minimum(grid_y1.unsqueeze(0), by1.unsqueeze(1))
                - torch.maximum(grid_y0.unsqueeze(0), by0.unsqueeze(1))).clamp_min(0)

        if w_is_t:
            w = weight[start:end].unsqueeze(1)   # [chunk, 1]
            density.add_((x_ov * w).t() @ y_ov)
        else:
            density.add_(weight * (x_ov.t() @ y_ov))


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

class DensityFunction(torch.autograd.Function):
    """
    Differentiable density map accumulation.

    Forward  : sum of (weighted overlap area / bin area) for all cells.
    Backward : computed by replaying through the autograd-enabled inner forward.
    """

    @staticmethod
    def forward(ctx, boundary, xy, wh, weight, grid_size: int, chunk: int):
        ctx.grid_size = grid_size
        ctx.chunk     = chunk
        ctx.w_is_t    = torch.is_tensor(weight)
        if ctx.w_is_t:
            ctx.save_for_backward(boundary, xy, wh, weight)
        else:
            ctx.save_for_backward(boundary, xy, wh)
            ctx.weight_scalar = 1.0 if weight is None else float(weight)

        xmin, ymin = boundary[0, 0], boundary[0, 1]
        xmax, ymax = boundary[1, 0], boundary[1, 1]
        gx = gy = grid_size
        lx = (xmax - xmin) / gx
        ly = (ymax - ymin) / gy

        gx0 = xmin + torch.arange(gx, device=xy.device, dtype=xy.dtype) * lx
        gy0 = ymin + torch.arange(gy, device=xy.device, dtype=xy.dtype) * ly

        density = torch.zeros((gx, gy), dtype=xy.dtype, device=xy.device)
        _accumulate_density(density, gx0, gy0, gx0 + lx, gy0 + ly,
                            xy, wh, weight, chunk)
        density.div_(lx * ly)
        return density

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return (None,) * 6

        if ctx.w_is_t:
            boundary, xy, wh, weight = ctx.saved_tensors
        else:
            boundary, xy, wh = ctx.saved_tensors
            weight = ctx.weight_scalar

        needs = ctx.needs_input_grad

        with torch.enable_grad():
            bnd = boundary.detach().requires_grad_(needs[0])
            x   = xy.detach().requires_grad_(needs[1])
            w   = wh.detach().requires_grad_(needs[2])
            wt  = weight.detach().requires_grad_(needs[3]) if ctx.w_is_t else weight

            density = _density_autograd(bnd, x, w, wt, ctx.grid_size, ctx.chunk)
            inputs  = [bnd, x, w] + ([wt] if ctx.w_is_t else [])
            needed  = [needs[0], needs[1], needs[2]] + ([needs[3]] if ctx.w_is_t else [])
            req_inp = [inp for inp, need in zip(inputs, needed) if need]
            if req_inp:
                grads = torch.autograd.grad(density, req_inp, grad_out,
                                            create_graph=False, allow_unused=True)
            else:
                grads = []

        it = iter(grads)
        out = [None] * 6
        for i, need in enumerate([needs[0], needs[1], needs[2],
                                   needs[3] if ctx.w_is_t else False]):
            if need:
                out[i] = next(it, None)
        return tuple(out)


def _density_autograd(boundary, xy, wh, weight, grid_size, chunk):
    """Re-run the forward pass under autograd for gradient computation."""
    xmin, ymin = boundary[0, 0], boundary[0, 1]
    xmax, ymax = boundary[1, 0], boundary[1, 1]
    gx = gy = grid_size
    lx = (xmax - xmin) / gx
    ly = (ymax - ymin) / gy
    gx0 = xmin + torch.arange(gx, device=xy.device, dtype=xy.dtype) * lx
    gy0 = ymin + torch.arange(gy, device=xy.device, dtype=xy.dtype) * ly
    density = torch.zeros((gx, gy), dtype=xy.dtype, device=xy.device)
    _accumulate_density(density, gx0, gy0, gx0 + lx, gy0 + ly, xy, wh, weight, chunk)
    density = density / (lx * ly)
    return density


def _add_to_density(
    density: torch.Tensor,
    boundary: torch.Tensor,
    xy: torch.Tensor,
    wh: torch.Tensor,
    grid_size: int,
    chunk: int,
    weight=None,
) -> torch.Tensor:
    if xy is None or xy.numel() == 0:
        return density
    if (_HAS_CUDA_EXT and xy.is_cuda and xy.dtype == torch.float32
            and not (torch.is_tensor(weight) and weight.dtype != xy.dtype)):
        return density + _cuda_ext.forward(boundary, xy, wh, weight, grid_size)
    return density + DensityFunction.apply(boundary, xy, wh, weight, grid_size, chunk)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_density(
    grid_size: int,
    graph_data,
    b_flat: torch.Tensor,
    chunk: int = 256,
) -> torch.Tensor:
    """
    Compute the full placement density map [grid_size, grid_size].

    Contributions:
    * Hard blockages and macro cells (fixed, weight=1).
    * Soft blockages (fixed, weight=1).
    * Standard cells (via ``cell_box``, weight=1).
    * Soft-inserted buffers (weighted by ``b_flat`` ∈ [0,1]).

    Parameters
    ----------
    grid_size  : resolution of the density grid (e.g. 64).
    graph_data : STAGraph (provides cell_box, buffer_wh, pos_xy, etc.)
    b_flat     : [N]  buffer insertion probabilities (sigmoid output).

    Returns
    -------
    density : [grid_size, grid_size]  fractional area occupancy per bin.
    """
    gd = graph_data
    bnd = gd.boundary.to(device=b_flat.device, dtype=b_flat.dtype)
    density = torch.zeros((grid_size, grid_size), dtype=b_flat.dtype, device=b_flat.device)

    density = _add_to_density(density, bnd, gd.soft_blockage_xy, gd.soft_blockage_wh,
                               grid_size, chunk)
    density = _add_to_density(density, bnd, gd.hard_blockage_xy, gd.hard_blockage_wh,
                               grid_size, chunk)
    if gd.cell_box.numel():
        density = _add_to_density(density, bnd,
                                  gd.cell_box[:, :2], gd.cell_box[:, 2:],
                                  grid_size, chunk)
    if gd.buffer_wh.numel():
        pos = gd.pos_xy
        par = gd.stp_parent_idx
        buf_xy = (pos[par] + pos) * 0.5 - gd.xcen_buf
        density = _add_to_density(density, bnd, buf_xy, gd.buffer_wh,
                                  grid_size, chunk, weight=b_flat)
    return density


# ---------------------------------------------------------------------------
# Electrostatic solver (for ePlace-style density gradient)
# ---------------------------------------------------------------------------

class DCT2D(nn.Module):
    """2-D Type-II DCT via matrix multiply (for small grids ≤ 128)."""

    def __init__(self, H: int, W: int) -> None:
        super().__init__()
        self.H, self.W = H, W
        self.register_buffer("M_H", self._dct_matrix(H))
        self.register_buffer("M_W", self._dct_matrix(W))

    @staticmethod
    def _dct_matrix(N: int) -> torch.Tensor:
        k = torch.arange(N).unsqueeze(1).float()
        n = torch.arange(N).unsqueeze(0).float()
        mat = torch.cos(math.pi / N * (n + 0.5) * k)
        mat[0] *= 1.0 / math.sqrt(2)
        mat *= math.sqrt(2.0 / N)
        return mat

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if not inverse:
            return self.M_H @ x @ self.M_W.t()
        return self.M_H.t() @ x @ self.M_W
