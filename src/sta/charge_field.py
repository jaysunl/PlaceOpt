"""
charge_field.py — Differentiable placement density and electrostatic force computation.

Provides:
  * ChargeKernel / CudaChargeKernel  — custom autograd Functions
  * ChargeFieldModule (nn.Module)    — wraps ChargeKernel
  * CosineBasis / SineBasis          — DCT/DST matrices
  * InvSineCosBasis / InvCosSineBasis — mixed inverse transforms
  * PoissonFieldSolver               — spectral Poisson solver for e-field
  * compute_charge_density()         — top-level density accumulation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.sta.params import DENSITY_BINS

try:
    from src.sta.density_ext import density_cuda_ext
    _CUDA_AVAIL = True
except Exception:
    density_cuda_ext = None
    _CUDA_AVAIL = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _broadcast_dims(ref_xy, dims, device, dtype):
    """Expand dims tensor to match number of objects in ref_xy."""
    dims = dims.to(device=device, dtype=dtype)
    if dims.dim() == 1:
        dims = dims.view(1, 2) if dims.numel() == 2 else dims.view(1, -1)
    if dims.dim() == 2 and dims.shape[0] == 1 and ref_xy.shape[0] > 1:
        dims = dims.expand(ref_xy.shape[0], dims.shape[1])
    return dims


def _bin_accumulate(field, gx0, gy0, gx1, gy1, rect_xy, rect_dims,
                    weight=None, chunk=256, device="cpu", dtype=torch.float32):
    """Accumulate rectangular object contributions into a density bin grid."""
    if rect_xy.numel() == 0:
        return
    rect_xy  = rect_xy.to(device=device, dtype=dtype)
    rect_dims = _broadcast_dims(rect_xy, rect_dims, device=device, dtype=dtype)
    w_tensor = torch.is_tensor(weight)
    if weight is None:
        weight = 1.0
    if w_tensor:
        weight = weight.to(device=device, dtype=dtype)
        if weight.ndim == 0:
            w_tensor = False; weight = weight.item()
    n = int(rect_xy.shape[0])
    for s in range(0, n, chunk):
        e  = min(s + chunk, n)
        bx0 = rect_xy[s:e, 0]; by0 = rect_xy[s:e, 1]
        bx1 = bx0 + rect_dims[s:e, 0]; by1 = by0 + rect_dims[s:e, 1]
        ox  = (torch.minimum(gx1.unsqueeze(0), bx1.unsqueeze(1)) -
               torch.maximum(gx0.unsqueeze(0), bx0.unsqueeze(1))).clamp_min(0)
        oy  = (torch.minimum(gy1.unsqueeze(0), by1.unsqueeze(1)) -
               torch.maximum(gy0.unsqueeze(0), by0.unsqueeze(1))).clamp_min(0)
        if w_tensor:
            w = weight[s:e].unsqueeze(1)
            field.add_((ox * w).transpose(0, 1) @ oy)
        else:
            field.add_(weight * (ox.transpose(0, 1) @ oy))


def _charge_autograd(boundary, xy, dims, weight, grid_size, chunk):
    """Differentiable density forward pass (used inside backward)."""
    xmin, ymin = boundary[0, 0], boundary[0, 1]
    xmax, ymax = boundary[1, 0], boundary[1, 1]
    field = torch.zeros((grid_size, grid_size), dtype=xy.dtype, device=xy.device)
    gx, gy = field.shape
    lx = (xmax - xmin) / gx; ly = (ymax - ymin) / gy
    xs  = torch.arange(gx, device=xy.device, dtype=xy.dtype)
    ys  = torch.arange(gy, device=xy.device, dtype=xy.dtype)
    gx0 = xmin + xs * lx; gy0 = ymin + ys * ly
    gx1 = gx0 + lx;       gy1 = gy0 + ly

    if xy.numel() == 0:
        return field.div_(lx * ly)

    w_tensor = torch.is_tensor(weight)
    if weight is None:
        weight = 1.0
    if w_tensor:
        weight = weight.to(device=xy.device, dtype=xy.dtype)
        if weight.ndim == 0:
            w_tensor = False; weight = weight.item()

    dims_bc = _broadcast_dims(xy, dims, device=xy.device, dtype=xy.dtype)
    n = int(xy.shape[0])
    for s in range(0, n, chunk):
        e  = min(s + chunk, n)
        bx0 = xy[s:e, 0]; by0 = xy[s:e, 1]
        bx1 = bx0 + dims_bc[s:e, 0]; by1 = by0 + dims_bc[s:e, 1]
        ox  = (torch.minimum(gx1.unsqueeze(0), bx1.unsqueeze(1)) -
               torch.maximum(gx0.unsqueeze(0), bx0.unsqueeze(1))).clamp_min(0)
        oy  = (torch.minimum(gy1.unsqueeze(0), by1.unsqueeze(1)) -
               torch.maximum(gy0.unsqueeze(0), by0.unsqueeze(1))).clamp_min(0)
        if w_tensor:
            w = weight[s:e].unsqueeze(1)
            field = field + (ox * w).transpose(0, 1) @ oy
        else:
            field = field + weight * (ox.transpose(0, 1) @ oy)
    return field.div_(lx * ly)


# ---------------------------------------------------------------------------
# Custom autograd kernels
# ---------------------------------------------------------------------------

class ChargeKernel(torch.autograd.Function):
    """Differentiable placement density kernel (CPU / non-CUDA path)."""

    @staticmethod
    def forward(ctx, boundary, xy, dims, weight, grid_size, chunk):
        ctx.set_materialize_grads(False)
        ctx.grid_size = int(grid_size)
        ctx.chunk     = int(chunk)
        ctx.w_tensor  = torch.is_tensor(weight)
        if ctx.w_tensor:
            ctx.save_for_backward(boundary, xy, dims, weight)
        else:
            ctx.save_for_backward(boundary, xy, dims)
            ctx.w_const = weight

        xmin, ymin = boundary[0, 0], boundary[0, 1]
        xmax, ymax = boundary[1, 0], boundary[1, 1]
        field = torch.zeros((grid_size, grid_size), dtype=xy.dtype, device=xy.device)
        gx, gy = field.shape
        lx = (xmax - xmin) / gx; ly = (ymax - ymin) / gy
        xs  = torch.arange(gx, device=xy.device, dtype=xy.dtype)
        ys  = torch.arange(gy, device=xy.device, dtype=xy.dtype)
        gx0 = xmin + xs * lx; gy0 = ymin + ys * ly
        gx1 = gx0 + lx;       gy1 = gy0 + ly
        _bin_accumulate(field, gx0, gy0, gx1, gy1, xy, dims,
                        weight=weight, chunk=chunk, device=xy.device, dtype=xy.dtype)
        field.div_(lx * ly)
        return field

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None or not any(ctx.needs_input_grad):
            return None, None, None, None, None, None
        if ctx.w_tensor:
            boundary, xy, dims, weight = ctx.saved_tensors
        else:
            boundary, xy, dims = ctx.saved_tensors
            weight = ctx.w_const
        with torch.enable_grad():
            needs = ctx.needs_input_grad
            boundary = boundary.detach().requires_grad_(needs[0])
            xy   = xy.detach().requires_grad_(needs[1])
            dims = dims.detach().requires_grad_(needs[2])
            if ctx.w_tensor:
                weight = weight.detach().requires_grad_(needs[3])
            fwd = _charge_autograd(boundary, xy, dims, weight, ctx.grid_size, ctx.chunk)
            grad_out = torch.zeros_like(fwd) if grad_out is None else grad_out
            inp  = [boundary, xy, dims] + ([weight] if ctx.w_tensor else [])
            need = [needs[0], needs[1], needs[2]] + ([needs[3]] if ctx.w_tensor else [])
            req  = [i for i, n in zip(inp, need) if n]
            gs   = torch.autograd.grad(fwd, req, grad_out,
                                       retain_graph=False, create_graph=False,
                                       allow_unused=True) if req else []
        grads = [None] * 4
        it = iter(gs)
        for idx, n in enumerate(need):
            if n:
                grads[idx] = next(it)
        return grads[0], grads[1], grads[2], grads[3] if ctx.w_tensor else None, None, None


class CudaChargeKernel(torch.autograd.Function):
    """CUDA-accelerated density kernel (requires density_cuda_ext)."""

    @staticmethod
    def forward(ctx, boundary, xy, dims, weight, grid_size):
        w_tensor = torch.is_tensor(weight)
        if w_tensor:
            wt = weight.to(device=xy.device, dtype=xy.dtype)
            ws = 1.0
        else:
            wt = xy.new_empty((0,))
            ws = 1.0 if weight is None else float(weight)
        out = density_cuda_ext.forward(
            boundary.contiguous(), xy.contiguous(), dims.contiguous(),
            wt.contiguous(), ws, bool(w_tensor), int(grid_size),
        )
        ctx.save_for_backward(boundary, xy, dims, wt)
        ctx.w_tensor = bool(w_tensor); ctx.ws = ws; ctx.grid_size = int(grid_size)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if grad_out is None:
            return None, None, None, None, None
        boundary, xy, dims, wt = ctx.saved_tensors
        gxy, gdims, gw = density_cuda_ext.backward(
            grad_out.contiguous(), boundary, xy, dims, wt,
            ctx.ws, ctx.w_tensor, ctx.grid_size,
        )
        return None, gxy, gdims, (gw if ctx.w_tensor else None), None


# ---------------------------------------------------------------------------
# Module wrapper
# ---------------------------------------------------------------------------

class ChargeFieldModule(nn.Module):
    """Wraps ChargeKernel as an nn.Module for placement density evaluation."""

    def __init__(self, grid_size, chunk=256):
        super().__init__()
        self.grid_size = int(grid_size)
        self.chunk     = int(chunk)

    def forward(self, boundary, xy, dims, charge_weight):
        return ChargeKernel.apply(boundary, xy, dims, charge_weight,
                                  self.grid_size, self.chunk)

    def accumulate(self, field, gx0, gy0, gx1, gy1, xy, dims, weight, chunk=256):
        _bin_accumulate(field, gx0, gy0, gx1, gy1, xy, dims,
                        weight=weight, chunk=chunk,
                        device=xy.device, dtype=xy.dtype)


def _field_contribute(field, boundary, xy, dims, grid_size, chunk, weight=None):
    """Add one layer's contribution to the density field, selecting CUDA if available."""
    if xy is None or xy.numel() == 0:
        return field
    if _CUDA_AVAIL and xy.is_cuda and xy.dtype == torch.float32:
        if torch.is_tensor(weight) and weight.dtype != xy.dtype:
            return field + ChargeKernel.apply(boundary, xy, dims, weight, grid_size, chunk)
        return field + CudaChargeKernel.apply(boundary, xy, dims, weight, grid_size)
    return field + ChargeKernel.apply(boundary, xy, dims, weight, grid_size, chunk)


# ---------------------------------------------------------------------------
# Top-level density computation
# ---------------------------------------------------------------------------

def compute_charge_density(grid_bins, db, b_flat, chunk=256):
    """Compute placement density field from PlaceDB and buffer weights.

    Parameters
    ----------
    grid_bins : int
        Number of bins per side.
    db : PlaceDB
        Placement database (holds geometry and blockage info).
    b_flat : Tensor
        Buffer probability weights, shape [N] (one per STP node).
    chunk : int
        Chunk size for CPU accumulation.

    Returns
    -------
    field : Tensor [grid_bins, grid_bins]
    """
    boundary = db.boundary.to(device=b_flat.device, dtype=b_flat.dtype)
    dev  = boundary.device
    dt   = boundary.dtype
    field = torch.zeros((grid_bins, grid_bins), dtype=dt, device=dev)
    field = _field_contribute(field, boundary, db.soft_blockage_xy, db.soft_blockage_wh, grid_bins, chunk)
    field = _field_contribute(field, boundary, db.hard_blockage_xy, db.hard_blockage_wh, grid_bins, chunk)
    if db.cell_box.numel() != 0:
        field = _field_contribute(field, boundary,
                                  db.cell_box[:, 0:2], db.cell_box[:, 2:4],
                                  grid_bins, chunk)
    if db.buffer_wh.numel() != 0:
        pos    = db.pos_xy
        par    = db.stp_parent_idx
        buf_xy = (pos[par] + pos) * 0.5 - db.xcen_buf
        field  = _field_contribute(field, boundary, buf_xy, db.buffer_wh,
                                   grid_bins, chunk, weight=b_flat)
    return field


# ---------------------------------------------------------------------------
# Spectral transforms
# ---------------------------------------------------------------------------

class CosineBasis(nn.Module):
    """2-D DCT-II basis (orthonormal, same convention as DREAMPlace BinGrid)."""

    def __init__(self, H, W):
        super().__init__()
        self.H = H; self.W = W
        self.register_buffer("Mh", self._dct_matrix(H))
        self.register_buffer("Mw", self._dct_matrix(W))

    @staticmethod
    def _dct_matrix(N):
        k = torch.arange(N).unsqueeze(1).float()
        n = torch.arange(N).unsqueeze(0).float()
        M = torch.cos((math.pi / N) * (n + 0.5) * k)
        M[0, :] *= 1.0 / math.sqrt(2)
        return M * math.sqrt(2.0 / N)

    def forward(self, x, inverse=False):
        if not inverse:
            return torch.matmul(torch.matmul(self.Mh, x), self.Mw.t())
        return torch.matmul(torch.matmul(self.Mh.t(), x), self.Mw)


class SineBasis(nn.Module):
    """1-D half-grid sine basis for mixed DSCT/DCST transforms."""

    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.register_buffer("Sm", self._sine_matrix(N))

    @staticmethod
    def _sine_matrix(N):
        k = torch.arange(N).unsqueeze(1).float()
        n = torch.arange(N).unsqueeze(0).float()
        M = torch.sin((math.pi / N) * (n + 0.5) * k)
        if N > 1:
            M[1:, :] *= math.sqrt(2.0 / N)
        M[0, :] = 0.0
        return M

    def inverse(self, coeff):
        return torch.matmul(self.Sm.t(), coeff)


class InvSineCosBasis(nn.Module):
    """Inverse 2-D S_x · C_y mixed transform (DSCT)."""

    def __init__(self, Nx, Ny, dct: CosineBasis, sine_x: SineBasis):
        super().__init__()
        self.dct = dct; self.sine_x = sine_x

    def forward(self, coeff):
        return torch.matmul(torch.matmul(self.sine_x.Sm.t(), coeff), self.dct.Mw)


class InvCosSineBasis(nn.Module):
    """Inverse 2-D C_x · S_y mixed transform (DCST)."""

    def __init__(self, Nx, Ny, dct: CosineBasis, sine_y: SineBasis):
        super().__init__()
        self.dct = dct; self.sine_y = sine_y

    def forward(self, coeff):
        return torch.matmul(torch.matmul(self.dct.Mh.t(), coeff), self.sine_y.Sm)


class PoissonFieldSolver(nn.Module):
    """Spectral electrostatic solver: density → electric field.

    Implements the BinGrid Poisson solve used in DREAMPlace-style
    global placement: phi_hat = rho_hat / (wx^2 + wy^2).
    """

    def __init__(self, Nx, Ny, size_x, size_y):
        super().__init__()
        self.Nx = int(Nx); self.Ny = int(Ny)
        self.size_x = size_x; self.size_y = size_y

        self.dct    = CosineBasis(self.Nx, self.Ny)
        self.sine_x = SineBasis(self.Nx)
        self.sine_y = SineBasis(self.Ny)
        self.idsct  = InvSineCosBasis(self.Nx, self.Ny, self.dct, self.sine_x)
        self.idcst  = InvCosSineBasis(self.Nx, self.Ny, self.dct, self.sine_y)

        wx   = torch.arange(self.Nx).float() * (math.pi / self.Nx)
        wy   = torch.arange(self.Ny).float() * (math.pi / self.Ny)
        denom = wx[:, None].square() + wy[None, :].square()
        denom[0, 0] = 1.0
        self.register_buffer("inv_lap", 1.0 / denom)
        self.register_buffer("wx_buf", wx.view(1, 1, self.Nx, 1))
        self.register_buffer("wy_buf", wy.view(1, 1, 1, self.Ny))

    def forward(self, density_map):
        """
        density_map: (B, 1, Nx, Ny)
        Returns: e_field_x, e_field_y, phi  — each (B, 1, Nx, Ny)
        """
        rho_hat = self.dct(density_map, inverse=False)
        phi_hat = rho_hat * self.inv_lap
        phi_hat[:, :, 0, 0] = 0.0
        ex_hat  = -(phi_hat * self.wx_buf)
        ey_hat  = -(phi_hat * self.wy_buf)
        ex      = self.idsct(ex_hat)
        ey      = self.idcst(ey_hat)
        phi     = self.dct(phi_hat, inverse=True)
        return ex, ey, phi
