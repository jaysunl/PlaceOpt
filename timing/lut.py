"""
timing.lut — Liberty NLDM timing-table wrapper.

``TensorTable`` wraps a single cell timing arc (one (in_pin, out_pin,
rise/fall) tuple) as a PyTorch ``nn.Module`` so that all table tensors can
be moved to CUDA with a single ``.to(device)`` call.

Bilinear interpolation is provided both as a module method and as a
standalone batched function used by the timing engine.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Batched bilinear interpolation (shared utility)
# ---------------------------------------------------------------------------

def _axis_interval(axis: torch.Tensor, x: torch.Tensor):
    """
    Find the interval index and fractional offset for *x* on *axis*.

    Parameters
    ----------
    axis : [M, K]  monotonically increasing axis values.
    x    : [M]

    Returns
    -------
    i0 : [M]  lower bracket index in [0, K-2].
    t  : [M]  fractional position in [0, 1].
    """
    eps = 1e-30
    # Number of axis entries *strictly less than* x (excluding first).
    i = torch.sum(x.unsqueeze(-1) >= axis[:, 1:], dim=-1)
    i0 = torch.clamp(i, 0, axis.shape[1] - 2)
    i1 = i0 + 1

    x0 = axis.gather(1, i0.unsqueeze(-1)).squeeze(-1)
    x1 = axis.gather(1, i1.unsqueeze(-1)).squeeze(-1)

    denom = x1 - x0
    t = torch.where(denom.abs() > eps, (x - x0) / (denom + eps), torch.zeros_like(x))
    return i0, t


def bilinear_interp_batch(
    table: torch.Tensor,    # [M, R, C]
    axis_row: torch.Tensor, # [M, R]
    axis_col: torch.Tensor, # [M, C]
    row_val: torch.Tensor,  # [M]
    col_val: torch.Tensor,  # [M]
) -> torch.Tensor:
    """
    Batched bilinear table lookup: one interpolation per sample.

    Supports arbitrary table sizes (not restricted to 7×7) by using
    ``gather`` on the full axis tensors.

    Returns
    -------
    values : [M]
    """
    M = row_val.shape[0]
    ar = torch.arange(M, device=row_val.device)

    i0, a = _axis_interval(axis_row, row_val)
    j0, b = _axis_interval(axis_col, col_val)
    i1 = i0 + 1
    j1 = j0 + 1

    v00 = table[ar, i0, j0]
    v01 = table[ar, i0, j1]
    v10 = table[ar, i1, j0]
    v11 = table[ar, i1, j1]

    return (1 - a) * (1 - b) * v00 + (1 - a) * b * v01 + a * (1 - b) * v10 + a * b * v11


# ---------------------------------------------------------------------------
# TensorTable
# ---------------------------------------------------------------------------

class TensorTable(nn.Module):
    """
    A single Liberty timing arc represented as GPU-ready PyTorch buffers.

    Stores the 2D delay and slew tables together with their axis vectors.
    All tensors are registered as buffers (not parameters) so they move
    with the module but do not receive gradients.

    Attributes
    ----------
    arc_description : human-readable arc string from Liberty
    in_pin_name     : Liberty input pin name for this arc
    out_pin_name    : Liberty output pin name for this arc
    in_rf           : input transition ("^" rise / "v" fall)
    out_rf          : output transition ("^" rise / "v" fall)
    axis_0          : input-slew axis  [K0]
    axis_1          : output-load axis [K1]
    delay_table     : delay values     [K0, K1]
    slew_table      : output-slew values [K0, K1]
    driver_rd       : effective driver output resistance (Ω), estimated from
                      the slope d(delay)/d(load) at mid-range.
    """

    def __init__(self, c_model) -> None:
        super().__init__()

        self.arc_description: str = c_model.arc_description
        self.in_pin_name: str = c_model.in_pin_name
        self.out_pin_name: str = c_model.out_pin_name
        self.in_rf: str = c_model.in_rf
        self.out_rf: str = c_model.out_rf

        axis0 = torch.tensor(list(c_model.table_axis0), dtype=torch.float32)
        axis1 = torch.tensor(list(c_model.table_axis1), dtype=torch.float32)
        delay = torch.tensor([list(row) for row in c_model.delay_table], dtype=torch.float32)
        slew  = torch.tensor([list(row) for row in c_model.slew_table],  dtype=torch.float32)

        self.register_buffer("axis_0",      axis0)
        self.register_buffer("axis_1",      axis1)
        self.register_buffer("delay_table", delay)
        self.register_buffer("slew_table",  slew)

        self.driver_rd: float = self._estimate_driver_rd()

    # ------------------------------------------------------------------

    def _estimate_driver_rd(self) -> float:
        """
        Estimate the effective driver output resistance (Ω):

            Rd = −ln(0.5) · |Δdelay / Δload|

        evaluated at the midpoint of both axes using a 1 fF capacitance
        perturbation.  This follows the standard Liberty Rd extraction
        formula used by OpenSTA.
        """
        ax0 = self.axis_0
        ax1 = self.axis_1
        n0, n1 = len(ax0), len(ax1)

        if n0 > 2:
            mid_slew = (ax0[n0 // 2 + 1] + ax0[n0 // 2]) / 2
        else:
            mid_slew = ax0[-1] * 0.75 + ax0[0] * 0.25

        if n1 > 2:
            mid_cap = (ax1[n1 // 2 + 1] + ax1[n1 // 2]) / 2
        else:
            mid_cap = ax1[-1] * 0.75 + ax1[0] * 0.25

        delta_c = 1e-15  # 1 fF perturbation
        d1 = self._lookup_delay(mid_slew, mid_cap)
        d2 = self._lookup_delay(mid_slew, mid_cap + delta_c)
        rd = float(-torch.log(torch.tensor(0.5)) * (d2 - d1).abs() / delta_c)
        return max(rd, 0.001)

    def _lookup_delay(self, slew, load) -> torch.Tensor:
        ax0 = self.axis_0.unsqueeze(0)    # [1, K0]
        ax1 = self.axis_1.unsqueeze(0)    # [1, K1]
        tbl = self.delay_table.unsqueeze(0)  # [1, K0, K1]
        return bilinear_interp_batch(tbl, ax0, ax1, slew.unsqueeze(0), load.unsqueeze(0))

    def lookup_delay(self, in_slew: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        """Batched delay lookup: in_slew, load → [M]."""
        return bilinear_interp_batch(
            self.delay_table.unsqueeze(0),
            self.axis_0.unsqueeze(0),
            self.axis_1.unsqueeze(0),
            in_slew.unsqueeze(0),
            load.unsqueeze(0),
        )

    def lookup_slew(self, in_slew: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        """Batched output-slew lookup: in_slew, load → [M]."""
        return bilinear_interp_batch(
            self.slew_table.unsqueeze(0),
            self.axis_0.unsqueeze(0),
            self.axis_1.unsqueeze(0),
            in_slew.unsqueeze(0),
            load.unsqueeze(0),
        )
