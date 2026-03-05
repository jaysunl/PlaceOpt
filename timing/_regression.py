"""
timing._regression — Polynomial regression models for buffer delay/slew.

Fits a 5-coefficient model of the form:

    f(s, c) = a₀ + a₁·(c/c_sc) + a₂·(s/s_sc) + a₃·cbrt(s/s_sc) + a₄·(s/s_sc)²

to a 7×7 Liberty timing table via ordinary least-squares.  The resulting
coefficients are used by ``schedule._make_buf_funcs`` to build differentiable
closure functions that can be evaluated at arbitrary (slew, load) values
during the timing engine forward pass.

This avoids bilinear-table interpolation for the buffer cell (which has only
a single master) and allows the buffer delay/slew to be expressed as a smooth
polynomial, making gradients through the buffer model numerically cleaner.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def fit_buf_model(
    table: torch.Tensor,   # [K0, K1]  delay or slew table
    axis_slew: torch.Tensor,  # [K0]
    axis_cap:  torch.Tensor,  # [K1]
) -> Optional[Dict]:
    """
    Fit the 5-coefficient buffer polynomial model to *table*.

    Parameters
    ----------
    table      : Liberty table values [K0, K1].
    axis_slew  : slew axis [K0] (seconds).
    axis_cap   : load axis [K1] (Farads).

    Returns
    -------
    dict with keys:
        "coeffs"  : list of 5 floats [a0..a4]
        "s_scale" : float, normalisation factor for slew axis
        "c_scale" : float, normalisation factor for cap  axis
    or None if the fit fails.
    """
    try:
        s_scale = float(axis_slew.abs().max().clamp_min(1e-20))
        c_scale = float(axis_cap.abs().max().clamp_min(1e-20))

        # Build the [K0·K1, 5] feature matrix.
        S = axis_slew.float()   # [K0]
        C = axis_cap.float()    # [K1]
        s_n    = S / s_scale    # normalised slew  [K0]
        c_n    = C / c_scale    # normalised cap   [K1]

        # Expand to [K0·K1] pairs.
        s_flat = s_n.unsqueeze(1).expand(-1, len(C)).reshape(-1)
        c_flat = c_n.unsqueeze(0).expand(len(S), -1).reshape(-1)
        y_flat = table.float().reshape(-1)

        # Feature columns: [1, c, s, cbrt(s), s²]
        s_cbrt = torch.sign(s_flat) * (s_flat.abs() + 1e-10).pow(1.0 / 3.0)
        X = torch.stack([
            torch.ones_like(s_flat),
            c_flat,
            s_flat,
            s_cbrt,
            s_flat ** 2,
        ], dim=1)  # [N, 5]

        # Least-squares via the normal equations: (XᵀX)⁻¹ Xᵀ y
        coeffs, *_ = torch.linalg.lstsq(X, y_flat.unsqueeze(1))
        coeffs = coeffs.squeeze(1).tolist()

        return {"coeffs": coeffs, "s_scale": s_scale, "c_scale": c_scale}
    except Exception as exc:
        # Regression failure is non-fatal; callers fall back to default coeffs.
        print(f"[WARN] Buffer regression failed: {exc}")
        return None


def delay_regression(table: torch.Tensor, axis_slew: torch.Tensor,
                     axis_cap: torch.Tensor) -> Optional[Dict]:
    """Alias of ``fit_buf_model`` for delay tables."""
    return fit_buf_model(table, axis_slew, axis_cap)


def slew_regression(table: torch.Tensor, axis_slew: torch.Tensor,
                    axis_cap: torch.Tensor) -> Optional[Dict]:
    """Alias of ``fit_buf_model`` for slew tables."""
    return fit_buf_model(table, axis_slew, axis_cap)
