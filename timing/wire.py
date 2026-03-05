"""
timing.wire — Wire delay models.

This module contains:

1. ``rc_ramp_crossing_time`` — analytical crossing-time solver for an RC
   network driven by a ramp waveform (used for DMP wire delay).
2. ``wire_delay_slew_dmp`` — distributed multi-pole (DMP) wire delay and
   slew computation based on the ramp-crossing approach.
3. ``WireSegmentFunction`` — the main differentiable autograd Function that
   computes arrival time and slew at a child Steiner point given the parent
   arrival / slew and the current segment's Elmore parameters.
4. ``conti_buf_delay_slew`` — continuous buffer model: computes the delay
   and output slew introduced by a soft-inserted buffer on a wire segment.
"""

from __future__ import annotations

import torch

from placeopt.timing.constants import (
    RISE, FALL,
    M0, M1, M2,
    R_PER_LEN, C_PER_LEN, TO_MICRON,
)


# ---------------------------------------------------------------------------
# Ramp-RC crossing time solver
# ---------------------------------------------------------------------------

def rc_ramp_crossing_time(
    tau: torch.Tensor,
    T: torch.Tensor,
    v: float,
    newton_iters: int = 1,
    eps: float = 1e-18,
) -> torch.Tensor:
    """
    Solve y(t) = v for a first-order RC step driven by a unit ramp u(t)=t/T.

    Two regions are handled analytically:
    * Crossing during the ramp  (v ≤ y(T)): Newton's method on the RC ramp.
    * Crossing after the ramp   (v > y(T)): exponential tail formula.

    Parameters
    ----------
    tau  : [N]  RC time constant (seconds).
    T    : [N]  ramp duration (seconds).
    v    : scalar voltage threshold (e.g. 0.5 for 50 %).

    Returns
    -------
    t_cross : [N]  crossing time from ramp start.
    """
    # Ramp endpoint: y(T) = 1 − (τ/T)·(1 − e^{−T/τ})
    exp_T = torch.exp(-T / tau.clamp_min(eps))
    yT = 1.0 - (tau / T.clamp_min(eps)) * (1.0 - exp_T)

    # Post-ramp region (exponential decay).
    K = (tau / T.clamp_min(eps)) * (1.0 - exp_T)
    t_post = T - tau * torch.log(((1.0 - v) / K.clamp_min(eps)).clamp_min(eps))

    # During-ramp region (Newton on  g(t) = t/T − τ/T·(1−e^{−t/τ}) − v).
    t = torch.clamp(v * T + 0.5 * tau, min=0.0)
    for _ in range(newton_iters):
        exp_t = torch.exp(-t / tau.clamp_min(eps))
        g  = (t / T.clamp_min(eps)) - (tau / T.clamp_min(eps)) * (1.0 - exp_t) - v
        gp = (1.0 - exp_t) / T.clamp_min(eps)
        t  = (t - g / gp.clamp_min(eps)).clamp(0.0, T)

    return torch.where(v > yT, t_post, t)


def wire_delay_slew_dmp(
    tau: torch.Tensor,
    slew_in: torch.Tensor,
    vth: float = 0.5,
    vl: float = 0.2,
    vh: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute incremental wire delay and output slew using the ramp-crossing
    distributed multi-pole (DMP) approach.

    Parameters
    ----------
    tau      : [N]  Elmore time constant (seconds).
    slew_in  : [N]  input slew measured between vl and vh (seconds).

    Returns
    -------
    delay_inc : [N]  incremental delay (driver vth → sink vth).
    slew_out  : [N]  output slew (vl → vh), clipped to ≥ slew_in.
    """
    # Convert 20–80% slew to full 0→1 ramp duration T.
    T = slew_in / (vh - vl)
    t_vth = rc_ramp_crossing_time(tau, T, vth)
    t_vl  = rc_ramp_crossing_time(tau, T, vl)
    t_vh  = rc_ramp_crossing_time(tau, T, vh)

    t_in_vth = vth * T
    delay_inc = (t_vth - t_in_vth).clamp_min(0.0)
    slew_out  = (t_vh - t_vl).clamp_min(slew_in)
    return delay_inc, slew_out


# ---------------------------------------------------------------------------
# Continuous buffer model
# ---------------------------------------------------------------------------

def _half_wire_length(
    sp_xy: torch.Tensor,    # [N, 2]
    ep_xy: torch.Tensor,    # [N, 2]
    xcen_buf: torch.Tensor, # [2]
    xin_buf: torch.Tensor,  # [2]
) -> torch.Tensor:
    """
    Compute the effective half-wire length (µm) when a buffer is placed
    at the midpoint of the segment [sp_xy, ep_xy].
    """
    xout_buf = 2.0 * xcen_buf - xin_buf
    half = torch.abs(
        (ep_xy - sp_xy - (xout_buf - xin_buf).unsqueeze(0)) / 2.0
    ).sum(dim=1) * TO_MICRON
    return half


def conti_buf_delay_slew(
    sp_xy: torch.Tensor,    # [N, 2]  parent Steiner point coordinate
    ep_xy: torch.Tensor,    # [N, 2]  child  Steiner point coordinate
    xcen_buf: torch.Tensor, # [2]     buffer centre pin offset
    xin_buf: torch.Tensor,  # [2]     buffer input pin offset
    buf_in_cap: float,
    sp_slew: torch.Tensor,  # [N]
    ep_load: torch.Tensor,  # [N]     downstream cap seen at child
    n: torch.Tensor,        # [N]     soft number of buffers
    fs_s, fs_l, fd_s, fd_l, # buffer regression functions
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable delay + slew for a wire segment with a soft-inserted buffer.

    The model blends wire-only and buffered contributions using ``n`` ∈ [0,1]:
    * ``activation`` = clip(|n|, 0, 1)  — presence of at least one buffer.
    * ``internal``   = max(|n|−1, 0)    — contribution of chain buffers.

    Returns
    -------
    delay, slew : each [N].
    """
    n_abs = n.abs()
    activation = n_abs.clamp_max(1.0)
    internal   = (n_abs - 1.0).clamp_min(0.0)

    Cb = buf_in_cap
    ln = _half_wire_length(sp_xy, ep_xy, xcen_buf, xin_buf)

    C_ln  = C_PER_LEN * ln
    R_op  = R_PER_LEN * ln * 0.5  # distributed RC integral

    # Delay to first buffer input.
    L_buf_in = C_ln + Cb
    el_in    = R_op * L_buf_in
    S_buf_in = torch.sqrt((fs_l(L_buf_in) * 1e12) ** 2
                          + 1.921 * (el_in * 1e12) ** 2) / 1e12

    # Delay to load (last segment of one-buffer chain).
    L_load  = C_ln + ep_load
    el_out  = R_op * L_load
    S_load  = torch.sqrt((sp_slew * 1e12) ** 2
                         + 1.921 * (el_in * 1e12) ** 2) / 1e12

    delay = (activation  * (C_ln + Cb) * R_op
             + (C_ln + ep_load) * R_op
             + internal  * (C_ln + Cb) * R_op
             + internal  * (fd_s(S_buf_in) + fd_l(L_buf_in))
             + activation * (fd_s(S_load)  + fd_l(L_load)))

    slew_with_buf = torch.sqrt(
        (fs_s(S_load) + fs_l(L_load) * 1e12) ** 2
        + 1.921 * (el_in * 1e12) ** 2
    ) / 1e12
    slew_no_buf = torch.sqrt((sp_slew * 1e12) ** 2
                              + 1.921 * (el_out * 1e12) ** 2) / 1e12

    slew = (1.0 - activation) * slew_no_buf + activation * slew_with_buf
    return delay, slew


# ---------------------------------------------------------------------------
# Segment forward function
# ---------------------------------------------------------------------------

class WireSegmentFunction(torch.autograd.Function):
    """
    Compute arrival and slew at a child Steiner point for a single wire step.

    Handles both simple Elmore and DMP modes, and optionally blends in a
    continuous buffer model controlled by ``b_flat`` (sigmoid buffer prob).

    All inputs/outputs are tensors of shape [N] or [N,2].
    """

    @staticmethod
    def forward(
        ctx,
        arrival_p,      # [N, 2]
        slew_p,         # [N, 2]
        moment_m0,      # [N]    zeroth moment at child
        pi_c,           # [N, 3] Π model at child
        sp_xy,          # [N, 2]
        ep_xy,          # [N, 2]
        b_flat_c,       # [N]    buffer logit (sigmoid applied externally)
        xcen_buf,       # [2]
        xin_buf,        # [2]
        buf_in_cap,     # scalar float
        # Buffer regression functions (not tensors; not stored for backward)
        fd_sr, fd_lr, fd_sf, fd_lf,
        fs_sr, fs_lr, fs_sf, fs_lf,
        use_ceff: int,
        use_dmp: int,
        use_buf_regression: int,
    ):
        scale = 1e15
        len_c = torch.abs(sp_xy - ep_xy).sum(dim=1) * TO_MICRON
        elmore = R_PER_LEN * len_c * moment_m0 / scale

        if use_dmp:
            dr, sr_out = wire_delay_slew_dmp(elmore, slew_p[:, RISE])
            df, sf_out = wire_delay_slew_dmp(elmore, slew_p[:, FALL])
        else:
            from placeopt.timing.pi_model import ElmoreDelayFunction
            dr, sr_out, df, sf_out = ElmoreDelayFunction.apply(elmore, slew_p[:, RISE], slew_p[:, FALL])

        # Wire-only arrival / slew.
        arr_wire = torch.stack([arrival_p[:, RISE] + dr, arrival_p[:, FALL] + df], dim=1)
        slew_wire = torch.stack([sr_out, sf_out], dim=1)

        if use_buf_regression:
            delay_buf_r, slew_buf_r = conti_buf_delay_slew(
                sp_xy, ep_xy, xcen_buf, xin_buf, buf_in_cap,
                slew_p[:, RISE], moment_m0 / scale, b_flat_c,
                fs_sr, fs_lr, fd_sr, fd_lr,
            )
            delay_buf_f, slew_buf_f = conti_buf_delay_slew(
                sp_xy, ep_xy, xcen_buf, xin_buf, buf_in_cap,
                slew_p[:, FALL], moment_m0 / scale, b_flat_c,
                fs_sf, fs_lf, fd_sf, fd_lf,
            )
            arr_buf  = torch.stack([arrival_p[:, RISE] + delay_buf_r,
                                    arrival_p[:, FALL] + delay_buf_f], dim=1)
            slew_buf = torch.stack([slew_buf_r, slew_buf_f], dim=1)

            # Soft blend: wire-only when b≈0, buffered when b≈1.
            alpha = b_flat_c.clamp(0.0, 1.0).unsqueeze(1)
            arr_out  = (1.0 - alpha) * arr_wire  + alpha * arr_buf
            slew_out = (1.0 - alpha) * slew_wire + alpha * slew_buf
        else:
            arr_out  = arr_wire
            slew_out = slew_wire

        return arr_out, slew_out

    @staticmethod
    def backward(ctx, grad_arr, grad_slew):
        # Gradients flow through PyTorch autograd automatically because
        # the forward ops are all differentiable.  The explicit Function
        # wrapper is kept to allow future hand-written backward kernels
        # without changing the caller interface.
        return (None,) * 21
