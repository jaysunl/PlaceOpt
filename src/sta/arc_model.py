"""
arc_model.py — Differentiable timing arc evaluation for gradient-based placement.

Covers:
  * Pi-network moment-to-load transformation (PiNetworkGrad)
  * Effective capacitance computation — scalar (effective_load_cap) and
    batched GPU variant (batch_effective_load)
  * Gate arc LUT bilinear interpolation (lut_bilinear_batch, eval_arc_timing)
  * Wire RC delay propagation (ramp_threshold_time, rc_propagate)
  * Continuous buffer timing model (half_span_wire, distributed_buf_timing)
  * LUT regression fitting (fit_delay_model, fit_slew_model, arc_model_predict)
"""

import math
import torch
import torch.nn as nn
import numpy as np
from src.sta.params import (
    RF_RISE, RF_FALL, MOM0, MOM1, MOM2, PI_C1, PI_C2, PI_RPI,
    WIRE_R, WIRE_C, DBU_NM,
)


# ---------------------------------------------------------------------------
# Pi-network gradient autograd function
# ---------------------------------------------------------------------------

class PiNetworkGrad(torch.autograd.Function):
    """Converts wire Elmore moments (M0, M1, M2) to pi-model parameters (C1, C2, Rpi).

    Input  moment_3: [N, 3]  — M0=total-cap, M1=first-moment, M2=second-moment
    Output c1, c2, rpi: each [N]
    """
    R_VIA = 15.85  # via resistance added to each moment before pi extraction

    @staticmethod
    def forward(ctx, moment_3):
        if isinstance(moment_3, torch.Tensor) and moment_3.dim() == 1:
            moment_3 = moment_3.view(1, -1)

        rv = PiNetworkGrad.R_VIA
        m0 = moment_3[:, MOM0] + 1e-5
        m1 = moment_3[:, MOM1] - rv * (moment_3[:, MOM0] ** 2) - 1e-10
        m2 = (moment_3[:, MOM2]
              - 2.0 * rv * (moment_3[:, MOM0] * moment_3[:, MOM1])
              + (rv * rv) * (moment_3[:, MOM0] ** 3)
              + 1e-15)

        c1  = (m1 * m1) / m2
        c2  = m0 - c1
        rpi = -(m2 / m1) ** 2 / m1
        ctx.save_for_backward(moment_3[:, MOM0], moment_3[:, MOM1], m1, m2)
        return c1, c2, rpi

    @staticmethod
    def backward(ctx, dL_c1, dL_c2, dL_rpi):
        m0, m1_orig, y1, y2 = ctx.saved_tensors
        rv = PiNetworkGrad.R_VIA

        g_y0 = dL_c2

        tc1_y1 = 2 * y1 / y2
        trpi_y1 = 3 * (y2 ** 2) / (y1 ** 4)
        g_y1 = dL_c1 * tc1_y1 - dL_c2 * tc1_y1 + dL_rpi * trpi_y1

        tc1_y2  = -(y1 ** 2) / (y2 ** 2)
        trpi_y2 = -2 * y2 / (y1 ** 3)
        g_y2 = dL_c1 * tc1_y2 - dL_c2 * tc1_y2 + dL_rpi * trpi_y2

        g_m2 = g_y2
        g_m1 = g_y1 + g_y2 * (-2.0 * rv * m0)
        g_m0 = (g_y0
                + g_y1 * (-2.0 * rv * m0)
                + g_y2 * (-2.0 * rv * m1_orig + 3.0 * (rv ** 2) * (m0 ** 2)))

        return torch.stack([g_m0, g_m1, g_m2], dim=1)


# ---------------------------------------------------------------------------
# Scalar Ceff solver (CPU, matches OpenSTA DMP algorithm)
# ---------------------------------------------------------------------------

class SolverDivergence(RuntimeError):
    pass


def _lookup_2d(ax0, ax1, tbl, slew, cap):
    """Scalar bilinear LUT lookup for a single (slew, cap) query."""
    Ns, Nc = len(ax0), len(ax1)
    i = 0
    while i + 1 < Ns and ax0[i + 1] <= slew:
        i += 1
    j = 0
    while j + 1 < Nc and ax1[j + 1] <= cap:
        j += 1
    i = max(0, min(i, Ns - 2))
    j = max(0, min(j, Nc - 2))
    eps = 1e-20
    ds = (slew - ax0[i]) / (ax0[i + 1] - ax0[i] + eps)
    dc = (cap  - ax1[j]) / (ax1[j + 1] - ax1[j] + eps)
    v00, v01 = tbl[i][j], tbl[i][j + 1]
    v10, v11 = tbl[i + 1][j], tbl[i + 1][j + 1]
    return (1 - ds) * (1 - dc) * v00 + (1 - ds) * dc * v01 + ds * (1 - dc) * v10 + ds * dc * v11


def _lu_factor(a: np.ndarray, n: int, piv: np.ndarray, sc: np.ndarray,
               tiny: float = 1e-30) -> None:
    """In-place Crout LU with partial pivoting (Numerical-Recipes style)."""
    for i in range(n):
        big = np.max(np.abs(a[i, :]))
        if big == 0.0:
            raise SolverDivergence("Singular matrix in _lu_factor")
        sc[i] = 1.0 / big
    for k in range(n):
        big, imax = 0.0, k
        for i in range(k, n):
            t = sc[i] * abs(a[i, k])
            if t > big:
                big, imax = t, i
        if imax != k:
            a[[k, imax], :] = a[[imax, k], :]
            sc[k], sc[imax] = sc[imax], sc[k]
        piv[k] = imax
        if a[k, k] == 0.0:
            a[k, k] = tiny
        for i in range(k + 1, n):
            a[i, k] /= a[k, k]
            a[i, k + 1:n] -= a[i, k] * a[k, k + 1:n]


def _lu_backsolve(lu: np.ndarray, n: int, piv: np.ndarray, b: np.ndarray) -> None:
    """Solve LU·x = b in-place using output of _lu_factor."""
    ii = -1
    for i in range(n):
        ip = piv[i]
        s = b[ip]; b[ip] = b[i]
        if ii != -1:
            s -= np.dot(lu[i, ii:i], b[ii:i])
        elif s != 0.0:
            ii = i
        b[i] = s
    for i in range(n - 1, -1, -1):
        s = b[i]
        if i + 1 < n:
            s -= np.dot(lu[i, i + 1:n], b[i + 1:n])
        b[i] = s / lu[i, i]


def _iterative_solve(max_iter: int, x: np.ndarray, size: int, x_tol: float,
                     eval_fn, fvec, fjac, piv, p, sc) -> None:
    """Newton-Raphson solver mirroring OpenSTA newtonRaphson()."""
    for _ in range(max_iter):
        eval_fn()
        p[:size] = -fvec[:size]
        _lu_factor(fjac, size, piv, sc)
        _lu_backsolve(fjac, size, piv, p)
        converged = True
        for i in range(size):
            if abs(p[i]) > abs(x[i]) * x_tol:
                converged = False
            x[i] += p[i]
        if converged:
            return


def effective_load_cap(
    in_slew, c1, c2, rpi,
    delay_table, slew_table, slew_axis, load_axis,
    vth=0.5, vl=0.2, vh=0.8, is_rise=True,
    max_iter=30, tol=1e-6, ceff_init=None, ceff_clip=True,
    slew_derate=1.0, x_tol=1e-2,
):
    """Scalar DMP Ceff solver (matches OpenSTA findDriverParamsPi).

    Returns effective load capacitance seen by the gate driver.
    """
    print("input_slew:", in_slew)
    print("c1:", c1, " c2:", c2, " rpi:", rpi)

    def _arc_eval(ceff):
        d = _lookup_2d(slew_axis, load_axis, delay_table, in_slew, ceff)
        s = _lookup_2d(slew_axis, load_axis, slew_table,  in_slew, ceff)
        return float(d), float(s)

    ceff_cap = c1 + c2
    if ceff_cap <= 0.0:
        return 0.0

    d1 = _lookup_2d(slew_axis, load_axis, delay_table, in_slew, c2 + c1)
    d2 = _lookup_2d(slew_axis, load_axis, delay_table, in_slew, c2 + c1 + 1e-15)
    rd = -math.log(vth) * abs(d2 - d1) / 1e-15
    print("r_driver:", rd)

    if (rd < 1e-2) or (rpi <= 0.0) or (c1 <= 0.0) or (rpi < rd * 1e-3) or (c1 < c2 * 1e-3):
        print("effective_load_cap: fallback to pure cap")
        return float(ceff_cap)

    if c2 < c1 * 1e-3:
        ce = float(c1)
        if ceff_clip:
            ce = max(0.0, min(ce, ceff_cap))
        print("effective_load_cap: zero-c2 fallback, ceff =", ce)
        return ce

    z1   = 1.0 / (rpi * c1)
    a_rp = rpi * rd * c1 * c2
    b_rp = rd * (c1 + c2) + rpi * c1
    disc = b_rp * b_rp - 4.0 * a_rp
    if disc <= 0.0:
        ce = float(ceff_cap)
        if ceff_clip:
            ce = max(0.0, min(ce, ceff_cap))
        print("effective_load_cap: disc<=0 fallback", ce)
        return ce

    sq = math.sqrt(disc)
    p1 = (b_rp + sq) / (2.0 * a_rp)
    p2 = (b_rp - sq) / (2.0 * a_rp)
    p1p2 = p1 * p2
    k2 = z1 / p1p2
    k1 = (1.0 - k2 * (p1 + p2)) / p1p2
    k4 = (k1 * p1 + k2) / (p2 - p1)
    k3 = -k1 - k4

    z  = (c1 + c2) / (rpi * c1 * c2)
    A  = z / p1p2
    B  = (z - p1) / (p1 * (p1 - p2))
    D  = (z - p2) / (p2 * (p2 - p1))

    def _y0(t, cl):
        if t <= 0.0: return 0.0
        x = t / (rd * cl)
        return t - rd * cl * (1.0 - math.exp(-x))

    def _y0dt(t, cl):
        if t <= 0.0: return 0.0
        return 1.0 - math.exp(-t / (rd * cl))

    def _y0dcl(t, cl):
        if t <= 0.0: return 0.0
        x = t / (rd * cl)
        return rd * ((1.0 + x) * math.exp(-x) - 1)

    def _y(t, t0, dt, cl):
        t1 = t - t0
        if t1 <= 0.0: return 0.0
        if t1 <= dt: return _y0(t1, cl) / dt
        return (_y0(t1, cl) - _y0(t1 - dt, cl)) / dt

    def _dy(t, t0, dt, cl):
        t1 = t - t0
        if t1 <= 0.0:
            return 0.0, 0.0, 0.0
        if t1 <= dt:
            dt0 = -_y0dt(t1, cl) / dt
            ddt = -_y0(t1, cl) / (dt * dt)
            dcl = _y0dcl(t1, cl) / dt
        else:
            dt0 = -((_y0dt(t1, cl) - _y0dt(t1 - dt, cl)) / dt)
            ddt = (-((_y0(t1, cl) + _y0(t1 - dt, cl)) / (dt * dt))
                   + _y0dt(t1 - dt, cl) / dt)
            dcl = (_y0dcl(t1, cl) - _y0dcl(t1 - dt, cl)) / dt
        dt0 = -((_y0dt(t1, cl) - _y0dt(t1 - dt, cl)) / dt)
        ddt = (-((_y0(t1, cl) + _y0(t1 - dt, cl)) / (dt * dt))
               + _y0dt(t1 - dt, cl) / dt)
        dcl = (_y0dcl(t1, cl) - _y0dcl(t1 - dt, cl)) / dt
        return dt0, ddt, dcl

    def _current_balance(dt, ceff_t, ceff):
        ep1 = math.exp(-p1 * dt)
        ep2 = math.exp(-p2 * dt)
        ipi = (A * dt + (B / p1) * (1.0 - ep1) + (D / p2) * (1.0 - ep2)) / (rd * dt * dt)
        k = rd * ceff
        edt = math.exp(-dt / k) if ceff != 0.0 else 0.0
        iceff = (k * dt - (k * k) * (1.0 - edt)) / (rd * dt * dt)
        return ipi - iceff

    def _dcur_ddt(dt, ceff):
        ep1 = math.exp(-p1 * dt)
        ep2 = math.exp(-p2 * dt)
        k = rd * ceff
        edt = math.exp(-dt / k) if ceff != 0.0 else 0.0
        num = (-A * dt + B * dt * ep1 - (2 * B / p1) * (1 - ep1)
               + D * dt * ep2 - (2 * D / p2) * (1 - ep2)
               + k * (dt + dt * edt - 2 * k * (1 - edt)))
        return num / (rd * dt * dt * dt)

    def _dcur_dceff(dt, ceff):
        k = rd * ceff
        edt = math.exp(-dt / k) if ceff != 0.0 else 0.0
        return (2 * k - dt - (2 * k + dt) * edt) / (dt * dt)

    def _gate_timing(ceff):
        t50, s_tbl = _arc_eval(ceff)
        s = s_tbl * slew_derate
        t20 = t50 - s * (vth - vl) / (vh - vl)
        return t50, t20, s

    x_nr = [None]
    fvec = np.zeros(3, dtype=np.float64)
    fjac = np.zeros((3, 3), dtype=np.float64)

    def _residuals():
        t0, dt, ceff = float(x_nr[0][0]), float(x_nr[0][1]), float(x_nr[0][2])
        if ceff < 0.0: raise SolverDivergence("ceff < 0")
        if ceff > (c1 + c2): raise SolverDivergence("ceff > c1+c2")
        if dt <= 0.0: raise SolverDivergence("dt <= 0")
        t50, t20, slew = _gate_timing(ceff)
        if slew == 0.0: raise SolverDivergence("slew = 0")
        ct = slew / (vh - vl)
        if ct > 1.4 * dt: ct = 1.4 * dt
        y50 = _y(t50, t0, dt, ceff)
        y20 = _y(t20, t0, dt, ceff)
        fvec[0] = _current_balance(dt, ct, ceff)
        fvec[1] = y50 - vth
        fvec[2] = y20 - vl
        fjac[0, 0] = 0.0
        fjac[0, 1] = _dcur_ddt(dt, ceff)
        fjac[0, 2] = _dcur_dceff(dt, ceff)
        d50_t0, d50_dt, d50_ceff = _dy(t50, t0, dt, ceff)
        fjac[1, 0], fjac[1, 1], fjac[1, 2] = d50_t0, d50_dt, d50_ceff
        d20_t0, d20_dt, d20_ceff = _dy(t20, t0, dt, ceff)
        fjac[2, 0], fjac[2, 1], fjac[2, 2] = d20_t0, d20_dt, d20_ceff

    def _init_guess(ceff):
        t50, _, slew = _gate_timing(ceff)
        dt = slew / (vh - vl)
        t0 = t50 + math.log(1.0 - vth) * rd * ceff - vth * dt
        xv = np.array([t0, dt, ceff], dtype=np.float64)
        x_nr[0] = xv
        piv = np.zeros(3, dtype=np.int64)
        pp  = np.zeros(3, dtype=np.float64)
        sc  = np.zeros(3, dtype=np.float64)
        _iterative_solve(100, xv, 3, 1e-6, _residuals, fvec, fjac, piv, pp, sc)
        return xv.tolist()

    sol = _init_guess(c1 + c2)
    return sol[2]


# ---------------------------------------------------------------------------
# Batched GPU Ceff (torch, differentiable)
# ---------------------------------------------------------------------------

def _axis_locate(axis: torch.Tensor, x: torch.Tensor):
    """Find lower-bound index and interpolation weight for x in axis. [N,K] → [N]."""
    eps = 1e-30
    i0  = torch.clamp(torch.sum(x.unsqueeze(-1) >= axis[:, 1:], dim=-1), 0, 5)
    i0u = i0.unsqueeze(-1)
    x0  = axis.gather(1, i0u).squeeze(-1)
    x1  = axis.gather(1, i0u + 1).squeeze(-1)
    denom = x1 - x0
    t = torch.where(denom.abs() > eps, (x - x0) / (denom + eps), torch.zeros_like(x))
    return i0, t


def _lut_lookup_batch(ax0: torch.Tensor, ax1: torch.Tensor, tbl: torch.Tensor,
                      s: torch.Tensor, c: torch.Tensor, eps: float = 1e-18):
    """Batched bilinear LUT interpolation. ax0/ax1: [T,K], tbl: [T,K,K]."""
    T = s.shape[0]
    dev = s.device
    i0, a = _axis_locate(ax0, s)
    j0, b = _axis_locate(ax1, c)
    ar = torch.arange(T, device=dev)
    i1, j1 = i0 + 1, j0 + 1
    v00 = tbl[ar, i0, j0]; v01 = tbl[ar, i0, j1]
    v10 = tbl[ar, i1, j0]; v11 = tbl[ar, i1, j1]
    return (1 - a) * (1 - b) * v00 + (1 - a) * b * v01 + a * (1 - b) * v10 + a * b * v11


def _ramp_derivatives(t, cl, rd, cap_to_F, eps):
    """Compute y0, dy0/dt, dy0/dcl for a ramp-driven RC node (vectorized)."""
    tp = torch.clamp(t, min=0.0)
    rc = rd * cl * cap_to_F
    rc_safe = (rc + eps).clamp_min(1e-18)
    x = tp / rc_safe
    ex = torch.exp(-x)
    z = torch.zeros_like(tp)
    y0   = torch.where(t > 0.0, tp - rc * (1.0 - ex), z)
    y0dt = torch.where(t > 0.0, 1.0 - ex,              z)
    y0dc = torch.where(t > 0.0, (rd * cap_to_F) * ((1.0 + x) * ex - 1.0), z)
    return y0, y0dt, y0dc


def _ramp_output_and_grad(t, t0, dt, cl, rd, cap_to_F, eps):
    t1 = t - t0
    z  = torch.zeros_like(t1)
    m1 = t1 > 0.0
    t1p = torch.where(m1, t1, z)
    m2  = m1 & (t1p <= dt)
    m3  = m1 & (t1p > dt)
    inv = 1.0 / (dt + eps)
    inv2 = inv * inv
    y0a, y0dt_a, y0dc_a = _ramp_derivatives(t1p,      cl, rd, cap_to_F, eps)
    y0b, y0dt_b, y0dc_b = _ramp_derivatives(t1p - dt, cl, rd, cap_to_F, eps)
    y     = torch.where(m2, y0a * inv, torch.where(m3, (y0a - y0b) * inv, z))
    dt0   = torch.where(m2, -y0dt_a * inv, torch.where(m3, -(y0dt_a - y0dt_b) * inv, z))
    ddt   = torch.where(m2, -y0a * inv2,   torch.where(m3, -(y0a + y0b) * inv2 + y0dt_b * inv, z))
    dcl   = torch.where(m2,  y0dc_a * inv, torch.where(m3, (y0dc_a - y0dc_b) * inv, z))
    return y, dt0, ddt, dcl


def _current_residual(dt, ceff, p1, p2, A, B, D, rd, c2F, eps):
    ep1 = torch.exp(-p1 * dt)
    ep2 = torch.exp(-p2 * dt)
    ipi = (A * dt + (B / p1) * (1 - ep1) + (D / p2) * (1 - ep2)) / (rd * dt * dt + eps)
    k  = rd * ceff * c2F
    ks = (k + eps).clamp_min(1e-18)
    ek = torch.exp(-dt / ks)
    iceff = (k * dt - k * k * (1 - ek)) / (rd * dt * dt + eps)
    return ipi - iceff


def _current_dt_grad(dt, ceff, p1, p2, A, B, D, rd, c2F, eps):
    ep1 = torch.exp(-p1 * dt)
    ep2 = torch.exp(-p2 * dt)
    k  = rd * ceff * c2F
    ks = (k + eps).clamp_min(1e-18)
    ek = torch.exp(-dt / ks)
    num = (-A * dt
           + B * dt * ep1 - (2 * B / p1) * (1 - ep1)
           + D * dt * ep2 - (2 * D / p2) * (1 - ep2)
           + k * (dt + dt * ek - 2 * k * (1 - ek)))
    return num / (rd * dt * dt * dt + eps)


def _current_ceff_grad(dt, ceff, rd, c2F, eps):
    k  = rd * ceff * c2F
    ks = (k + eps).clamp_min(1e-18)
    ek = torch.exp(-dt / ks)
    return (2 * k - dt - (2 * k + dt) * ek) / (dt * dt + eps) * c2F


def _arc_thresholds(ceff, in_slew, axslw, axcap, dtab, stab, vth, vl, vh, derate, c2F, eps):
    t50, s_tbl = (_lut_lookup_batch(axslw, axcap, dtab, in_slew, ceff * c2F, eps),
                  _lut_lookup_batch(axslw, axcap, stab, in_slew, ceff * c2F, eps))
    sm = s_tbl * derate
    t20 = t50 - sm * (vth - vl) / (vh - vl)
    return t50, t20, sm


def batch_effective_load(
    in_slew: torch.Tensor,
    c1: torch.Tensor, c2: torch.Tensor, rpi: torch.Tensor,
    delay_table: torch.Tensor, slew_table: torch.Tensor,
    slew_axis: torch.Tensor, load_axis: torch.Tensor,
    vth: float = 0.5, vl: float = 0.2, vh: float = 0.8,
    slew_derate: float = 1.0,
    eps: float = 1e-30, damp: float = 1.0, diag_eps: float = 1e-18,
    use_where: bool = True,
) -> torch.Tensor:
    """Batched DMP Ceff solver (GPU-accelerated, one Newton step per call).

    Shapes: in_slew [T], c1/c2/rpi [T], tables [T,7,7], axes [T,7].
    Returns ceff [T] in Farads.
    """
    c2F  = 1e-15   # fF -> F
    csc  = 1e15    # F  -> fF
    c1ff = c1 * csc; c2ff = c2 * csc
    ctot = c1ff + c2ff

    d1 = _lut_lookup_batch(slew_axis, load_axis, delay_table, in_slew, ctot * c2F)
    d2 = _lut_lookup_batch(slew_axis, load_axis, delay_table, in_slew, (ctot + 1.0) * c2F)
    rd = -math.log(vth) * abs(d2 - d1) / c2F

    ceff_out = torch.clamp(ctot, min=0.0) * c2F

    active = (ctot > 0.0) & (rd >= 1e-2) & (rpi > 0.0) & (c1ff > 0.0) & \
             (rpi >= rd * 1e-3) & (c1ff >= c2ff * 1e-3)

    zero_c2 = active & (c2ff < c1ff * 1e-3)
    c1_clip = torch.minimum(torch.maximum(c1ff, torch.zeros_like(c1ff)), ctot)
    ceff_out = torch.where(zero_c2, c1_clip * c2F, ceff_out)
    active   = active & (~zero_c2)

    c1s, c2s = c1ff.clamp_min(eps), c2ff.clamp_min(eps)
    rpis, rds = rpi.clamp_min(eps), rd.clamp_min(eps)

    af = rpis * rds * c1s * c2s
    bf = rds * (c1s + c2s) + rpis * c1s
    disc = bf * bf - 4.0 * af
    active &= (disc > 0.0)

    sq   = torch.sqrt(torch.where(active, disc, torch.ones_like(disc)))
    i2a  = 1.0 / (2.0 * torch.where(active, af, torch.ones_like(af)) + eps)
    p1   = ((bf + sq) * i2a) / c2F
    p2   = ((bf - sq) * i2a) / c2F
    p1   = torch.where(active, p1, torch.ones_like(p1))
    p2   = torch.where(active, p2, torch.full_like(p2, 2.0))

    dz   = torch.where(active, (rpis * c1s * c2s).clamp_min(1e-18), torch.ones_like(rpis))
    nz   = torch.where(active, c1s + c2s, torch.zeros_like(c1s))
    z    = (nz / (dz + eps)) / c2F
    p1p2 = p1 * p2
    A    = torch.where(active, z / (p1p2 + eps), torch.zeros_like(z))
    B    = torch.where(active, (z - p1) / (p1 * (p1 - p2) + eps), torch.zeros_like(z))
    D    = torch.where(active, (z - p2) / (p2 * (p2 - p1) + eps), torch.zeros_like(z))

    ceff = torch.clamp(ctot, min=eps)
    t50, t20, slew0 = _arc_thresholds(ceff, in_slew, slew_axis, load_axis,
                                       delay_table, slew_table,
                                       vth, vl, vh, slew_derate, c2F, eps)
    dt   = torch.clamp(slew0 / (vh - vl + eps), min=eps)
    rd_s = torch.where(active, rd, torch.ones_like(rd))
    t0   = t50 + math.log(1.0 - vth) * rd_s * (ceff * c2F) - vth * dt

    y50, d50_t0, d50_dt, d50_ceff = _ramp_output_and_grad(t50, t0, dt, ceff, rd_s, c2F, eps)
    y20, d20_t0, d20_dt, d20_ceff = _ramp_output_and_grad(t20, t0, dt, ceff, rd_s, c2F, eps)
    f0 = _current_residual(dt, ceff, p1, p2, A, B, D, rd_s, c2F, eps)
    f1 = y50 - vth; f2 = y20 - vl

    j01 = _current_dt_grad(dt, ceff, p1, p2, A, B, D, rd_s, c2F, eps)
    j02 = _current_ceff_grad(dt, ceff, rd_s, c2F, eps)

    a21, a22, a23 = d50_t0, d50_dt + diag_eps, d50_ceff
    a31, a32, a33 = d20_t0, d20_dt,            d20_ceff + diag_eps
    trm1 = a21 * a33 - a23 * a31
    trm2 = a21 * a32 - a22 * a31
    det  = -j01 * trm1 + j02 * trm2
    det_s = det + eps * torch.where(det >= 0.0, torch.ones_like(det), -torch.ones_like(det))
    r1, r2, r3 = -f0, -f1, -f2
    num_dc = -j01 * (a21 * r3 - r2 * a31) + r1 * trm2
    delta  = torch.where(active, (num_dc / det_s) * damp, torch.zeros_like(num_dc))
    ceff   = torch.minimum(torch.maximum(ceff + delta, torch.zeros_like(ceff)), ctot)
    ceff_out = torch.where(active, ceff * c2F, ceff_out)
    return ceff_out


# ---------------------------------------------------------------------------
# Gate arc LUT interpolation helpers (used by circuit graph evaluator)
# ---------------------------------------------------------------------------

def lut_axis_index(axis_k: torch.Tensor, x: torch.Tensor):
    """Return (i0, alpha) for batched bilinear lookup. axis_k: [M,7], x: [M]."""
    eps = 1e-30
    i0  = torch.clamp(torch.sum(x.unsqueeze(-1) >= axis_k[:, 1:], dim=-1), 0, 5)
    i0u = i0.unsqueeze(-1)
    x0  = axis_k.gather(1, i0u).squeeze(-1)
    x1  = axis_k.gather(1, i0u + 1).squeeze(-1)
    denom = x1 - x0
    t = torch.where(denom.abs() > eps, (x - x0) / (denom + eps), torch.zeros_like(x))
    return i0, t


def lut_bilinear_batch(tbl_7x7, ax_slew, ax_cap, in_slew, cap):
    """Bilinear interpolation into [M,7,7] table. All inputs [M]."""
    M   = in_slew.shape[0]
    dev = in_slew.device
    ar  = torch.arange(M, device=dev)
    i0, a = lut_axis_index(ax_slew, in_slew)
    j0, b = lut_axis_index(ax_cap,  cap)
    i1, j1 = i0 + 1, j0 + 1
    v00 = tbl_7x7[ar, i0, j0]; v01 = tbl_7x7[ar, i0, j1]
    v10 = tbl_7x7[ar, i1, j0]; v11 = tbl_7x7[ar, i1, j1]
    return (1 - a) * (1 - b) * v00 + (1 - a) * b * v01 + a * (1 - b) * v10 + a * b * v11


def eval_arc_timing(in_slew, load, dtab, stab, ax_slew, ax_cap):
    """Evaluate gate arc delay and output slew from LUTs.

    in_slew/load: [T], tables: [A,7,7], axes: [A,7].
    Returns (arc_delay, arc_slew): both [T].
    """
    T   = dtab.shape[0]
    dev = in_slew.device
    ar  = torch.arange(T, device=dev)
    i0, a = lut_axis_index(ax_slew, in_slew)
    j0, b = lut_axis_index(ax_cap,  load)
    i1, j1 = i0 + 1, j0 + 1

    def _bl(tbl):
        v00 = tbl[ar, i0, j0]; v01 = tbl[ar, i0, j1]
        v10 = tbl[ar, i1, j0]; v11 = tbl[ar, i1, j1]
        return (1-a)*(1-b)*v00 + (1-a)*b*v01 + a*(1-b)*v10 + a*b*v11

    return _bl(dtab), _bl(stab)


# ---------------------------------------------------------------------------
# Wire RC propagation
# ---------------------------------------------------------------------------

def ramp_threshold_time(tau, T, v, newton_iters=1, eps=1e-18):
    """Compute time for a ramp-driven RC node to cross threshold v.

    tau: Elmore time constant [s], T: ramp duration [s], v: threshold.
    All tensors, broadcastable.
    """
    exp_T = torch.exp(-T / torch.clamp(tau, min=eps))
    yT    = 1.0 - (tau / torch.clamp(T, min=eps)) * (1.0 - exp_T)
    K     = (tau / torch.clamp(T, min=eps)) * (1.0 - exp_T)
    t_after = T - tau * torch.log(torch.clamp((1.0 - v) / torch.clamp(K, min=eps), min=eps))
    t = torch.clamp(v * T + 0.5 * tau, min=0.0)
    for _ in range(newton_iters):
        exp_t = torch.exp(-t / torch.clamp(tau, min=eps))
        g  = (t / torch.clamp(T, min=eps)) - (tau / torch.clamp(T, min=eps)) * (1.0 - exp_t) - v
        gp = (1.0 - exp_t) / torch.clamp(T, min=eps)
        t  = torch.clamp(t - g / torch.clamp(gp, min=eps), min=0.0)
        t  = torch.minimum(t, T)
    return torch.where(v > yT, t_after, t)


def rc_propagate(tau, slew_in, vth=0.5, vl=0.2, vh=0.8, slew_derate=1.0):
    """Wire delay and slew at a sink driven by a ramp with Elmore constant tau.

    slew_in: input slew measured between vl and vh.
    Returns (delay_inc, slew_out).
    """
    T     = slew_in / (vh - vl)
    t_vth = ramp_threshold_time(tau, T, torch.as_tensor(vth, device=tau.device, dtype=tau.dtype))
    t_vl  = ramp_threshold_time(tau, T, torch.as_tensor(vl,  device=tau.device, dtype=tau.dtype))
    t_vh  = ramp_threshold_time(tau, T, torch.as_tensor(vh,  device=tau.device, dtype=tau.dtype))
    t_ref = vth * T
    delay_inc = torch.clamp(t_vth - t_ref, min=0.0)
    slew_out  = torch.maximum(torch.clamp((t_vh - t_vl) / slew_derate, min=slew_in), slew_in)
    return delay_inc, slew_out


# ---------------------------------------------------------------------------
# Continuous buffer timing model
# ---------------------------------------------------------------------------

def _broadcast_n(n, ref):
    n_t = n if torch.is_tensor(n) else torch.tensor(n, device=ref.device, dtype=ref.dtype)
    n_t = n_t.squeeze(1) if (n_t.dim() == 2 and n_t.shape[1] == 1) else n_t
    nb  = n_t.unsqueeze(1) if n_t.dim() == 1 else n_t
    return n_t.reshape(-1) if n_t.dim() <= 1 else n_t, nb


def half_span_wire(sp_xy, ep_xy, xcen_buf, xin_buf):
    """Half-span of wire after buffer placement (Manhattan length)."""
    _, nb = _broadcast_n(1, sp_xy)
    xout_buf = 2 * xcen_buf - xin_buf
    return torch.abs((ep_xy - sp_xy - nb * (xout_buf - xin_buf)) / 2).sum(dim=1) * DBU_NM


def distributed_buf_timing(sp_xy, ep_xy, xcen_buf, xin_buf, buf_in_cap,
                            sp_slew, ep_load, n,
                            fs_s, fs_l, fd_s, fd_l):
    """Compute delay and slew for a distributed buffer chain.

    Returns (delay, slew) — rise/fall selection is done by the caller.
    """
    n_vec, _ = _broadcast_n(n, sp_xy)
    ln   = half_span_wire(sp_xy, ep_xy, xcen_buf, xin_buf)
    Cb   = buf_in_cap
    Lo   = ep_load
    nabs = torch.abs(n_vec)
    act  = torch.minimum(nabs, torch.ones_like(nabs))
    inner = torch.clamp(nabs - 1, min=0)

    C_ln  = WIRE_C * ln
    R_op  = WIRE_R * ln * 0.5

    L_in  = C_ln + Cb
    el_in = R_op * L_in
    S_in  = torch.sqrt((fs_l(L_in) * 1e12) ** 2 + 1.921 * (1e12 * el_in) ** 2) / 1e12

    L_out  = C_ln + Lo
    el_out = R_op * L_out
    S_sp   = torch.sqrt((sp_slew * 1e12) ** 2 + 1.921 * (1e12 * el_in) ** 2) / 1e12

    delay = (act   * (C_ln + Cb) * R_op
           + (C_ln + Lo) * R_op
           + inner * (C_ln + Cb) * R_op
           + inner * (fd_s(S_in) + fd_l(L_in))
           + act   * (fd_s(S_sp) + fd_l(L_out)))

    slew_buf = torch.sqrt((fs_s(S_sp) + fs_l(L_out) * 1e12) ** 2
                          + 1.921 * (1e12 * el_in) ** 2) / 1e12
    slew = ((1 - act) * torch.sqrt((sp_slew * 1e12) ** 2
                                   + 1.921 * (1e12 * el_out) ** 2) / 1e12
           + act * slew_buf)
    return delay, slew


# ---------------------------------------------------------------------------
# LUT regression fitting (used to build analytical buffer models)
# ---------------------------------------------------------------------------

def _poly_feature_matrix(s_flat, c_flat, s_sc, c_sc):
    sn = s_flat / s_sc
    cn = c_flat / c_sc
    sc = torch.sign(sn) * torch.abs(sn).pow(1.0 / 3.0)
    cols = [torch.ones_like(s_flat), cn, sn, sc, sn * sn]
    return torch.stack(cols, dim=1), ["bias", "c_lin", "s_lin", "s_cbrt", "s_sq"]


def _weighted_lstsq(X, y):
    sw = torch.sqrt(1.0 / (torch.abs(y) + 1e-12))
    return torch.linalg.lstsq(X * sw.unsqueeze(1), (y * sw).unsqueeze(1)).solution.squeeze()


def _fit_arc_model(table, slew_axis, cap_axis, kind: str):
    table = torch.as_tensor(table, dtype=torch.float32)
    s = torch.as_tensor(slew_axis, dtype=torch.float32).reshape(-1)
    c = torch.as_tensor(cap_axis,  dtype=torch.float32).reshape(-1)
    s_sc = torch.max(torch.abs(s)).clamp_min(1e-12)
    c_sc = torch.max(torch.abs(c)).clamp_min(1e-12)
    s, c = (t.flatten() for t in torch.meshgrid(s, c, indexing="ij"))
    X, names = _poly_feature_matrix(s, c, s_sc, c_sc)
    y = table.flatten()
    y_sc = torch.max(torch.abs(y)).clamp_min(1e-12)
    coeffs = _weighted_lstsq(X, y / y_sc) * y_sc
    return {"kind": kind, "coeffs": coeffs,
            "s_scale": s_sc.item(), "c_scale": c_sc.item(), "feat_names": names}


def fit_delay_model(delay_table, slew_axis, cap_axis):
    return _fit_arc_model(delay_table, slew_axis, cap_axis, kind="delay_poly")


def fit_slew_model(slew_table, slew_axis, cap_axis):
    return _fit_arc_model(slew_table, slew_axis, cap_axis, kind="slew_poly")


def arc_model_predict(model, slew_axis, cap_axis):
    coeffs = torch.as_tensor(model["coeffs"], dtype=torch.float32)
    s_sc   = torch.as_tensor(model.get("s_scale", 1.0), dtype=torch.float32)
    c_sc   = torch.as_tensor(model.get("c_scale", 1.0), dtype=torch.float32)
    s = torch.as_tensor(slew_axis, dtype=torch.float32).reshape(-1)
    c = torch.as_tensor(cap_axis,  dtype=torch.float32).reshape(-1)
    S, C = torch.meshgrid(s, c, indexing="ij")
    X, _ = _poly_feature_matrix(S.flatten(), C.flatten(), s_sc, c_sc)
    return (X @ coeffs).reshape(S.shape)
