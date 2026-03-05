"""
timing.pi_model — Differentiable two-pole Π load model.

Given the three Elmore moments (M₀, M₁, M₂) of a net rooted at a driver
pin, this module computes the equivalent Π model seen by the driver:

         Rπ
    ○──┤vvvv├──○
    |              |
   C₂             C₁
    |              |
   GND           GND

The Π model is used by the gate delay kernel to compute the effective
capacitance (Ceff) seen by the driver, accounting for distributed RC
effects in long or multi-fanout nets.

Moment definitions (scaled by 1e15 for numerical stability):
    M₀  ≈ total downstream capacitance
    M₁  ≈ −Σ (Rᵢ · Cᵢ²)    (first resistance-cap product)
    M₂  ≈ Σ (Rᵢ² · Cᵢ³)    (second resistance-cap product)

Conversion formulae (AWE two-pole fit):
    C₁  = M₁² / M₂
    C₂  = M₀ − C₁
    Rπ  = −(M₂/M₁)² / M₁

A custom autograd Function is used so the backward pass can be expressed
analytically, which is faster and more numerically stable than letting
PyTorch trace through the moment-to-Π algebra.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from placeopt.timing.constants import M0, M1, M2, VIA_RESISTANCE


class PiModelFunction(torch.autograd.Function):
    """
    Forward: (M₀, M₁, M₂) → (C₁, C₂, Rπ)
    Backward: analytical gradients via chain rule on the conversion formulae.
    """

    @staticmethod
    def forward(ctx, moments: torch.Tensor):
        """
        Parameters
        ----------
        moments : [N, 3]  columns = [M₀, M₁, M₂] (scaled)

        Returns
        -------
        c1, c2, rpi : each [N]
        """
        if moments.dim() == 1:
            moments = moments.unsqueeze(0)

        r = VIA_RESISTANCE

        # Apply via-resistance correction to moments at the endpoint.
        m0 = moments[:, M0] + 1e-5
        m1 = moments[:, M1] - r * moments[:, M0] ** 2 - 1e-10
        m2 = (moments[:, M2]
              - 2.0 * r * moments[:, M0] * moments[:, M1]
              + r ** 2 * moments[:, M0] ** 3
              + 1e-15)

        c1  =  m1 * m1 / m2
        c2  =  m0 - c1
        rpi = -(m2 / m1) ** 2 / m1

        ctx.save_for_backward(moments[:, M0], moments[:, M1], m1, m2)
        return c1, c2, rpi

    @staticmethod
    def backward(ctx, d_c1, d_c2, d_rpi):
        m0_raw, m1_raw, y1, y2 = ctx.saved_tensors
        r = VIA_RESISTANCE

        # Gradient w.r.t. y0 (= m0 before via correction)
        grad_y0 = d_c2.clone()

        # Gradient w.r.t. y1 (= m1 after via correction)
        grad_y1 = (d_c1 * 2.0 * y1 / y2
                   + d_c2 * (-2.0 * y1 / y2)
                   + d_rpi * 3.0 * y2 ** 2 / y1 ** 4)

        # Gradient w.r.t. y2 (= m2 after via correction)
        grad_y2 = (d_c1 * (-(y1 ** 2) / y2 ** 2)
                   + d_c2 * (y1 ** 2 / y2 ** 2)
                   + d_rpi * (-2.0 * y2 / y1 ** 3))

        # Chain rule back to M₀, M₁, M₂
        # dy0/dM0 = 1,  dy1/dM0 = 0,  dy2/dM0 = -2r·M0 (approx) + 3r²·M0²
        # dy1/dM1 = 1,  dy2/dM1 = -2r·M0
        # dy2/dM2 = 1
        grad_M0 = grad_y0 + grad_y2 * (-2.0 * r * m0_raw + 3.0 * r ** 2 * m0_raw ** 2)
        grad_M1 = grad_y1 + grad_y2 * (-2.0 * r * m0_raw)
        grad_M2 = grad_y2

        grad_moments = torch.stack([grad_M0, grad_M1, grad_M2], dim=1)
        return grad_moments


def compute_pi_model(moments: torch.Tensor):
    """
    Convenience wrapper: returns (c1, c2, rpi) from a [N,3] moment tensor.
    """
    return PiModelFunction.apply(moments)


# ---------------------------------------------------------------------------
# Elmore-delay custom autograd (simple ramp model)
# ---------------------------------------------------------------------------

class ElmoreDelayFunction(torch.autograd.Function):
    """
    Compute Elmore wire delay and output slew for a single RC segment.

    The driver is modelled as a linear ramp; the output waveform is the
    standard RC-ramp convolution:

        delay_inc ≈ elmore_tau − slew_in · 0.38
        slew_out  ≈ √(slew_in² + (2.2 · elmore_tau)²)

    The Elmore time constant is  τ = R · M₀  where M₀ is the downstream
    capacitance (zeroth moment) of the sub-tree rooted at the current node.
    """

    @staticmethod
    def forward(ctx, tau, slew_r, slew_f):
        """
        Parameters
        ----------
        tau    : [N]  Elmore time constant (seconds)
        slew_r : [N]  input rise slew (seconds)
        slew_f : [N]  input fall slew (seconds)

        Returns
        -------
        delay_r, slew_out_r, delay_f, slew_out_f : each [N]
        """
        K = 0.38   # Elmore 50%-crossing factor for ramp input
        S = 2.2    # factor for output-slew Elmore formula

        delay_r = (tau - slew_r * K).clamp_min(0.0)
        delay_f = (tau - slew_f * K).clamp_min(0.0)

        slew_out_r = torch.sqrt(slew_r ** 2 + (S * tau) ** 2).clamp_min(slew_r)
        slew_out_f = torch.sqrt(slew_f ** 2 + (S * tau) ** 2).clamp_min(slew_f)

        ctx.save_for_backward(tau, slew_r, slew_f, slew_out_r, slew_out_f,
                               torch.tensor(K), torch.tensor(S))
        return delay_r, slew_out_r, delay_f, slew_out_f

    @staticmethod
    def backward(ctx, d_dr, d_sr_out, d_df, d_sf_out):
        tau, slew_r, slew_f, sr_out, sf_out, K_t, S_t = ctx.saved_tensors
        K = K_t.item()
        S = S_t.item()

        # d(delay_r)/d(tau) = 1 where delay_r > 0 else 0
        mask_r = (tau - slew_r * K) > 0
        mask_f = (tau - slew_f * K) > 0

        grad_tau = (d_dr * mask_r.float()
                    + d_df * mask_f.float()
                    + d_sr_out * S ** 2 * tau / sr_out.clamp_min(1e-30)
                    + d_sf_out * S ** 2 * tau / sf_out.clamp_min(1e-30))

        grad_slew_r = (-d_dr * K * mask_r.float()
                       + d_sr_out * slew_r / sr_out.clamp_min(1e-30))
        grad_slew_f = (-d_df * K * mask_f.float()
                       + d_sf_out * slew_f / sf_out.clamp_min(1e-30))

        return grad_tau, grad_slew_r, grad_slew_f
