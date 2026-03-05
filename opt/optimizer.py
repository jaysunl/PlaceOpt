"""
opt.optimizer — Gradient-based joint gate-sizing and buffer-insertion optimizer.

``GradientOptimizer`` wraps three independent Adam optimizers that jointly
update gate-sizing logits, buffer-insertion logits, and cell positions.

The loss function is:

    L = λ_t · ΔTNS/b_TNS  +  λ_p · ΔPower/b_Power
      + λ_q(epoch) · Q_penalty            ← grows over time (annealing)
      + λ_d · max(density − 1, 0)²        ← density overflow penalty
      + λ_m · avg_displacement             ← legalization anchor

where Δ* indicates the improvement over the baseline (signed, so that
improvement gives a negative loss term and worsening gives positive).

Scheduling
----------
* ``quant_scale`` is multiplied by ``quant_anneal`` each epoch, which
  gradually tightens the discretization pressure.
* Fixed cells and macro cells have their position gradients zeroed so that
  they cannot be moved.
"""

from __future__ import annotations

from typing import Optional, Tuple
import time

import torch
import torch.nn as nn

from placeopt.timing.engine import TimingEngine
from placeopt.timing.graph import STAGraph


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

class OptConfig:
    """Hyperparameters for the gradient-based optimizer."""

    # Learning rates.
    lr_size:   float = 0.01    # gate sizing (U_flat)
    lr_buf:    float = 0.005   # buffer insertion (buffering_tensor)
    lr_pos:    float = 1.5     # cell movement (cell_xy), in DBU

    # Loss weights.
    tns_weight:   float = 80.0
    power_weight: float = 40.0
    density_weight: float = 0.0    # set > 0 to activate density penalty
    movement_weight: float = 0.0   # set > 0 to anchor cells near initial pos

    # Quantization penalty schedule.
    quant_init:   float = 0.001    # initial scale
    quant_anneal: float = 1.15     # multiplier per epoch
    quant_max:    float = 50.0     # cap on the scale

    # Density overflow threshold (fraction of bin area).
    density_clip: float = 1.0

    # Number of optimization epochs.
    epochs: int = 150

    # Wire / gate mode passed to TimingEngine.forward().
    wire_mode: str = "simple"
    gate_mode: str = "simple"


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class GradientOptimizer:
    """
    Joint gate-sizing, buffer-insertion, and placement optimizer.

    Parameters
    ----------
    engine  : TimingEngine
    config  : OptConfig (optional; uses defaults if None)
    device  : torch.device

    Example
    -------
    ::

        opt = GradientOptimizer(engine, config, device)
        opt.optimize(baseline_tns=b_tns, baseline_power=b_pow)
    """

    def __init__(
        self,
        engine: TimingEngine,
        config: Optional[OptConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.engine  = engine
        self.cfg     = config or OptConfig()
        self.device  = device or torch.device("cpu")
        gd = engine.gd

        # Three separate Adam optimizers so each parameter group can have
        # its own learning rate and potentially different schedules.
        self.opt_size = torch.optim.Adam([gd.U_flat],           lr=self.cfg.lr_size)
        self.opt_buf  = torch.optim.Adam([gd.buffering_tensor], lr=self.cfg.lr_buf)
        self.opt_pos  = torch.optim.Adam([gd.cell_xy],          lr=self.cfg.lr_pos)

    # ------------------------------------------------------------------

    def optimize(
        self,
        baseline_tns:   float,
        baseline_power: float,
    ) -> None:
        """
        Run the full optimization loop.

        Parameters
        ----------
        baseline_tns   : OpenSTA TNS before optimization (negative, in seconds).
        baseline_power : OpenSTA dynamic power before optimization (Watts).
        """
        cfg = self.cfg
        gd  = self.engine.gd

        q_scale = cfg.quant_init

        print(f"[OPT] Starting gradient optimization: "
              f"{cfg.epochs} epochs, "
              f"lr_size={cfg.lr_size}, lr_buf={cfg.lr_buf}, lr_pos={cfg.lr_pos}")

        # Normalise baselines so the loss is dimensionless.
        b_tns  = float(baseline_tns)   if abs(baseline_tns)   > 1e-20 else -1e-12
        b_pow  = float(baseline_power) if abs(baseline_power) > 1e-20 else  1e-12

        for epoch in range(cfg.epochs):
            t0 = time.perf_counter()
            self.opt_size.zero_grad()
            self.opt_buf.zero_grad()
            self.opt_pos.zero_grad()

            tns, wns, power, q_pen, density = self.engine.forward(
                wire_mode=cfg.wire_mode,
                gate_mode=cfg.gate_mode,
            )

            loss = (cfg.tns_weight   * tns   / abs(b_tns)
                    + cfg.power_weight * power / abs(b_pow)
                    + q_scale * q_pen)

            if cfg.density_weight > 0 and density is not None:
                overflow = (density - cfg.density_clip).clamp_min(0.0).pow(2).sum()
                loss = loss + cfg.density_weight * overflow

            if cfg.movement_weight > 0 and gd.init_cell_idx.numel():
                disp = (gd.cell_xy[gd.init_cell_idx] - gd.init_cell_xy).norm(dim=1).mean()
                loss = loss + cfg.movement_weight * disp

            loss.backward()

            # Zero gradients for fixed / macro cells so they can't move.
            if gd.gate_movable_mask.numel():
                with torch.no_grad():
                    gd.cell_xy.grad *= gd.gate_movable_mask.unsqueeze(1)

            self.opt_size.step()
            self.opt_buf.step()
            self.opt_pos.step()

            q_scale = min(q_scale * cfg.quant_anneal, cfg.quant_max)

            if epoch % 10 == 0 or epoch == cfg.epochs - 1:
                elapsed = time.perf_counter() - t0
                print(f"[OPT] epoch {epoch:4d}/{cfg.epochs}"
                      f"  TNS={tns.item():.3e}"
                      f"  WNS={wns.item():.3e}"
                      f"  P={power.item():.3e}"
                      f"  Q={q_pen.item():.3e}"
                      f"  q_scale={q_scale:.3f}"
                      f"  t={elapsed:.2f}s")

    def run_pass(
        self,
        baseline_tns: float,
        baseline_power: float,
        buf_bias: float = -12.0,
        epochs: Optional[int] = None,
    ) -> None:
        """
        Convenience wrapper that re-initialises the buffer logits to *buf_bias*
        before running ``optimize()``.

        A more negative ``buf_bias`` makes the optimizer start from a
        "no-buffer" prior (conserves area); a less negative value allows the
        optimizer more freedom.

        Parameters
        ----------
        buf_bias : initial value for ``buffering_tensor`` (applied in-place).
        epochs   : override config epoch count.
        """
        with torch.no_grad():
            self.engine.gd.buffering_tensor.fill_(buf_bias)
        if epochs is not None:
            self.cfg.epochs = epochs
        self.optimize(baseline_tns, baseline_power)
