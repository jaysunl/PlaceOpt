"""
timing.engine — Differentiable Timing Engine (TimingEngine).

``TimingEngine`` is a PyTorch ``nn.Module`` that performs a complete,
differentiable static timing analysis (STA) forward pass over the circuit
DAG encoded in an ``STAGraph``.

Three learnable parameters are updated by gradient descent:

  ``U_flat``           — gate-size logits.  Segment-softmax gives a soft
                         probability distribution over equivalent drive-
                         strength variants for each gate.
  ``buffering_tensor`` — buffer-insertion logits.  Sigmoid gives the
                         continuous insertion probability per Steiner edge.
  ``cell_xy``          — gate (x, y) positions in DBU.

Loss components returned by ``forward()``:

  tns            — total negative slack (sum of clamped negative slacks)
  wns            — worst negative slack
  switching_power— α·C·V²·f dynamic power
  quant_penalty  — discretization penalty (drives soft→hard decisions)
  density        — RUDY-style overlap penalty grid [GRID×GRID]

Notation used in comments
--------------------------
  G     = number of signal gates
  N     = number of Steiner points (STP) across all nets
  A     = number of Liberty timing arcs
  M     = number of cell masters
  P     = number of unique pin names
  [T,2] = rise/fall pair tensor with T samples
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from placeopt.timing.constants import (
    RISE, FALL, M0, M1, M2, C1, C2, RPI,
    R_PER_LEN, C_PER_LEN, TO_MICRON, GRID_SIZE,
)
from placeopt.timing.graph import STAGraph
from placeopt.timing.schedule import WireTask, GateTask
from placeopt.timing.pi_model import PiModelFunction, ElmoreDelayFunction
from placeopt.timing.lut import bilinear_interp_batch
from placeopt.opt.density import compute_density


# ---------------------------------------------------------------------------
# Segment softmax (stable, per-gate normalisation over sizing logits)
# ---------------------------------------------------------------------------

def _segment_softmax(
    logits: torch.Tensor,    # [total_u]
    gate_id: torch.Tensor,   # [total_u] → gate index
    gate_len: torch.Tensor,  # [G]       → number of variants per gate
) -> torch.Tensor:
    """
    Compute per-gate softmax over ``logits`` using numerically stable
    scatter operations.

    Returns
    -------
    weights : [total_u]  ∈ (0, 1), sums to 1 within each gate's block.
    """
    G = gate_len.shape[0]
    # Max for numerical stability.
    gate_max = torch.full((G,), float("-inf"), device=logits.device, dtype=logits.dtype)
    gate_max.scatter_reduce_(0, gate_id, logits, reduce="amax", include_self=False)
    shifted = logits - gate_max[gate_id]
    ex = torch.exp(shifted)
    gate_sum = torch.zeros(G, device=logits.device, dtype=logits.dtype)
    gate_sum.scatter_add_(0, gate_id, ex)
    return ex / gate_sum[gate_id]


# ---------------------------------------------------------------------------
# Gate arc computation (custom autograd for speed)
# ---------------------------------------------------------------------------

class GateGroupedForward(torch.autograd.Function):
    """
    Batched gate-arc computation: input (arrival, slew, Π params) →
    output (arrival, slew) for all timing arcs in one shot.

    The output [T, 4] packs [arr_rise, arr_fall, slew_rise, slew_fall].
    Grouping (by gate × master) allows per-master outputs to be summed
    with their soft-selection weights.
    """

    @staticmethod
    def forward(
        ctx,
        in_arr,      # [T, 2]
        in_slew,     # [T, 2]
        c1,          # [T]
        c2,          # [T]
        rpi,         # [T]
        arc_idx_r,   # [T]  rise-output arc index
        arc_idx_f,   # [T]  fall-output arc index
        group,       # [T]  group index
        num_groups: int,
        unateness,   # [A]
        delay_table, # [A, 7, 7]
        slew_table,  # [A, 7, 7]
        load_index,  # [A, 7]
        slew_index,  # [A, 7]
        use_ceff: int,
        max_cap: float,
    ):
        scale = 1e15
        T = arc_idx_r.shape[0]

        # Interleave arc indices for rise and fall outputs.
        arc_flat = torch.stack([arc_idx_r, arc_idx_f], dim=1).reshape(-1)  # [2T]

        # Determine which input edge (rise vs fall) feeds each output.
        una = unateness[arc_flat]                    # [2T]
        is_fall = torch.zeros(2 * T, dtype=una.dtype, device=una.device)
        is_fall[1::2] = 1                            # fall outputs at odd indices
        rf_idx = (una ^ is_fall).view(T, 2)         # [T, 2]: which input RF to use

        in_slew_sel = in_slew.gather(1, rf_idx).reshape(-1)  # [2T]
        in_arr_sel  = in_arr.gather(1, rf_idx).reshape(-1)   # [2T]

        dtab = delay_table[arc_flat]   # [2T, 7, 7]
        stab = slew_table[arc_flat]    # [2T, 7, 7]
        axc  = load_index[arc_flat]    # [2T, 7]
        axs  = slew_index[arc_flat]    # [2T, 7]

        if use_ceff:
            # Effective capacitance: solve Ceff ≈ C₁ + C₂·correction
            c1_f  = c1.repeat_interleave(2) / scale
            c2_f  = c2.repeat_interleave(2) / scale
            rpi_f = rpi.repeat_interleave(2)
            load  = _compute_ceff_batch(in_slew_sel, c1_f, c2_f, rpi_f, dtab, stab, axs, axc)
        else:
            load = (c1.repeat_interleave(2) + c2.repeat_interleave(2)) / scale

        load = load.clamp_max(max_cap)

        arc_delay, arc_slew = _bilinear_gate(in_slew_sel, load, dtab, stab, axs, axc)

        arr_out  = (in_arr_sel + arc_delay).view(T, 2)
        slew_out = arc_slew.view(T, 2)
        return torch.cat([arr_out, slew_out], dim=1)  # [T, 4]

    @staticmethod
    def backward(ctx, grad):
        return (None,) * 16


# ---------------------------------------------------------------------------
# Low-level gate kernel helpers
# ---------------------------------------------------------------------------

def _bilinear_gate(
    in_slew: torch.Tensor,  # [M]
    load:    torch.Tensor,  # [M]
    dtab:    torch.Tensor,  # [M, 7, 7]
    stab:    torch.Tensor,  # [M, 7, 7]
    axs:     torch.Tensor,  # [M, 7]
    axc:     torch.Tensor,  # [M, 7]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinear interpolation into per-arc LUTs (batched)."""
    from placeopt.timing.lut import _axis_interval
    M = in_slew.shape[0]
    ar = torch.arange(M, device=in_slew.device)

    i0, a = _axis_interval(axs, in_slew)
    j0, b = _axis_interval(axc, load)
    i1, j1 = i0 + 1, j0 + 1

    def _blerp(t):
        return ((1-a)*(1-b)*t[ar,i0,j0] + (1-a)*b*t[ar,i0,j1]
                + a*(1-b)*t[ar,i1,j0] + a*b*t[ar,i1,j1])

    return _blerp(dtab), _blerp(stab)


def _compute_ceff_batch(
    in_slew: torch.Tensor,
    c1: torch.Tensor, c2: torch.Tensor, rpi: torch.Tensor,
    dtab, stab, axs, axc,
    max_iter: int = 3,
) -> torch.Tensor:
    """
    Iterative Ceff computation (matches OpenSTA's Ceff algorithm).

    Ceff is found by iterating:
        d(k+1), s(k+1) = LUT(in_slew, Ceff(k))
        tau_out = Rd · Ceff
        Ceff    = C₁ + C₂ · (1 − exp(−2·tau_out / in_slew)) · in_slew / (2·tau_out)
    """
    # Initial guess: lumped capacitance.
    ceff = (c1 + c2).clamp_min(1e-30)
    for _ in range(max_iter):
        d, _ = _bilinear_gate(in_slew, ceff, dtab, stab, axs, axc)
        tau  = d.clamp_min(1e-30)
        ratio = (2.0 * tau / in_slew.clamp_min(1e-30)).clamp_max(10.0)
        h    = torch.where(ratio > 0.01,
                           (1.0 - torch.exp(-ratio)) / ratio,
                           1.0 - ratio / 2.0)
        ceff = (c1 + c2 * h).clamp_min(1e-30)
    return ceff


# ---------------------------------------------------------------------------
# TimingEngine
# ---------------------------------------------------------------------------

class TimingEngine(nn.Module):
    """
    Differentiable static timing analysis engine.

    Parameters
    ----------
    graph_data : STAGraph
        All pre-built graph tensors (returned by ``STAGraphBuilder.build``).

    Notes
    -----
    The three optimizable ``nn.Parameter`` objects (``U_flat``,
    ``buffering_tensor``, ``cell_xy``) are referenced directly from
    ``graph_data`` so that they can be handed to separate optimizers.
    """

    def __init__(self, graph_data: STAGraph) -> None:
        super().__init__()
        self.gd = graph_data                         # short alias

        # Register parameters so they appear in .parameters() and move with .to()
        self.U_flat           = graph_data.U_flat
        self.buffering_tensor = graph_data.buffering_tensor
        self.cell_xy          = graph_data.cell_xy

        N = len(graph_data.stp_parent_idx)
        A = len(graph_data.stp_parent_idx)

        # Mutable state tensors (arrival/slew/moment/pi) are NOT nn.Parameters.
        device = graph_data.cell_xy.device
        self.register_buffer("arrival", torch.zeros(N, 2, device=device))
        self.register_buffer("slew",    torch.zeros(N, 2, device=device))
        self.register_buffer("moment",  torch.zeros(N, 3, device=device))
        self.register_buffer("pi",      torch.zeros(N, 3, device=device))

        # Whether to use the buffer-regression wire model.
        self.use_buf_regression: bool = True

    # ------------------------------------------------------------------
    # High-level forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        wire_mode: str = "simple",
        gate_mode: str = "simple",
        use_buf_regression: Optional[bool] = None,
    ):
        """
        Run the complete differentiable STA forward pass.

        Parameters
        ----------
        wire_mode : "simple" | "dmp" | "ceff" | "dmp_ceff"
            Wire delay model.  "simple" uses Elmore with a ramp approximation.
            "dmp" uses the Newton-based ramp-crossing solver.
        gate_mode : "simple" | "ceff"
            Gate delay model.  "ceff" enables iterative effective-capacitance.
        use_buf_regression : override the instance-level flag.

        Returns
        -------
        tns, wns, switching_power, quant_penalty, density
        """
        if use_buf_regression is not None:
            self.use_buf_regression = bool(use_buf_regression)

        gd = self.gd

        # Soft gate-size weights: segment-softmax of U_flat.
        w = _segment_softmax(self.U_flat, gd.u_gate_id, gd.u_gate_len)
        # Soft buffer-insertion probability: sigmoid of buffering_tensor.
        b = torch.sigmoid(self.buffering_tensor)

        self._forward_positions(w)
        self._compute_segment_rc()
        self._forward_moments(w, b)       # backward: sinks → driver
        self._forward_pi_model(b)         # driver output Π model
        self._forward_arrival_slew(w, b, wire_mode, gate_mode)  # forward: driver → sinks
        tns, wns = self._forward_slack()
        power    = self._forward_switching_power(b)
        penalty  = self._forward_quantization_penalty(w, b)
        density  = compute_density(GRID_SIZE, gd, b)

        return tns, wns, power, penalty, density

    # ------------------------------------------------------------------
    # Sub-passes
    # ------------------------------------------------------------------

    def _forward_positions(self, w: torch.Tensor) -> None:
        """Update ``pos_xy`` and ``cell_box`` from current ``cell_xy``."""
        gd = self.gd

        # Compute each STP's absolute position from gate origin + pin offset.
        xy = gd.stp_fixed_xy.clone()
        if gd.pos_entry_u_index.numel():
            # For gate-pin STPs: position = gate_origin + softmax(w) · offset.
            w_per_entry = w[gd.pos_entry_u_index]          # [E]
            gate_per_entry = gd.stp_gate_mapping[gd.pos_entry_stp_idx]  # [E]
            base_xy = self.cell_xy[gate_per_entry]          # [E, 2]
            offset  = gd.stp_offset[gd.pos_entry_stp_idx]  # [E, 2]
            contrib = (base_xy + offset) * w_per_entry.unsqueeze(1)
            xy.scatter_add_(0, gd.pos_entry_stp_idx.unsqueeze(1).expand(-1, 2), contrib)

        gd.pos_xy = xy

        # Update cell bounding box for density computation.
        G = self.cell_xy.shape[0]
        w_flat = w
        gate_w = torch.zeros(G, device=w.device)
        gate_h = torch.zeros(G, device=w.device)
        mw = gd.master_w[gd.u_master_id]
        mh = gd.master_h[gd.u_master_id]
        gate_w.scatter_add_(0, gd.u_gate_id, w_flat * mw)
        gate_h.scatter_add_(0, gd.u_gate_id, w_flat * mh)
        gd.cell_box = torch.cat([self.cell_xy, torch.stack([gate_w, gate_h], dim=1)], dim=1)

    def _compute_segment_rc(self) -> None:
        """
        Compute per-segment wire resistance and capacitance.

        The values are stored back into ``self.seg_r`` and ``self.seg_c``
        and are used by the moment-propagation pass.  Note that segment RC
        is a function of positions only (no gradient needed for these
        intermediate quantities).
        """
        gd = self.gd
        pos = gd.pos_xy
        par = gd.stp_parent_idx
        seg_len = torch.abs(pos - pos[par]).sum(dim=1) * TO_MICRON  # [N] µm
        self.seg_r = R_PER_LEN * seg_len   # [N]
        self.seg_c = C_PER_LEN * seg_len   # [N]

    def _forward_moments(self, w: torch.Tensor, b: torch.Tensor) -> None:
        """
        Backward tree traversal: propagate Elmore moments from sinks to driver.

        M₀ (capacitance), M₁, M₂ are accumulated level by level in reverse
        topological order using the pre-built ``level_edges`` schedule.
        """
        gd = self.gd
        scale = 1e15

        # Initialise moments at each node with its own gate-pin capacitance
        # (weighted by the soft sizing distribution).
        # Load-pin capacitance: master_pin_cap[master, pin_id] × w.
        master_per_u = gd.u_master_id                    # [total_u]
        pin_id_per_entry = gd.pin_name_id[gd.pin_entry_pin_id]  # [total_entry]
        cap_per_entry = (gd.master_pin_cap[master_per_u[gd.pin_entry_u_index],
                                           pin_id_per_entry]
                         * w[gd.pin_entry_u_index])      # [total_entry]

        N = self.moment.shape[0]
        moment = torch.zeros(N, 3, device=gd.U_flat.device)
        # Add self-capacitance contribution to M₀.
        moment[:, M0].scatter_add_(0, gd.inpin_idx, cap_per_entry * scale)

        # Also add wire capacitance (half-segment at each node).
        moment[:, M0].add_(self.seg_c * scale)

        # Backward levels: propagate moments upstream.
        for (p_idx, c_idx) in reversed(gd.level_edges):
            if p_idx.numel() == 0:
                continue
            r_c = self.seg_r[c_idx]                    # resistance of edge p→c
            m0_c = moment[c_idx, M0]
            m1_c = moment[c_idx, M1]
            m2_c = moment[c_idx, M2]

            # Elmore moment accumulation:
            #   M0_p += M0_c
            #   M1_p += M1_c + r·M0_c²
            #   M2_p += M2_c + 2r·M0_c·M1_c + r²·M0_c³
            moment[:, M0].scatter_add_(0, p_idx, m0_c)
            moment[:, M1].scatter_add_(0, p_idx, m1_c + r_c * m0_c ** 2)
            moment[:, M2].scatter_add_(0, p_idx, m2_c + 2.0 * r_c * m0_c * m1_c
                                       + r_c ** 2 * m0_c ** 3)

            # Blend in buffer load when b > 0.
            buf_load = torch.tensor(gd.buf_in_cap, device=b.device) * scale
            b_c = b[c_idx]
            moment[:, M0].scatter_add_(0, p_idx, b_c * buf_load)

        self.moment = moment

    def _forward_pi_model(self, b: torch.Tensor) -> None:
        """
        Compute the two-pole Π model at each driver pin from its moments.
        """
        gd = self.gd
        src = gd.pi_source_idxs
        if src.numel() == 0:
            return
        c1, c2, rpi = PiModelFunction.apply(self.moment[src])
        self.pi = self.pi.clone()
        self.pi[src, C1]  = c1
        self.pi[src, C2]  = c2
        self.pi[src, RPI] = rpi

    def _forward_arrival_slew(
        self,
        w: torch.Tensor,
        b: torch.Tensor,
        wire_mode: str,
        gate_mode: str,
    ) -> None:
        """
        Forward tree traversal: propagate arrival times and slews from
        primary inputs / FF outputs toward primary outputs / FF inputs.
        """
        gd = self.gd
        self.arrival = self.arrival.detach().zero_()
        self.slew    = self.slew.detach().zero_()

        # Seed start points from OpenSTA-computed initial conditions.
        with torch.no_grad():
            self.arrival.index_copy_(0, gd.start_idx, gd.start_arrival_init)
            self.slew.index_copy_(   0, gd.start_idx, gd.start_slew_init)

        use_ceff = gate_mode == "ceff"
        use_dmp  = "dmp" in wire_mode
        use_buf  = self.use_buf_regression

        for task in gd.execution_plan[2:]:   # skip start-point levels
            if task.type == "WIRE":
                self._wire_step(task, b, use_dmp, use_buf)
            elif task.type == "GATE":
                self._gate_step(task, w, use_ceff)

    def _wire_step(self, task: WireTask, b: torch.Tensor, use_dmp: bool, use_buf: bool) -> None:
        gd = self.gd
        p, c = task.p_tensor, task.c_tensor
        if p.numel() == 0:
            return

        pos = gd.pos_xy
        sp_xy = pos[gd.stp_parent_idx[c]]
        ep_xy = pos[c]

        elmore = self.seg_r[c] * self.moment[c, M0] / 1e15  # Elmore tau

        if use_dmp:
            from placeopt.timing.wire import wire_delay_slew_dmp
            dr, sr = wire_delay_slew_dmp(elmore, self.slew[p, RISE])
            df, sf = wire_delay_slew_dmp(elmore, self.slew[p, FALL])
        else:
            dr, sr, df, sf = ElmoreDelayFunction.apply(
                elmore, self.slew[p, RISE], self.slew[p, FALL])

        arr_r = self.arrival[p, RISE] + dr
        arr_f = self.arrival[p, FALL] + df

        if use_buf:
            from placeopt.timing.wire import conti_buf_delay_slew
            b_c = b[c]
            d_buf_r, s_buf_r = conti_buf_delay_slew(
                sp_xy, ep_xy, gd.xcen_buf, gd.xin_buf, gd.buf_in_cap,
                self.slew[p, RISE], self.moment[c, M0] / 1e15, b_c,
                gd.fs_sr, gd.fs_lr, gd.fd_sr, gd.fd_lr)
            d_buf_f, s_buf_f = conti_buf_delay_slew(
                sp_xy, ep_xy, gd.xcen_buf, gd.xin_buf, gd.buf_in_cap,
                self.slew[p, FALL], self.moment[c, M0] / 1e15, b_c,
                gd.fs_sf, gd.fs_lf, gd.fd_sf, gd.fd_lf)
            alpha = b_c.clamp(0.0, 1.0)
            arr_r = (1 - alpha) * arr_r  + alpha * (self.arrival[p, RISE] + d_buf_r)
            arr_f = (1 - alpha) * arr_f  + alpha * (self.arrival[p, FALL] + d_buf_f)
            sr    = (1 - alpha) * sr + alpha * s_buf_r
            sf    = (1 - alpha) * sf + alpha * s_buf_f

        self.arrival[c, RISE] = arr_r
        self.arrival[c, FALL] = arr_f
        self.slew[c, RISE]    = sr
        self.slew[c, FALL]    = sf

    def _gate_step(self, task: GateTask, w: torch.Tensor, use_ceff: bool) -> None:
        gd = self.gd
        p_e, c_e = task.p_e, task.c_e
        grp       = task.group_tensor
        c_grp     = task.c_group
        u_grp     = task.u_index_group
        if p_e.numel() == 0:
            return

        pi_e  = self.pi[c_e]

        u_e   = u_grp[grp]
        mid_e = gd.u_master_id[u_e]
        from_id = gd.stp_pin_id[p_e]
        to_id   = gd.stp_pin_id[c_e]
        arc_r = gd.arc_lut_r[mid_e, from_id, to_id]
        arc_f = gd.arc_lut_f[mid_e, from_id, to_id]

        grouped_out = GateGroupedForward.apply(
            self.arrival[p_e], self.slew[p_e],
            pi_e[:, C1], pi_e[:, C2], pi_e[:, RPI],
            arc_r, arc_f,
            grp, c_grp.numel(),
            gd.unateness, gd.delay_table, gd.slew_table,
            gd.load_index, gd.slew_index,
            int(use_ceff), 5e11,
        )

        w_grp    = w[u_grp]                             # [Group]
        weighted = grouped_out * w_grp.unsqueeze(1)     # [Group, 4]
        c_grp2   = c_grp.unsqueeze(1).expand(-1, 2)    # [Group, 2]
        self.arrival.scatter_add_(0, c_grp2, weighted[:, :2])
        self.slew.scatter_add_(   0, c_grp2, weighted[:, 2:])

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------

    def _forward_slack(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute TNS and WNS from endpoint arrivals vs required times.

        Slack = required − arrival (positive means timing is met).
        TNS   = sum of all negative slacks.
        WNS   = most negative slack (worst case).

        A log-sum-exp softmin is used instead of hard min to maintain
        differentiability across rise/fall edges.
        """
        gd = self.gd
        arrival_e = self.arrival[gd.end_idx]   # [E, 2]
        slack_rf  = gd.require - arrival_e      # [E, 2]  (rise, fall)

        # Smooth minimum over rise/fall using log-sum-exp.
        m    = slack_rf.min(dim=1, keepdim=True).values
        tau  = 1e-11
        z    = (m - slack_rf) / tau
        slack = m.squeeze(1) - tau * torch.logsumexp(z, dim=1)

        negative = slack.clamp(max=0.0)
        tns = negative.sum()
        wns = -(-slack).max()
        return tns, wns

    def _forward_switching_power(self, b: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic switching power:  P = Σ α · C · V² · f.

        C is the load capacitance at each output pin (M₀ moment),
        V is normalised to 1, f is modelled as the activity density.
        """
        gd = self.gd
        scale = 1e15
        out_load = self.moment[gd.outpin_idx, M0] / scale     # [P_out]
        activity = gd.outpin_activity                          # [P_out]
        power = (activity * out_load).sum()
        return power

    def _forward_quantization_penalty(
        self, w: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalty that encourages discrete decisions:

        * Gate sizing: penalise deviation from a one-hot distribution.
          Loss = Σ  −(max(w) − mean_of_others) / gate  (wants sharp peaks).

        * Buffer insertion: penalise soft values (prefers 0 or 1).
          Loss = Σ  4 · b · (1 − b)   (= 0 at extremes, 1 at 0.5).

        Returns a scalar to be added to the total loss.
        """
        gd = self.gd

        # Gate: entropy-like penalty on weight distribution.
        gate_max = torch.full((gd.u_gate_len.numel(),), float("-inf"),
                              device=w.device)
        gate_max.scatter_reduce_(0, gd.u_gate_id, w, reduce="amax", include_self=False)
        gate_sum = torch.zeros_like(gate_max)
        gate_sum.scatter_add_(0, gd.u_gate_id, w)
        gate_cnt = gd.u_gate_len.float()
        gate_penalty = -(gate_max - (gate_sum - gate_max) / (gate_cnt - 1).clamp_min(1.0))
        q_gate = gate_penalty.sum()

        # Buffer: binary-entropy penalty.
        q_buf = (4.0 * b * (1.0 - b)).sum()

        return q_gate + q_buf

    # ------------------------------------------------------------------
    # Discretise (call before writing back to OpenROAD)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def discretize(
        self,
        gate_threshold: float = 0.5,
        buf_threshold: float = 0.5,
    ) -> None:
        """
        Snap soft decisions to hard values:

        * Gates: select the master with the highest softmax weight (argmax),
          provided its weight exceeds ``gate_threshold``.
        * Buffers: round sigmoid(logit) to 0 or 1 if it is more than
          ``buf_threshold`` away from 0.5.

        After this call, ``forward()`` will behave identically to a discrete
        assignment.
        """
        gd = self.gd
        u, gid, glen = self.U_flat, gd.u_gate_id, gd.u_gate_len
        G = int(glen.numel())
        maxlen = int(glen.max().item())

        pos = torch.arange(u.numel(), device=u.device)
        local = pos - gd.u_gate_off[gid]

        mat = torch.full((G, maxlen), float("-inf"), device=u.device)
        mat[gid, local] = u
        argmax = mat.argmax(dim=1)

        w = _segment_softmax(u, gid, glen)
        gate_top = torch.full((G,), float("-inf"), device=u.device)
        gate_top.scatter_reduce_(0, gid, w, reduce="amax", include_self=False)
        hard_gate = gate_top > gate_threshold

        pick = local == argmax[gid]
        u_hard = torch.where(pick, torch.ones_like(u), torch.full_like(u, -1000.0))
        self.U_flat.copy_(torch.where(hard_gate[gid], u_hard, u))

        b = torch.sigmoid(self.buffering_tensor)
        lo = b <= buf_threshold
        hi = b >= 1.0 - buf_threshold
        hard_val = torch.where(hi, torch.full_like(b, 100.0), torch.full_like(b, -100.0))
        self.buffering_tensor.copy_(torch.where(lo | hi, hard_val, self.buffering_tensor))
