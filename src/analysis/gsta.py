# Gradient-based Static Timing Analysis engine (GSTA)
from random import randint
import time
import gc

import openroad as ord
from openroad import Tech, Design, Timing
from pathlib import Path
from torch.utils.checkpoint import checkpoint

from src.util.helpers import isDriverPin, is_circuit_output

from src.sta.params import *
from src.sta.arc_model import *
from src.sta.place_db import *

import odb
import torch.nn as nn
import torch

import os
import math
import random

from src.sta.charge_field import compute_charge_density


class GSTA(nn.Module):
    """Gradient-based differentiable timing analysis engine."""

    def __init__(self, design, timing):
        super(GSTA, self).__init__()
        self.design = design
        self.timing = timing
        self._corr_plot_done = False
        self.use_buf_regression = True
        self.disp_baseline_avg = None

        self.circuitLib = None
        self.stt_mgr = None

        self.buf_mode = "single"  # single/multi

    def getTimingGraph(self, graph):
        self.timingGraph = graph

    def setLibrary(self, circuitLib):
        self.circuitLib = circuitLib

    def setSTPNetwork(self, stt_mgr):
        self.stt_mgr = stt_mgr

    def release(self):
        self.graph_data = None
        self.U_flat = None
        self.buffering_tensor = None
        self.cell_xy = None
        self.position = None
        self.arrival = None
        self.slew = None
        self.moment = None
        self.pi = None
        self.RUDY = None
        self.Density = None
        self.Potential = None
        self.circuitLib = None
        self.stt_mgr = None
        self.timingGraph = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def tensor_init(self, origin_gate_weights=6.0, origin_buffer_weights=-8.0, device=torch.device("cpu")):
        builder = PlaceDBFactory()
        self.graph_data = builder.build(origin_gate_weights, origin_buffer_weights, self.stt_mgr, self.circuitLib, self.timing, device)

        self.U_flat = self.graph_data.U_flat
        self.buffering_tensor = self.graph_data.buffering_tensor
        self.cell_xy = self.graph_data.cell_xy
        self.disp_baseline_avg = None

        N = len(self.stt_mgr.stpList)
        self.position = torch.zeros((N, 2), dtype=torch.float32, device=device)
        self.arrival = torch.zeros((N, 2), dtype=torch.float32, device=device)
        self.slew = torch.zeros((N, 2), dtype=torch.float32, device=device)
        self.moment = torch.zeros((N, 3), dtype=torch.float32, device=device)
        self.pi = torch.zeros((N, 3), dtype=torch.float32, device=device)

        self.RUDY = torch.zeros((DENSITY_BINS, DENSITY_BINS), dtype=torch.float32, device=device)
        self.Density = torch.zeros((DENSITY_BINS, DENSITY_BINS), dtype=torch.float32, device=device)
        self.Potential = torch.zeros((DENSITY_BINS, DENSITY_BINS), dtype=torch.float32, device=device)

    def forward_quantified_panelty(self, w_flat, b_flat):
        device = w_flat.device
        dtype = w_flat.dtype

        buf_pen = (4.0 * b_flat * (1.0 - b_flat)).mean()

        gid = self.graph_data.u_gate_id
        gate_len = self.graph_data.u_gate_len
        G = int(gate_len.numel())
        sumsq = torch.zeros((G,), device=device, dtype=dtype)
        sumsq.scatter_add_(0, gid, w_flat * w_flat)

        gate_len_f = gate_len
        denom = 1.0 - 1.0 / gate_len_f
        denom = torch.maximum(denom, torch.ones_like(denom) * 1e-12)
        gate_pen_per = (1.0 - sumsq) / denom
        gate_pen_per = torch.where(gate_len > 1, gate_pen_per, torch.zeros_like(gate_pen_per))
        gate_pen = gate_pen_per.mean()

        penalty = gate_pen + buf_pen
        return penalty

    def forward_movement_displacement(self):
        init_idx = getattr(self.graph_data, "init_cell_idx", None)
        init_xy = getattr(self.graph_data, "init_cell_xy", None)

        cur_xy = self.cell_xy[init_idx]
        n = min(int(init_xy.shape[0]), int(cur_xy.shape[0]))
        init_xy = init_xy[:n]
        cur_xy = cur_xy[:n]

        diff = cur_xy - init_xy
        dist = diff.abs().sum(dim=1)
        disp_avg = dist.mean()
        disp_max = dist.max()

        return disp_avg, disp_max

    def forward_moment(self, w_flat, b_flat):
        device = w_flat.device
        gd = self.graph_data
        pos = gd.pos_xy
        parent = gd.stp_parent_idx

        u_idx  = gd.pin_entry_u_index
        pin_id = gd.pin_entry_pin_id
        pin_name_id = gd.pin_name_id
        master_pin_cap = gd.master_pin_cap
        u_master_id = gd.u_master_id
        pin_name_id_e = pin_name_id[pin_id]
        cap_e = master_pin_cap[u_master_id[u_idx], pin_name_id_e]
        P = int(gd.inpin_idx.numel())

        contrib = w_flat[u_idx] * cap_e
        pin_cap = torch.zeros((P,), device=device, dtype=w_flat.dtype)
        pin_cap.scatter_add_(0, pin_id, contrib)

        self.moment.detach_()
        self.moment.zero_()

        scale = 1e15
        r_via = 15.85

        stp_idx = gd.inpin_idx
        self.moment[:, MOM0].index_add_(0, stp_idx, pin_cap * scale)

        _, child_idxs = gd.level_edges[0]
        self.moment[child_idxs, MOM1] = -r_via * (self.moment[child_idxs, MOM0] ** 2)
        self.moment[child_idxs, MOM2] = r_via**2 * (self.moment[child_idxs, MOM0] ** 3)

        buf_in_cap = gd.buf_in_cap * 1e15
        for parent_idxs, child_idxs in self.graph_data.level_edges[:]:
            if parent_idxs.numel() == 0:
                continue

            m0_c = self.moment[child_idxs, MOM0] + 0.5 * WIRE_C * self.graph_data.len[child_idxs] * scale
            m1_c = self.moment[child_idxs, MOM1]
            m2_c = self.moment[child_idxs, MOM2]
            r = WIRE_R * self.graph_data.len[child_idxs]

            contrib_wire_m0 = m0_c + 0.5 * WIRE_C * self.graph_data.len[child_idxs] * scale
            contrib_wire_m1 = m1_c - r * (m0_c ** 2)
            contrib_wire_m2 = m2_c - 2.0 * r * (m0_c * m1_c) + (r ** 2) * (m0_c ** 3)

            half_wire = half_span_wire(pos[parent_idxs], pos[child_idxs],
                                  self.graph_data.xcen_buf, self.graph_data.xin_buf)
            buf_m0 = (self.graph_data.buf_in_cap + 0.5 * half_wire * WIRE_C) * scale
            buf_m1 = -r_via * ((self.graph_data.buf_in_cap * scale) ** 2)
            buf_m2 = r_via**2 * ((self.graph_data.buf_in_cap * scale) ** 3)
            r_half_wire = half_wire * WIRE_R
            c_half_wire = half_wire * WIRE_C
            contrib_buf_m0 = buf_m0 + c_half_wire * 0.5 * scale
            contrib_buf_m1 = buf_m1 - r_half_wire * ((buf_m0) ** 2)
            contrib_buf_m2 = buf_m2 - 2.0 * r_half_wire * (buf_m0 * buf_m1) + (r_half_wire ** 2) * (buf_m0 ** 3)

            contrib_m0 = (1 - b_flat[child_idxs]) * contrib_wire_m0 + b_flat[child_idxs] * contrib_buf_m0
            contrib_m1 = (1 - b_flat[child_idxs]) * contrib_wire_m1 + b_flat[child_idxs] * contrib_buf_m1
            contrib_m2 = (1 - b_flat[child_idxs]) * contrib_wire_m2 + b_flat[child_idxs] * contrib_buf_m2

            self.moment[:, 0].index_add_(0, parent_idxs, contrib_m0)
            self.moment[:, 1].index_add_(0, parent_idxs, contrib_m1)
            self.moment[:, 2].index_add_(0, parent_idxs, contrib_m2)

        self.moment[:, MOM0].add_(WIRE_C * self.graph_data.len, alpha=0.5 * scale)
        return self.moment

    def forward_pi_model(self, b_flat):
        scale = 1e15
        gd = self.graph_data
        pos = gd.pos_xy
        parent = gd.stp_parent_idx
        pi_internal_idxs = gd.pi_internal_idxs
        pi_source_idxs = gd.pi_source_idxs
        sp_xy = pos[parent[pi_internal_idxs]]
        ep_xy = pos[pi_internal_idxs]
        buf_var = b_flat[pi_internal_idxs]
        xcen_buf = gd.xcen_buf
        xin_buf = gd.xin_buf
        pi = self.pi
        seg_len = gd.len

        pi.detach_()
        pi.zero_()
        c1, c2, rpi = PiNetworkGrad.apply(self.moment[pi_source_idxs])
        pi[pi_source_idxs] = torch.stack([c1, c2, rpi], dim=1)

        moment_net = self.moment[pi_internal_idxs]
        ln = half_span_wire(sp_xy, ep_xy, xcen_buf, xin_buf)
        R_ln = ln * WIRE_R
        C_ln = ln * WIRE_C

        m0_mod = moment_net[:, MOM0] + (0.5 * C_ln - 0.5 * WIRE_C * seg_len[pi_internal_idxs]) * scale
        moment_stp = torch.stack([m0_mod, moment_net[:, MOM1], moment_net[:, MOM2]], dim=1)

        m0_buf = moment_stp[:, MOM0] + 0.5 * C_ln * scale
        m1_buf = moment_stp[:, MOM1] - R_ln * (moment_stp[:, MOM0] ** 2)
        m2_buf = moment_stp[:, MOM2] - 2.0 * R_ln * (moment_stp[:, MOM0] * moment_stp[:, MOM1]) \
                               + (R_ln ** 2) * (moment_stp[:, MOM0] ** 3)
        moment_buf_stp = torch.stack([m0_buf, m1_buf, m2_buf], dim=1)

        c1, c2, rpi = PiNetworkGrad.apply(moment_buf_stp)
        pi[pi_internal_idxs] = torch.stack([c1, c2, rpi], dim=1)

    def compute_segment_res_caps(self):
        pos = self.graph_data.pos_xy
        parent = self.graph_data.stp_parent_idx
        diff = torch.abs(pos - pos[parent])
        self.graph_data.len = diff.sum(dim=1) * DBU_NM

    def forward_position(self, w_flat):
        gd = self.graph_data
        fixed_xy = gd.stp_fixed_xy

        if gd.pos_entry_u_index.numel() == 0:
            xy = fixed_xy
        else:
            pos_u = gd.pos_entry_u_index
            pos_idx = gd.pos_entry_stp_idx
            gate_idx = gd.stp_gate_mapping[pos_idx]
            master_id = gd.u_master_id[pos_u]
            w = gd.master_w[master_id]
            h = gd.master_h[master_id]
            orient = gd.gate_orient[gate_idx]
            is_single = gd.u_gate_len[gate_idx] == 1
            center_x = w * 0.5
            center_y = h * 0.5
            base_x = center_x
            base_y = center_y
            off_x = base_x
            off_y = base_y

            fallback = gd.stp_offset[pos_idx]
            pos_off_center = torch.stack([off_x, off_y], dim=1)
            pos_off = torch.where(is_single.unsqueeze(1), fallback, pos_off_center)

            pin_offset = torch.zeros_like(fixed_xy)
            contrib = w_flat[pos_u].unsqueeze(1) * pos_off
            pin_offset.index_add_(0, pos_idx, contrib)

            gate_idx = gd.stp_gate_mapping
            mask = gate_idx >= 0
            gate_xy = torch.zeros_like(fixed_xy)
            gate_xy[mask] = self.cell_xy[gate_idx[mask]]
            xy = torch.where(mask.unsqueeze(1), gate_xy + pin_offset, fixed_xy)

        G = int(self.cell_xy.shape[0])
        gate_w = torch.zeros((G,), device=w_flat.device, dtype=w_flat.dtype)
        gate_h = torch.zeros((G,), device=w_flat.device, dtype=w_flat.dtype)
        master_w_u = gd.master_w[gd.u_master_id]
        master_h_u = gd.master_h[gd.u_master_id]
        gate_w.scatter_add_(0, gd.u_gate_id, w_flat * master_w_u)
        gate_h.scatter_add_(0, gd.u_gate_id, w_flat * master_h_u)
        gate_wh = torch.stack([gate_w, gate_h], dim=1)

        gd.pos_xy = xy
        gd.cell_box = torch.cat([self.cell_xy, gate_wh], dim=1)

    def forward(self, wire_mode="simple", gate_mode="simple", use_buf_regression=None):
        if use_buf_regression is not None:
            self.use_buf_regression = bool(use_buf_regression)

        w_flat = _segment_softmax(
            self.U_flat,
            self.graph_data.u_gate_id,
            self.graph_data.u_gate_len,
        )
        b_flat = torch.sigmoid(self.buffering_tensor)

        self.forward_position(w_flat)
        self.compute_segment_res_caps()
        self.forward_moment(w_flat, b_flat)
        self.forward_pi_model(b_flat)
        self.forward_arrival_slew(w_flat, b_flat, wire_mode, gate_mode)

        tns, wns = self.forward_slack()
        switching_power = self.forward_switching_power(b_flat)
        leakage_power = self.forward_leakage_power(w_flat)
        quantified_penalty = self.forward_quantified_panelty(w_flat, b_flat)
        Density = compute_charge_density(DENSITY_BINS, self.graph_data, b_flat)

        return tns, wns, switching_power, leakage_power, quantified_penalty, Density

    def forward_arrival_slew(self, w_flat, b_flat, wire_mode, gate_mode):
        self.arrival.detach_()
        self.slew.detach_()
        self.arrival.zero_()
        self.slew.zero_()

        with torch.no_grad():
            idxs = self.graph_data.start_idx
            self.arrival.index_copy_(0, idxs, self.graph_data.start_arrival_init)
            self.slew.index_copy_(0, idxs, self.graph_data.start_slew_init)

        for level_idx in range(2, len(self.graph_data.execution_plan)):
            data = self.graph_data.execution_plan[level_idx]
            if data.type == "NET":
                self.wire_segment_kernel(data, b_flat, mode=wire_mode)
            elif data.type == "CELL":
                self.gate_kernel(data, w_flat, mode=gate_mode)

    def gate_kernel(self, data, w_flat, mode="simple"):
        p_e = data.p_e
        c_e = data.c_e
        group = data.group_tensor
        c_group = data.c_group
        u_index_group = data.u_index_group

        w_group = w_flat[u_index_group]
        in_slews_e = self.slew[p_e]
        in_arrs_e  = self.arrival[p_e]
        pi_e = self.pi[c_e]
        c1_e = pi_e[:, PI_C1]
        c2_e = pi_e[:, PI_C2]
        rpi_e = pi_e[:, PI_RPI]

        u_index_e = u_index_group[group]
        master_id_e = self.graph_data.u_master_id[u_index_e]
        pin_from_id = self.graph_data.stp_pin_id[p_e]
        pin_to_id = self.graph_data.stp_pin_id[c_e]
        arc_idx_r = self.graph_data.arc_lut_r[master_id_e, pin_from_id, pin_to_id]
        arc_idx_f = self.graph_data.arc_lut_f[master_id_e, pin_from_id, pin_to_id]

        use_ceff = 0 if mode == "simple" else 1
        grouped_out = GateGroupedOutFunction.apply(
            in_arrs_e,
            in_slews_e,
            c1_e,
            c2_e,
            rpi_e,
            arc_idx_r,
            arc_idx_f,
            group,
            c_group.numel(),
            self.graph_data.unateness,
            self.graph_data.delay_table,
            self.graph_data.slew_table,
            self.graph_data.load_index,
            self.graph_data.slew_index,
            use_ceff,
            5e11,
        )

        w_group = w_flat[u_index_group]
        weighted_out = grouped_out * w_group.unsqueeze(1)
        c_group_indices = c_group.unsqueeze(1).expand(-1, 2)
        self.arrival.scatter_add_(0, c_group_indices, weighted_out[:, 0:2])
        self.slew.scatter_add_(0, c_group_indices, weighted_out[:, 2:4])

    def wire_segment_kernel(self, data, b_flat, mode="simple"):
        p = data.p_tensor
        c = data.c_tensor
        gd = self.graph_data
        pos = gd.pos_xy
        parent = gd.stp_parent_idx
        sp_xy_c = pos[parent[c]]
        ep_xy_c = pos[c]
        use_ceff = mode in ("ceff", "dmp_ceff")
        use_dmp = mode in ("dmp", "dmp_ceff")
        use_buf_regression = bool(getattr(self, "use_buf_regression", False))

        arr_out, slew_out = WireSegmentFunction.apply(
            self.arrival[p],
            self.slew[p],
            self.moment[c, MOM0],
            self.pi[c],
            sp_xy_c,
            ep_xy_c,
            b_flat[c],
            gd.xcen_buf,
            gd.xin_buf,
            gd.buf_in_cap,
            gd.buffer_delay_table,
            gd.buffer_slew_table,
            gd.buffer_cap_axis,
            gd.buffer_slew_axis,
            gd.fd_sr,
            gd.fd_lr,
            gd.fd_sf,
            gd.fd_lf,
            gd.fs_sr,
            gd.fs_lr,
            gd.fs_sf,
            gd.fs_lf,
            use_ceff,
            use_dmp,
            use_buf_regression,
        )

        self.arrival[c] = arr_out
        self.slew[c] = slew_out

    def forward_leakage_power(self, w_flat):
        """Differentiable leakage: soft-weighted sum of per-master leakage values."""
        return (w_flat * self.graph_data.leakage_per_u).sum()

    def forward_switching_power(self, b_flat):
        voltage = self.timing.getVoltage()
        scale = 1e15
        out_idx = self.graph_data.outpin_idx
        load_out = self.moment[out_idx, MOM0] / scale - 0.5 * WIRE_C * self.graph_data.len[out_idx]
        power = 0.5 * load_out * self.graph_data.outpin_activity * (voltage ** 2)
        power_buf = 0.5 * (self.moment[:, MOM0] / scale) \
                    * self.graph_data.no_driver_mask * self.graph_data.stp_activity * b_flat * (voltage ** 2)
        return torch.sum(power) + torch.sum(power_buf)

    def forward_slack(self):
        out_arr = self.arrival[self.graph_data.end_idx]
        require = self.graph_data.require
        slack_rf = require - out_arr

        m = slack_rf.min(dim=1, keepdim=True).values
        tau = 1e-11
        z = (m - slack_rf) / tau
        slack = m.squeeze(1) - tau * torch.logsumexp(z, dim=1)
        negative_slacks = torch.clamp(slack, max=0.0)
        tns = torch.sum(negative_slacks)
        flat_slacks = slack.view(-1)
        wns = -torch.max(-1 * flat_slacks, dim=0)[0]
        return tns, wns

    def discretize(self, u_threshold=0.5, buf_threshold=0.5):
        with torch.no_grad():
            u = self.U_flat
            gate_len = self.graph_data.u_gate_len
            if u.numel() > 0 and gate_len.numel() > 0:
                gid = self.graph_data.u_gate_id
                num_gates = int(gate_len.numel())
                max_len = int(gate_len.max().item())

                entry_pos = torch.arange(u.numel(), device=u.device)
                local_pos = entry_pos - self.graph_data.u_gate_off[gid]

                gate_matrix = torch.full((num_gates, max_len), float("-inf"),
                                         device=u.device, dtype=u.dtype)
                gate_matrix[gid, local_pos] = u
                max_local = gate_matrix.argmax(dim=1)
                pick = local_pos == max_local[gid]

                w_flat = _segment_softmax(u, self.graph_data.u_gate_id, self.graph_data.u_gate_len)
                max_w = torch.full((num_gates,), float("-inf"), device=u.device, dtype=u.dtype)
                max_w.scatter_reduce_(0, gid, w_flat, reduce="amax", include_self=False)
                gate_pick = max_w > u_threshold
                gate_pick = gate_pick[gid]

                u_hard = torch.where(pick, torch.ones_like(u), torch.full_like(u, -1000.0))
                u.copy_(torch.where(gate_pick, u_hard, u))

            buf = self.buffering_tensor
            b_flat = torch.sigmoid(buf)
            pos_val = torch.tensor(100.0, device=buf.device, dtype=buf.dtype)
            neg_val = torch.tensor(-100.0, device=buf.device, dtype=buf.dtype)
            hard_mask = (b_flat <= buf_threshold) | (b_flat >= 1.0 - buf_threshold)
            hard_val = torch.where(b_flat >= 1.0 - buf_threshold, pos_val, neg_val)
            buf.copy_(torch.where(hard_mask, hard_val, buf))

    def get_pin_arrival(self, stp_idx, is_rise):
        return self.arrival[stp_idx, 0 if is_rise else 1].item()

    def get_pin_slew(self, stp_idx, is_rise):
        return self.slew[stp_idx, 0 if is_rise else 1].item()

    def get_pin_slack(self, stp_idx, is_rise):
        arrival = self.arrival[stp_idx, 0 if is_rise else 1].item()
        req = self.timing.getPinArrival(
            self.stt_mgr.stpList[stp_idx].Pin.db_ITerm,
            self.timing.Rise if is_rise else self.timing.Fall,
            self.timing.Max,
        )
        return req - arrival

    def get_pin_load(self, stp_idx):
        scale = 1e15
        return (self.moment[stp_idx, MOM0] / scale - 0.5 * WIRE_C * self.graph_data.len[stp_idx]).item()

    def get_pin_delay(self, from_stp_idx, to_stp_idx):
        if self.stt_mgr.stpList[from_stp_idx].Pin is not None and isDriverPin(self.stt_mgr.stpList[from_stp_idx].Pin):
            return
        arrival_to = self.arrival[to_stp_idx]
        arrival_from = self.arrival[from_stp_idx]
        return (arrival_to[0] - arrival_from[0]).item(), (arrival_to[1] - arrival_from[1]).item()

    def getPIModel(self, idxs):
        if isinstance(idxs, torch.Tensor) and idxs.dim() == 0:
            idxs = idxs.view(1)
        return PiNetworkGrad.apply(self.moment[idxs])


def _segment_softmax(u, gate_id, gate_len):
    num_gates = gate_len.size(0)
    gate_max = torch.full((num_gates,), float("-inf"), device=u.device, dtype=u.dtype)
    gate_max.scatter_reduce_(0, gate_id, u, reduce="amax", include_self=False)
    u_max_expanded = gate_max[gate_id]
    ex = torch.exp(u - u_max_expanded)
    gate_sum = torch.zeros((num_gates,), device=u.device, dtype=u.dtype)
    gate_sum.scatter_add_(0, gate_id, ex)
    return ex / gate_sum[gate_id]


def _gate_values_pair(in_arrs, in_slews, c1, c2, rpi, arc_idx_r, arc_idx_f,
                      unateness, delay_table, slew_table, load_index, slew_index, use_ceff):
    scale = 1e15
    T = int(arc_idx_r.numel())

    arc_idx = torch.stack([arc_idx_r, arc_idx_f], dim=1)
    arc_idx_flat = arc_idx.reshape(-1)

    unate = unateness[arc_idx_flat]
    invert = torch.zeros_like(arc_idx_flat, dtype=unate.dtype)
    invert[1::2] = 1
    rf_idx = (unate ^ invert).view(T, 2)

    in_slew = in_slews.gather(1, rf_idx).reshape(-1)
    in_arr  = in_arrs.gather(1, rf_idx).reshape(-1)

    dtab  = delay_table[arc_idx_flat]
    stab  = slew_table[arc_idx_flat]
    axcap = load_index[arc_idx_flat]
    axslw = slew_index[arc_idx_flat]

    c1_f  = c1.repeat_interleave(2)
    c2_f  = c2.repeat_interleave(2)
    rpi_f = rpi.repeat_interleave(2)

    if use_ceff:
        load = batch_effective_load(in_slew, c1_f / scale, c2_f / scale, rpi_f, dtab, stab, axslw, axcap)
    else:
        load = (c1_f + c2_f) / scale

    arc_delay, arc_outslew = eval_arc_timing(in_slew, load, dtab, stab, axslw, axcap)

    in_arr = in_arr.view(T, 2)
    arc_delay = arc_delay.view(T, 2)
    arc_outslew = arc_outslew.view(T, 2)

    return torch.cat([arc_delay + in_arr, arc_outslew], dim=1)


def _grouped_lse(values, group, num_groups, lse_beta):
    values = torch.nan_to_num(values, nan=-1e30, posinf=1e30, neginf=-1e30)
    group_indices = group.unsqueeze(1).expand(-1, values.size(1))

    shift = torch.full((num_groups, values.size(1)), float("-inf"), device=values.device, dtype=values.dtype)
    shift.scatter_reduce_(0, group_indices, values, reduce="amax", include_self=False)
    shift = torch.nan_to_num(shift, neginf=0.0, posinf=0.0)

    shift_expanded = shift.gather(0, group_indices)
    exp_values = torch.exp((values - shift_expanded) * lse_beta)

    sum_out = torch.zeros_like(shift)
    sum_out.scatter_add_(0, group_indices, exp_values)
    return shift + torch.log(sum_out.clamp_min(1e-30)) / lse_beta


def _wire_segment_outputs(arrival_p, slew_p, moment_m0, pi_c, sp_xy_c, ep_xy_c, b_flat_c,
                          xcen_buf, xin_buf, buf_in_cap, buffer_delay_table, buffer_slew_table,
                          buffer_cap_axis, buffer_slew_axis, fd_sr, fd_lr, fd_sf, fd_lf,
                          fs_sr, fs_lr, fs_sf, fs_lf, use_ceff, use_dmp, use_buf_regression):
    scale = 1e15
    len_c = torch.abs(sp_xy_c - ep_xy_c).sum(dim=1) * DBU_NM

    elmore_wire = WIRE_R * len_c * moment_m0 / scale
    if use_dmp:
        wire_delay_r, slew_out_r = rc_propagate(elmore_wire, slew_p[:, RF_RISE])
        wire_delay_f, slew_out_f = rc_propagate(elmore_wire, slew_p[:, RF_FALL])
    else:
        wire_delay_r, slew_out_r, wire_delay_f, slew_out_f = elmoreValue.apply(
            elmore_wire, slew_p[:, RF_RISE], slew_p[:, RF_FALL])

    arr_no_r  = arrival_p[:, RF_RISE] + wire_delay_r
    arr_no_f  = arrival_p[:, RF_FALL] + wire_delay_f
    slew_no_r = slew_out_r
    slew_no_f = slew_out_f

    ln = half_span_wire(sp_xy_c, ep_xy_c, xcen_buf, xin_buf)
    activation = torch.min(abs(b_flat_c), torch.ones_like(b_flat_c))
    internal = torch.max(abs(b_flat_c) - torch.ones_like(b_flat_c), torch.zeros_like(b_flat_c))

    elmore_buf_left = ln * WIRE_R * (buf_in_cap + 0.5 * ln * WIRE_C)
    if use_dmp:
        left_delay_r, left_slew_r = rc_propagate(elmore_buf_left, slew_p[:, RF_RISE])
        left_delay_f, left_slew_f = rc_propagate(elmore_buf_left, slew_p[:, RF_FALL])
    else:
        left_delay_r, left_slew_r, left_delay_f, left_slew_f = elmoreValue.apply(
            elmore_buf_left, slew_p[:, RF_RISE], slew_p[:, RF_FALL])

    buf_in_slew_r = left_slew_r
    buf_in_slew_f = left_slew_f

    M = int(len_c.numel())
    dtab_r  = buffer_delay_table[RF_RISE].expand(M, -1, -1)
    stab_r  = buffer_slew_table[RF_RISE].expand(M, -1, -1)
    axcap_r = buffer_cap_axis[RF_RISE].expand(M, -1)
    axslw_r = buffer_slew_axis[RF_RISE].expand(M, -1)
    dtab_f  = buffer_delay_table[RF_FALL].expand(M, -1, -1)
    stab_f  = buffer_slew_table[RF_FALL].expand(M, -1, -1)
    axcap_f = buffer_cap_axis[RF_FALL].expand(M, -1)
    axslw_f = buffer_slew_axis[RF_FALL].expand(M, -1)

    if use_ceff:
        c1 = pi_c[:, PI_C1]
        c2 = pi_c[:, PI_C2]
        rpi = pi_c[:, PI_RPI]
        ceff_r = batch_effective_load(buf_in_slew_r, c1 / scale, c2 / scale, rpi, dtab_r, stab_r, axslw_r, axcap_r)
        ceff_f = batch_effective_load(buf_in_slew_f, c1 / scale, c2 / scale, rpi, dtab_f, stab_f, axslw_f, axcap_f)
    else:
        moment_load = (pi_c[:, PI_C1] + pi_c[:, PI_C2]) / scale
        ceff_r = moment_load
        ceff_f = moment_load

    if use_buf_regression:
        buf_delay_r = fd_sr(buf_in_slew_r) + fd_lr(ceff_r)
        buf_delay_f = fd_sf(buf_in_slew_f) + fd_lf(ceff_f)
        buf_out_slew_r = fs_sr(buf_in_slew_r) + fs_lr(ceff_r)
        buf_out_slew_f = fs_sf(buf_in_slew_f) + fs_lf(ceff_f)
    else:
        buf_delay_r, buf_out_slew_r = eval_arc_timing(buf_in_slew_r, ceff_r, dtab_r, stab_r, axslw_r, axcap_r)
        buf_delay_f, buf_out_slew_f = eval_arc_timing(buf_in_slew_f, ceff_f, dtab_f, stab_f, axslw_f, axcap_f)

    load_c = moment_m0 / scale - 0.5 * WIRE_C * len_c
    elmore_buf_right = ln * WIRE_R * (load_c + 0.5 * ln * WIRE_C)
    if use_dmp:
        right_delay_r, right_slew_r = rc_propagate(elmore_buf_right, buf_out_slew_r)
        right_delay_f, right_slew_f = rc_propagate(elmore_buf_right, buf_out_slew_f)
    else:
        right_delay_r, right_slew_r, right_delay_f, right_slew_f = elmoreValue.apply(
            elmore_buf_right, buf_out_slew_r, buf_out_slew_f)

    arr_buf_r  = arrival_p[:, RF_RISE] + left_delay_r + buf_delay_r + right_delay_r
    arr_buf_f  = arrival_p[:, RF_FALL] + left_delay_f + buf_delay_f + right_delay_f
    slew_buf_r = right_slew_r
    slew_buf_f = right_slew_f

    b = b_flat_c
    arr_out = torch.stack([(1 - b) * arr_no_r + b * arr_buf_r,
                           (1 - b) * arr_no_f + b * arr_buf_f], dim=1)
    slew_out = torch.stack([(1 - b) * slew_no_r + b * slew_buf_r,
                            (1 - b) * slew_no_f + b * slew_buf_f], dim=1)
    return arr_out, slew_out


class WireSegmentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arrival_p, slew_p, moment_m0, pi_c, sp_xy_c, ep_xy_c, b_flat_c,
                xcen_buf, xin_buf, buf_in_cap, buffer_delay_table, buffer_slew_table,
                buffer_cap_axis, buffer_slew_axis, fd_sr, fd_lr, fd_sf, fd_lf,
                fs_sr, fs_lr, fs_sf, fs_lf, use_ceff, use_dmp, use_buf_regression):
        with torch.no_grad():
            arr_out, slew_out = _wire_segment_outputs(
                arrival_p, slew_p, moment_m0, pi_c, sp_xy_c, ep_xy_c, b_flat_c,
                xcen_buf, xin_buf, buf_in_cap, buffer_delay_table, buffer_slew_table,
                buffer_cap_axis, buffer_slew_axis, fd_sr, fd_lr, fd_sf, fd_lf,
                fs_sr, fs_lr, fs_sf, fs_lf, use_ceff, use_dmp, use_buf_regression)
        ctx.save_for_backward(arrival_p, slew_p, moment_m0, pi_c, sp_xy_c, ep_xy_c, b_flat_c)
        ctx.xcen_buf = xcen_buf
        ctx.xin_buf = xin_buf
        ctx.buf_in_cap = buf_in_cap
        ctx.buffer_delay_table = buffer_delay_table
        ctx.buffer_slew_table = buffer_slew_table
        ctx.buffer_cap_axis = buffer_cap_axis
        ctx.buffer_slew_axis = buffer_slew_axis
        ctx.fd_sr = fd_sr; ctx.fd_lr = fd_lr; ctx.fd_sf = fd_sf; ctx.fd_lf = fd_lf
        ctx.fs_sr = fs_sr; ctx.fs_lr = fs_lr; ctx.fs_sf = fs_sf; ctx.fs_lf = fs_lf
        ctx.use_ceff = use_ceff; ctx.use_dmp = use_dmp; ctx.use_buf_regression = use_buf_regression
        return arr_out, slew_out

    @staticmethod
    def backward(ctx, grad_arr, grad_slew):
        if (grad_arr is None and grad_slew is None) or not any(ctx.needs_input_grad):
            return (None,) * 25
        arrival_p, slew_p, moment_m0, pi_c, sp_xy_c, ep_xy_c, b_flat_c = ctx.saved_tensors
        with torch.enable_grad():
            arrival_p   = arrival_p.detach().requires_grad_(ctx.needs_input_grad[0])
            slew_p      = slew_p.detach().requires_grad_(ctx.needs_input_grad[1])
            moment_m0   = moment_m0.detach().requires_grad_(ctx.needs_input_grad[2])
            pi_c        = pi_c.detach().requires_grad_(ctx.needs_input_grad[3])
            sp_xy_c     = sp_xy_c.detach().requires_grad_(ctx.needs_input_grad[4])
            ep_xy_c     = ep_xy_c.detach().requires_grad_(ctx.needs_input_grad[5])
            b_flat_c    = b_flat_c.detach().requires_grad_(ctx.needs_input_grad[6])
            arr_out, slew_out = _wire_segment_outputs(
                arrival_p, slew_p, moment_m0, pi_c, sp_xy_c, ep_xy_c, b_flat_c,
                ctx.xcen_buf, ctx.xin_buf, ctx.buf_in_cap, ctx.buffer_delay_table, ctx.buffer_slew_table,
                ctx.buffer_cap_axis, ctx.buffer_slew_axis, ctx.fd_sr, ctx.fd_lr, ctx.fd_sf, ctx.fd_lf,
                ctx.fs_sr, ctx.fs_lr, ctx.fs_sf, ctx.fs_lf, ctx.use_ceff, ctx.use_dmp, ctx.use_buf_regression)
            inputs = (arrival_p, slew_p, moment_m0, pi_c, sp_xy_c, ep_xy_c, b_flat_c)
            needs = ctx.needs_input_grad[:7]
            inputs_req = [inp for inp, need in zip(inputs, needs) if need]
            if inputs_req:
                grad_arr   = torch.zeros_like(arr_out)  if grad_arr  is None else grad_arr
                grad_slew  = torch.zeros_like(slew_out) if grad_slew is None else grad_slew
                grads_req  = torch.autograd.grad((arr_out, slew_out), inputs_req, (grad_arr, grad_slew),
                                                 retain_graph=False, create_graph=False, allow_unused=True)
            else:
                grads_req = []
        grads = [None] * 7
        if grads_req:
            it = iter(grads_req)
            for idx, need in enumerate(needs):
                if need:
                    grads[idx] = next(it)
        return (*grads, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)


class GateGroupedOutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_arrs, in_slews, c1, c2, rpi, arc_idx_r, arc_idx_f,
                group, num_groups, unateness, delay_table, slew_table,
                load_index, slew_index, use_ceff, lse_beta):
        with torch.no_grad():
            values = _gate_values_pair(in_arrs, in_slews, c1, c2, rpi, arc_idx_r, arc_idx_f,
                                       unateness, delay_table, slew_table, load_index, slew_index, use_ceff)
            grouped_out = _grouped_lse(values, group, num_groups, lse_beta)
        ctx.save_for_backward(in_arrs, in_slews, c1, c2, rpi, arc_idx_r, arc_idx_f, group)
        ctx.unateness = unateness; ctx.delay_table = delay_table; ctx.slew_table = slew_table
        ctx.load_index = load_index; ctx.slew_index = slew_index
        ctx.use_ceff = use_ceff; ctx.lse_beta = lse_beta; ctx.num_groups = num_groups
        return grouped_out

    @staticmethod
    def backward(ctx, dL_dgrouped):
        if dL_dgrouped is None or not any(ctx.needs_input_grad):
            return (None,) * 16
        in_arrs, in_slews, c1, c2, rpi, arc_idx_r, arc_idx_f, group = ctx.saved_tensors
        with torch.enable_grad():
            in_arrs  = in_arrs.detach().requires_grad_(ctx.needs_input_grad[0])
            in_slews = in_slews.detach().requires_grad_(ctx.needs_input_grad[1])
            c1  = c1.detach().requires_grad_(ctx.needs_input_grad[2])
            c2  = c2.detach().requires_grad_(ctx.needs_input_grad[3])
            rpi = rpi.detach().requires_grad_(ctx.needs_input_grad[4])
            values = _gate_values_pair(in_arrs, in_slews, c1, c2, rpi, arc_idx_r, arc_idx_f,
                                       ctx.unateness, ctx.delay_table, ctx.slew_table,
                                       ctx.load_index, ctx.slew_index, ctx.use_ceff)
            grouped_out = _grouped_lse(values, group, ctx.num_groups, ctx.lse_beta)
            grads = torch.autograd.grad(grouped_out, (in_arrs, in_slews, c1, c2, rpi), dL_dgrouped,
                                        retain_graph=False, create_graph=False, allow_unused=True)
        return (*grads, None, None, None, None, None, None, None, None, None, None, None)


class elmoreValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, elmore_wire, slews_r, slews_f):
        slew_out_r = torch.sqrt((slews_r * 1e12).square() + 1.921 * (1e12 * elmore_wire).square()) / 1e12
        slew_out_f = torch.sqrt((slews_f * 1e12).square() + 1.921 * (1e12 * elmore_wire).square()) / 1e12
        ctx.save_for_backward(elmore_wire, slews_r, slews_f, slew_out_r, slew_out_f)
        return elmore_wire, slew_out_r, elmore_wire, slew_out_f

    @staticmethod
    def backward(ctx, dLdD_r, dLdS_r, dLdD_f, dLdS_f):
        elmore_wire, slews_r, slews_f, slew_out_r, slew_out_f = ctx.saved_tensors
        slew_out_r = slew_out_r.clamp_min(1e-30)
        slew_out_f = slew_out_f.clamp_min(1e-30)
        dS_out_r_dD_r  = 1.921 * elmore_wire / slew_out_r
        dS_out_f_dD_f  = 1.921 * elmore_wire / slew_out_f
        dS_out_r_dS_in_r = slews_r / slew_out_r
        dS_out_f_dS_in_f = slews_f / slew_out_f
        dL_dD_r = dLdD_r + dLdS_r * dS_out_r_dD_r
        dL_dD_f = dLdD_f + dLdS_f * dS_out_f_dD_f
        dL_dS_r = dLdS_r * dS_out_r_dS_in_r
        dL_dS_f = dLdS_f * dS_out_f_dS_in_f
        return dL_dD_r + dL_dD_f, dL_dS_r, dL_dS_f
