"""
timing.graph — STAGraph data container and builder.

``STAGraph`` is a frozen dataclass-like container that holds every tensor
needed by the timing engine.  ``STAGraphBuilder`` constructs it once from
the circuit objects (CellLibrary + SteinerNetworkBuilder) and caches the
result.

Tensors are organised into nine logical groups:

  1. Coordinates & connectivity (cell positions, parent/child STPs).
  2. Gate library look-up tables (NLDM delay/slew 2-D arrays).
  3. Differentiable gate-sizing parameter (``U_flat``, a ``nn.Parameter``).
  4. Pin-to-STP mapping and input-capacitance cache.
  5. Buffer insertion parameters.
  6. Timing graph boundary conditions (start arrival/slew, end required times).
  7. Execution plan (levelized list of WireTask / GateTask objects).
  8. Power activity data.
  9. Π-model source / internal STP index sets.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import torch
import torch.nn as nn

import openroad as ord  # type: ignore

from placeopt.circuit.library import CellLibrary
from placeopt.circuit.steiner import SteinerNetworkBuilder
from placeopt.circuit.components import SteinerPoint
from placeopt.timing.constants import (
    RISE, FALL, M0, R_PER_LEN, C_PER_LEN, TO_MICRON,
)
from placeopt.timing.schedule import (
    WireTask, GateTask,
    build_execution_plan,
    build_net_traverse_schedule,
)
from placeopt.eda.buffering import is_buffer, is_inverter


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

_ORIENT_MAP = {"R0": 0, "R90": 1, "R180": 2, "R270": 3,
               "MX": 4, "MY": 5, "MXR90": 6, "MYR90": 7}


def _orient_str(inst) -> str:
    orient = getattr(inst, "getOrient", lambda: None)()
    if orient is None:
        return "R0"
    return str(getattr(orient, "name", orient))


def _inst_origin(inst) -> Tuple[float, float]:
    if hasattr(inst, "getLocation"):
        loc = inst.getLocation()
        return float(loc[0]), float(loc[1])
    if hasattr(inst, "getOrigin"):
        loc = inst.getOrigin()
        return float(loc[0]), float(loc[1])
    bbox = inst.getBBox()
    return float(bbox.xMin()), float(bbox.yMin())


def _bbox_xywh(bbox):
    if bbox is None:
        return None
    return (float(bbox.xMin()), float(bbox.yMin()),
            float(bbox.xMax() - bbox.xMin()),
            float(bbox.yMax() - bbox.yMin()))


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class STAGraph:
    """
    Immutable container of all pre-built tensors for the timing engine.

    The only mutable fields are the three learnable ``nn.Parameter``
    objects: ``cell_xy``, ``U_flat``, and ``buffering_tensor``.
    """

    # ── 1. Positions ────────────────────────────────────────────────────────
    cell_xy: nn.Parameter          # [G, 2]  gate origins  (OPTIMISED)
    stp_gate_mapping: torch.Tensor # [N]     stp → gate index (-1 = fixed)
    stp_offset: torch.Tensor       # [N, 2]  pin offset within gate cell
    stp_parent_idx: torch.Tensor   # [N]     parent stp index
    stp_fixed_xy: torch.Tensor     # [N, 2]  static stp positions
    pos_xy: torch.Tensor           # [N, 2]  current stp positions (updated each fwd)
    pos_entry_u_index: torch.Tensor# [E]     U_flat index per pin entry
    pos_entry_stp_idx: torch.Tensor# [E]     stp index per pin entry
    stp_pin_id: torch.Tensor       # [N]     pin id for gate-pin stps
    u_master_id: torch.Tensor      # [total_u] master id per U entry
    master_w: torch.Tensor         # [M]
    master_h: torch.Tensor         # [M]
    gate_orient: torch.Tensor      # [G]
    gate_movable_mask: torch.Tensor# [G]
    cell_box: torch.Tensor         # [G, 4]  (x, y, w, h) for density
    boundary: torch.Tensor         # [2, 2]  [[xmin, ymin], [xmax, ymax]]
    constant_blockage_xy: torch.Tensor
    constant_blockage_wh: torch.Tensor
    hard_blockage_xy: torch.Tensor
    hard_blockage_wh: torch.Tensor
    soft_blockage_xy: torch.Tensor
    soft_blockage_wh: torch.Tensor
    macro_xy: torch.Tensor
    macro_wh: torch.Tensor
    init_cell_idx: torch.Tensor    # [Gi]
    init_cell_xy: torch.Tensor     # [Gi, 2]

    # ── 2. Liberty LUTs ─────────────────────────────────────────────────────
    level_edges: List[Tuple[torch.Tensor, torch.Tensor]]
    delay_table: torch.Tensor      # [A, K, K]
    slew_table: torch.Tensor       # [A, K, K]
    load_index: torch.Tensor       # [A, K]
    slew_index: torch.Tensor       # [A, K]
    r_driver: torch.Tensor         # [A]
    unateness: torch.Tensor        # [A]
    arc_index_map: Dict[Any, int]
    arc_lut_r: torch.Tensor        # [M, P, P]  arc index for rise output
    arc_lut_f: torch.Tensor        # [M, P, P]  arc index for fall output

    # ── 3. Gate sizing parameters ────────────────────────────────────────────
    U_flat: nn.Parameter           # [total_u]  (OPTIMISED)
    u_gate_len: torch.Tensor       # [G]
    u_gate_off: torch.Tensor       # [G+1]
    u_gate_id: torch.Tensor        # [total_u]

    # ── 4. Pin mapping ───────────────────────────────────────────────────────
    inpin_idx: torch.Tensor        # [P_in]
    outpin_idx: torch.Tensor       # [P_out]
    pin_entry_u_index: torch.Tensor# [total_entry]
    pin_entry_pin_id: torch.Tensor # [total_entry]
    pin_name_id: torch.Tensor      # [P_in]
    master_pin_cap: torch.Tensor   # [M, P_names]

    # ── 5. Buffer insertion ──────────────────────────────────────────────────
    buffering_tensor: nn.Parameter # [N]  logits  (OPTIMISED)
    chosen_buffer: Any
    chosen_inverter: Any
    buffer_wh: torch.Tensor        # [2]
    buffer_delay_table: torch.Tensor  # [2, K, K]
    buffer_slew_table: torch.Tensor   # [2, K, K]
    buffer_cap_axis: torch.Tensor     # [2, K]
    buffer_slew_axis: torch.Tensor    # [2, K]
    buf_in_cap: float
    inv_in_cap: float
    xin_buf: torch.Tensor          # [2]
    xout_buf: torch.Tensor         # [2]
    xcen_buf: torch.Tensor         # [2]
    fd_sr: Any; fd_lr: Any; fd_sf: Any; fd_lf: Any
    fs_sr: Any; fs_lr: Any; fs_sf: Any; fs_lf: Any

    # ── 6. Timing boundaries ─────────────────────────────────────────────────
    start_idx: torch.Tensor        # [S]
    start_arrival_init: torch.Tensor  # [S, 2]
    start_slew_init: torch.Tensor     # [S, 2]
    end_idx: torch.Tensor          # [E]
    require: torch.Tensor          # [E, 2]

    # ── 7. Execution plan ────────────────────────────────────────────────────
    execution_plan: List[Union[WireTask, GateTask]]

    # ── 8. Power activity ────────────────────────────────────────────────────
    inpin_activity: torch.Tensor
    outpin_activity: torch.Tensor
    stp_activity: torch.Tensor
    no_driver_mask: torch.Tensor
    driver_mask: torch.Tensor

    # ── 9. Π model indices ───────────────────────────────────────────────────
    pi_source_idxs: torch.Tensor
    pi_internal_idxs: torch.Tensor


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class STAGraphBuilder:
    """
    Constructs an ``STAGraph`` from circuit library and Steiner network objects.

    All data-collection loops run on the CPU; tensors are moved to *device*
    in a single batch at the end of each ``_build_*`` method.
    """

    def build(
        self,
        origin_gate_weights,
        origin_buffer_weights,
        steiner_net: SteinerNetworkBuilder,
        cell_lib: CellLibrary,
        timing,
        device: torch.device,
    ) -> STAGraph:
        print("[INFO] Building STA graph tensors...")
        t0 = time.perf_counter()
        design = cell_lib.design
        block = design.getBlock()

        # ── Positions ─────────────────────────────────────────────────────
        t1 = time.perf_counter()
        (pos_xy, stp_parent_idx, stp_fixed_xy) = self._build_point_positions(
            steiner_net.stp_list, device)
        print(f"[TIME] point positions: {time.perf_counter()-t1:.3f}s")

        # ── Liberty tables ─────────────────────────────────────────────────
        t1 = time.perf_counter()
        (delay_table, slew_table, load_index, slew_index,
         r_driver, unateness, arc_index_map) = self._build_arc_tables(cell_lib, device)
        print(f"[TIME] arc tables: {time.perf_counter()-t1:.3f}s")

        # ── Level edges (for moment backward pass) ─────────────────────────
        t1 = time.perf_counter()
        level_edges = build_net_traverse_schedule(cell_lib.signal_nets, device)
        print(f"[TIME] level edges: {time.perf_counter()-t1:.3f}s")

        # ── Gate sizing (U_flat) ───────────────────────────────────────────
        t1 = time.perf_counter()
        (U_flat, u_gate_len, u_gate_off, u_gate_id,
         inpin_idx, outpin_idx,
         pin_entry_u_index, pin_entry_pin_id,
         pin_name_id, master_pin_cap) = self._build_gate_sizing_cache(
             cell_lib, steiner_net, timing, origin_gate_weights, device)
        print(f"[TIME] gate sizing cache: {time.perf_counter()-t1:.3f}s")

        # ── Cell positions ─────────────────────────────────────────────────
        t1 = time.perf_counter()
        (cell_xy, stp_gate_mapping, stp_offset,
         pos_entry_u_index, pos_entry_stp_idx, stp_pin_id,
         u_master_id, master_w, master_h,
         gate_orient, gate_movable_mask,
         pin_name_to_id) = self._build_cell_position_tensors(
             steiner_net, cell_lib, u_gate_off, device)
        print(f"[TIME] cell positions: {time.perf_counter()-t1:.3f}s")

        # ── Arc LUTs (master × pin_from × pin_to → arc_idx) ───────────────
        arc_lut_r, arc_lut_f = self._build_arc_luts(
            arc_index_map, cell_lib.MasterVec, pin_name_to_id, device)

        # ── Blockages and boundary ─────────────────────────────────────────
        (boundary, hard_blockage_xy, hard_blockage_wh,
         soft_blockage_xy, soft_blockage_wh,
         macro_xy, macro_wh,
         constant_blockage_xy, constant_blockage_wh) = self._build_blockage_tensors(
             block, device)

        # ── Buffer insertion ───────────────────────────────────────────────
        t1 = time.perf_counter()
        buf_data = self._build_buffering_tensors(
            steiner_net, cell_lib, timing, origin_buffer_weights, device)
        print(f"[TIME] buffer tensors: {time.perf_counter()-t1:.3f}s")

        # ── Timing boundaries ──────────────────────────────────────────────
        t1 = time.perf_counter()
        (start_idx, start_arrival_init, start_slew_init,
         end_idx, require) = self._build_timing_boundaries(
             timing, steiner_net.start_points, steiner_net.end_points, device)
        print(f"[TIME] timing boundaries: {time.perf_counter()-t1:.3f}s")

        # ── Execution plan ─────────────────────────────────────────────────
        t1 = time.perf_counter()
        execution_plan = build_execution_plan(steiner_net, u_gate_off, device)
        print(f"[TIME] execution plan: {time.perf_counter()-t1:.3f}s")

        # ── Power activity ─────────────────────────────────────────────────
        t1 = time.perf_counter()
        (inpin_activity, outpin_activity,
         stp_activity, no_driver_mask, driver_mask) = self._build_power_activity(
             steiner_net, timing, inpin_idx, outpin_idx, device)
        print(f"[TIME] power activity: {time.perf_counter()-t1:.3f}s")

        # ── Π model indices ────────────────────────────────────────────────
        t1 = time.perf_counter()
        pi_source_idxs, pi_internal_idxs = self._build_pi_compute_indices(
            steiner_net, device)
        print(f"[TIME] pi indices: {time.perf_counter()-t1:.3f}s")

        init_cell_idx = torch.tensor(
            getattr(cell_lib, "init_cell_idxs", []), dtype=torch.long, device=device)
        init_cell_xy = torch.tensor(
            getattr(cell_lib, "init_pos", []), dtype=torch.float32, device=device)

        cell_box = cell_xy.data.new_zeros((cell_xy.shape[0], 4))

        print(f"[INFO] STAGraph built in {time.perf_counter()-t0:.2f}s")

        return STAGraph(
            cell_xy=cell_xy, stp_gate_mapping=stp_gate_mapping,
            stp_offset=stp_offset, stp_parent_idx=stp_parent_idx,
            stp_fixed_xy=stp_fixed_xy, pos_xy=pos_xy,
            pos_entry_u_index=pos_entry_u_index, pos_entry_stp_idx=pos_entry_stp_idx,
            stp_pin_id=stp_pin_id, u_master_id=u_master_id,
            master_w=master_w, master_h=master_h,
            gate_orient=gate_orient, gate_movable_mask=gate_movable_mask,
            cell_box=cell_box, boundary=boundary,
            constant_blockage_xy=constant_blockage_xy,
            constant_blockage_wh=constant_blockage_wh,
            hard_blockage_xy=hard_blockage_xy, hard_blockage_wh=hard_blockage_wh,
            soft_blockage_xy=soft_blockage_xy, soft_blockage_wh=soft_blockage_wh,
            macro_xy=macro_xy, macro_wh=macro_wh,
            init_cell_idx=init_cell_idx, init_cell_xy=init_cell_xy,
            level_edges=level_edges,
            delay_table=delay_table, slew_table=slew_table,
            load_index=load_index, slew_index=slew_index,
            r_driver=r_driver, unateness=unateness,
            arc_index_map=arc_index_map,
            arc_lut_r=arc_lut_r, arc_lut_f=arc_lut_f,
            U_flat=U_flat, u_gate_len=u_gate_len,
            u_gate_off=u_gate_off, u_gate_id=u_gate_id,
            inpin_idx=inpin_idx, outpin_idx=outpin_idx,
            pin_entry_u_index=pin_entry_u_index,
            pin_entry_pin_id=pin_entry_pin_id,
            pin_name_id=pin_name_id, master_pin_cap=master_pin_cap,
            **buf_data,
            start_idx=start_idx, start_arrival_init=start_arrival_init,
            start_slew_init=start_slew_init,
            end_idx=end_idx, require=require,
            execution_plan=execution_plan,
            inpin_activity=inpin_activity, outpin_activity=outpin_activity,
            stp_activity=stp_activity,
            no_driver_mask=no_driver_mask, driver_mask=driver_mask,
            pi_source_idxs=pi_source_idxs, pi_internal_idxs=pi_internal_idxs,
        )

    # ------------------------------------------------------------------
    # Individual build helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_point_positions(stp_list, device):
        pos, parents = [], []
        for stp in stp_list:
            pos.append([float(stp.x), float(stp.y)])
            from placeopt.circuit.components import SteinerPoint as SP
            from placeopt.circuit.library import _is_driver_pin
            if _is_driver_pin(stp.Pin) if stp.Pin else (not stp.prevs):
                parents.append(int(stp.idx))
            else:
                parents.append(int(stp.prevs[0].idx))
        pos_xy   = torch.tensor(pos, dtype=torch.float32, device=device)
        par_idx  = torch.tensor(parents, dtype=torch.long, device=device)
        return pos_xy, par_idx, pos_xy.clone()

    @staticmethod
    def _build_arc_tables(cell_lib: CellLibrary, device):
        delay_t, slew_t, load_t, slew_t2, rd_t, una = [], [], [], [], [], []
        arc_index_map: Dict = {}
        cnt = 0
        for master in cell_lib.MasterVec:
            tables = cell_lib.get_table(master)
            for (in_pin, out_pin, rf), tbl in tables.items():
                arc_index_map[(master.getName(), (in_pin, out_pin, rf))] = cnt
                d = tbl.delay_table
                s = tbl.slew_table
                def _pad7(t):
                    if t.shape == (7, 7):
                        return t
                    v = t[0, 0]
                    return v.view(1, 1).expand(7, 7)
                def _padax(a):
                    if a.numel() == 7:
                        return a
                    return torch.linspace(a[0].item(), a[-1].item(), 7,
                                         dtype=a.dtype, device=a.device)
                delay_t.append(_pad7(d))
                slew_t.append(_pad7(s))
                load_t.append(_padax(tbl.axis_1))
                slew_t2.append(_padax(tbl.axis_0))
                rd_t.append(torch.tensor(tbl.driver_rd, dtype=torch.float32).reshape(()))
                una.append(1 if rf != tbl.in_rf else 0)
                cnt += 1
        return (
            torch.stack(delay_t).to(device),
            torch.stack(slew_t).to(device),
            torch.stack(load_t).to(device),
            torch.stack(slew_t2).to(device),
            torch.stack(rd_t).to(device),
            torch.tensor(una, dtype=torch.long, device=device),
            arc_index_map,
        )

    @staticmethod
    def _build_arc_luts(arc_index_map, master_vec, pin_name_to_id, device):
        M = len(master_vec)
        P = len(pin_name_to_id)
        m_to_id = {m.getName(): i for i, m in enumerate(master_vec)}
        r = torch.full((M, P, P), -1, dtype=torch.long, device=device)
        f = torch.full((M, P, P), -1, dtype=torch.long, device=device)
        for (mname, (fp, tp, rf)), idx in arc_index_map.items():
            mid = m_to_id.get(mname)
            fid = pin_name_to_id.get(fp)
            tid = pin_name_to_id.get(tp)
            if None in (mid, fid, tid):
                continue
            (r if rf == "^" else f)[mid, fid, tid] = idx
        return r, f

    @staticmethod
    def _build_gate_sizing_cache(cell_lib, steiner_net, timing, origin_weight, device):
        gates = cell_lib.signal_gates
        G = len(gates)
        gate_len  = [len(g.equiv_masters) for g in gates]
        len_t     = torch.tensor(gate_len, dtype=torch.long, device=device)
        off_t     = torch.zeros(G + 1, dtype=torch.long, device=device)
        off_t[1:] = torch.cumsum(len_t, 0)
        gate_id   = torch.repeat_interleave(torch.arange(G, dtype=torch.long, device=device), len_t)

        # Initialise U so that the first (reference) variant is preferred.
        u_init = []
        for count in gate_len:
            if count:
                u_init.append(float(origin_weight))
                u_init.extend([1.0] * (count - 1))
        U_flat = nn.Parameter(torch.tensor(u_init, dtype=torch.float32, device=device))

        inpin_idx, outpin_idx, gate_for_pin, pin_names = [], [], [], []
        for gi, g in enumerate(gates):
            for pin in g.input_pins:
                if pin.steiner_pt is None:
                    continue
                inpin_idx.append(pin.steiner_pt.idx)
                gate_for_pin.append(gi)
                pin_names.append(pin.name)
            for pin in g.output_pins:
                if pin.steiner_pt is None:
                    continue
                outpin_idx.append(pin.steiner_pt.idx)

        inpin_t  = torch.tensor(inpin_idx, dtype=torch.long, device=device)
        outpin_t = torch.tensor(outpin_idx, dtype=torch.long, device=device)
        pin_gate = torch.tensor(gate_for_pin, dtype=torch.long, device=device)

        # Assign a compact integer ID to each unique pin name.
        pin_name_to_id: Dict[str, int] = {}
        pin_name_id_list = []
        for nm in pin_names:
            pid = pin_name_to_id.setdefault(nm, len(pin_name_to_id))
            pin_name_id_list.append(pid)
        pin_name_id_t = torch.tensor(pin_name_id_list, dtype=torch.long, device=device)

        # Build per-pin U-entry mappings.
        P = len(inpin_idx)
        pin_len = len_t[pin_gate]
        pin_off = torch.zeros(P + 1, dtype=torch.long, device=device)
        pin_off[1:] = torch.cumsum(pin_len, 0)
        total_entry = int(pin_off[-1].item())

        entry_pin_id  = torch.repeat_interleave(torch.arange(P, dtype=torch.long, device=device), pin_len)
        entry_pos     = torch.arange(total_entry, dtype=torch.long, device=device)
        local         = entry_pos - pin_off[entry_pin_id]
        entry_u_idx   = off_t[pin_gate[entry_pin_id]] + local

        # Input capacitance table per (master, pin_name_id).
        corner0   = timing.getCorners()[0]
        max_mode  = timing.Max
        pid_to_name = [None] * len(pin_name_to_id)
        for nm, pid in pin_name_to_id.items():
            pid_to_name[pid] = nm

        mp_cap = [[0.0] * len(pin_name_to_id) for _ in range(len(cell_lib.MasterVec))]
        for m_idx, master in enumerate(cell_lib.MasterVec):
            for pid, nm in enumerate(pid_to_name):
                if nm is None:
                    continue
                mterm = master.findMTerm(nm)
                mp_cap[m_idx][pid] = timing.getPortCap(mterm, corner0, max_mode) if mterm else 0.0
        mp_cap_t = torch.tensor(mp_cap, dtype=torch.float32, device=device)

        return (U_flat, len_t, off_t, gate_id,
                inpin_t, outpin_t, entry_u_idx, entry_pin_id,
                pin_name_id_t, mp_cap_t)

    @staticmethod
    def _build_cell_position_tensors(steiner_net, cell_lib, u_gate_off, device):
        gates   = cell_lib.signal_gates
        masters = cell_lib.MasterVec
        stps    = steiner_net.stp_list
        N, G, M = len(stps), len(gates), len(masters)

        master_name_to_id = {m.getName(): i for i, m in enumerate(masters)}

        # Collect unique pin names across all signal gates.
        pin_name_to_id: Dict[str, int] = {}
        for g in gates:
            for p in g.pins.values():
                pin_name_to_id.setdefault(p.name, len(pin_name_to_id))

        stp_gate_idx   = [-1] * N
        stp_offset     = [[0.0, 0.0]] * N
        stp_pin_id     = [-1] * N
        cell_xy_list   = [[0.0, 0.0]] * G
        gate_orient    = [0] * G
        gate_movable   = [1] * G
        master_w       = [0.0] * M
        master_h       = [0.0] * M

        for mi, m in enumerate(masters):
            master_w[mi] = float(m.getWidth())
            master_h[mi] = float(m.getHeight())

        for g in gates:
            gi = g.idx
            if gi < 0:
                continue
            inst = g.db_inst
            master = inst.getMaster()
            gate_movable[gi] = 0 if (master and master.isBlock()) else 1
            gate_orient[gi]  = _ORIENT_MAP.get(_orient_str(inst), 0)
            cell_xy_list[gi] = list(_inst_origin(inst))

        u_gate_off_list = u_gate_off.detach().cpu().tolist()
        pos_entry_u_idx  = []
        pos_entry_stp_idx = []
        u_master_id_list = []

        for g in gates:
            for m in g.equiv_masters:
                u_master_id_list.append(master_name_to_id.get(m.getName(), -1))

        for stp in stps:
            g = getattr(stp.Pin, "gate", None) if stp.Pin else None
            gi = getattr(g, "idx", -1)
            stp_gate_idx[int(stp.idx)] = gi
            if gi < 0:
                stp_offset[int(stp.idx)] = [float(stp.x), float(stp.y)]
                continue
            pname   = stp.Pin.name
            pin_id  = pin_name_to_id.get(pname, -1)
            stp_pin_id[int(stp.idx)] = pin_id
            bx, by  = cell_xy_list[gi]
            stp_offset[int(stp.idx)] = [float(stp.x) - bx, float(stp.y) - by]
            for mi in range(len(g.equiv_masters)):
                pos_entry_u_idx.append(int(u_gate_off_list[gi]) + mi)
                pos_entry_stp_idx.append(int(stp.idx))

        cell_xy = nn.Parameter(
            torch.tensor(cell_xy_list, dtype=torch.float32, device=device))
        stp_gate_t = torch.tensor(stp_gate_idx, dtype=torch.long, device=device)
        stp_off_t  = torch.tensor(stp_offset,   dtype=torch.float32, device=device)
        stp_pin_t  = torch.tensor(stp_pin_id,   dtype=torch.long, device=device)
        u_master_t = torch.tensor(u_master_id_list, dtype=torch.long, device=device)
        mw_t = torch.tensor(master_w, dtype=torch.float32, device=device)
        mh_t = torch.tensor(master_h, dtype=torch.float32, device=device)
        orient_t   = torch.tensor(gate_orient, dtype=torch.long, device=device)
        movable_t  = torch.tensor(gate_movable, dtype=torch.float32, device=device)

        if pos_entry_u_idx:
            peu_t = torch.tensor(pos_entry_u_idx,   dtype=torch.long, device=device)
            pes_t = torch.tensor(pos_entry_stp_idx, dtype=torch.long, device=device)
        else:
            peu_t = torch.zeros(0, dtype=torch.long, device=device)
            pes_t = torch.zeros(0, dtype=torch.long, device=device)

        return (cell_xy, stp_gate_t, stp_off_t,
                peu_t, pes_t, stp_pin_t, u_master_t,
                mw_t, mh_t, orient_t, movable_t, pin_name_to_id)

    @staticmethod
    def _build_blockage_tensors(block, device):
        bbox = block.getBBox()
        boundary = torch.tensor(
            [[bbox.xMin(), bbox.yMin()], [bbox.xMax(), bbox.yMax()]],
            dtype=torch.float32, device=device)

        hard_xy, hard_wh, soft_xy, soft_wh = [], [], [], []
        for blk in block.getBlockages():
            rect = _bbox_xywh(blk.getBBox())
            if rect is None:
                continue
            x0, y0, w, h = rect
            if getattr(blk, "isSoft", lambda: False)():
                soft_xy.append([x0, y0]); soft_wh.append([w, h])
            else:
                hard_xy.append([x0, y0]); hard_wh.append([w, h])

        macro_xy_l, macro_wh_l = [], []
        for inst in block.getInsts():
            m = inst.getMaster()
            if m is None or not m.isBlock():
                continue
            rect = _bbox_xywh(inst.getBBox())
            if rect is None:
                continue
            x0, y0, w, h = rect
            macro_xy_l.append([x0, y0]); macro_wh_l.append([w, h])

        def _t(lst):
            return torch.tensor(lst, dtype=torch.float32, device=device) if lst \
                   else torch.zeros((0, 2), dtype=torch.float32, device=device)

        hard_xy_t = _t(hard_xy); hard_wh_t = _t(hard_wh)
        soft_xy_t = _t(soft_xy); soft_wh_t = _t(soft_wh)
        macro_xy_t= _t(macro_xy_l); macro_wh_t= _t(macro_wh_l)

        const_xy = hard_xy + macro_xy_l
        const_wh = hard_wh + macro_wh_l
        const_xy_t = _t(const_xy); const_wh_t = _t(const_wh)

        return (boundary, hard_xy_t, hard_wh_t, soft_xy_t, soft_wh_t,
                macro_xy_t, macro_wh_t, const_xy_t, const_wh_t)

    @staticmethod
    def _build_buffering_tensors(steiner_net, cell_lib, timing, init_value, device):
        from placeopt.timing.schedule import _make_buf_funcs, _buf_regression
        N = len(steiner_net.stp_list)
        buf_tensor = nn.Parameter(
            torch.full((N,), init_value, dtype=torch.float32, device=device))

        corner0 = timing.getCorners()[0]
        chosen_buf, chosen_inv = _pick_smallest_buf_inv(cell_lib, timing, corner0)

        buf_in_cap = timing.getPortCap(chosen_buf.findMTerm("A"), corner0, timing.Max)
        inv_in_cap = timing.getPortCap(chosen_inv.findMTerm("A"), corner0, timing.Max)

        buf_wh = torch.tensor(
            [float(chosen_buf.getWidth()), float(chosen_buf.getHeight())],
            dtype=torch.float32, device=device)

        buf_delay = torch.zeros((2, 7, 7), dtype=torch.float32, device=device)
        buf_slew  = torch.zeros((2, 7, 7), dtype=torch.float32, device=device)
        buf_cap_ax= torch.zeros((2, 7), dtype=torch.float32, device=device)
        buf_slw_ax= torch.zeros((2, 7), dtype=torch.float32, device=device)

        models_d = [None, None]
        models_s = [None, None]
        tables = cell_lib.get_table(chosen_buf)
        for tbl in tables.values():
            idx = 0 if tbl.out_rf == "^" else 1
            buf_delay[idx] = tbl.delay_table.to(device=device, dtype=torch.float32)
            buf_slew[idx]  = tbl.slew_table.to(device=device, dtype=torch.float32)
            buf_cap_ax[idx]= tbl.axis_1.to(device=device, dtype=torch.float32)
            buf_slw_ax[idx]= tbl.axis_0.to(device=device, dtype=torch.float32)
            if models_d[idx] is None:
                models_d[idx] = _buf_regression(tbl.delay_table, tbl.axis_0, tbl.axis_1)
                models_s[idx] = _buf_regression(tbl.slew_table,  tbl.axis_0, tbl.axis_1)

        def _coeff(m, key, defaults):
            if m is not None and "coeffs" in m:
                return torch.as_tensor(m["coeffs"], dtype=torch.float32, device=device)
            return torch.tensor(defaults[key], dtype=torch.float32, device=device)

        _DEFAULT = {
            "dr": [5.2228e-12, 4.9731e-09, 4.6879e-12, 2.3443e-11, -4.2855e-12],
            "df": [6.2871e-12, 4.1882e-09, 2.7307e-11, 2.1773e-11, -5.9683e-12],
            "sr": [3.8538e-12, 1.0859e-08, 7.2963e-12, 8.7341e-13, -1.2711e-12],
            "sf": [3.8091e-12, 8.4023e-09, 9.4455e-12, 1.0353e-12, -2.3475e-12],
        }

        def _scale(m, ax, fallback):
            return torch.tensor(
                m.get("s_scale", fallback.abs().max().clamp_min(1e-12).item())
                if m else fallback.abs().max().clamp_min(1e-12).item(),
                dtype=torch.float32, device=device)

        ss_r = _scale(models_d[0], buf_slw_ax[0], buf_slw_ax[0])
        cs_r = _scale(models_d[0], buf_cap_ax[0], buf_cap_ax[0])
        ss_f = _scale(models_d[1], buf_slw_ax[1], buf_slw_ax[1])
        cs_f = _scale(models_d[1], buf_cap_ax[1], buf_cap_ax[1])

        fd_sr, fd_lr = _make_buf_funcs(_coeff(models_d[0], "dr", _DEFAULT), ss_r, cs_r)
        fd_sf, fd_lf = _make_buf_funcs(_coeff(models_d[1], "df", _DEFAULT), ss_f, cs_f)
        fs_sr, fs_lr = _make_buf_funcs(_coeff(models_s[0], "sr", _DEFAULT), ss_r, cs_r)
        fs_sf, fs_lf = _make_buf_funcs(_coeff(models_s[1], "sf", _DEFAULT), ss_f, cs_f)

        xin_buf, xout_buf, xcen_buf = _buf_pin_coords(chosen_buf, device)

        return dict(
            buffering_tensor=buf_tensor, chosen_buffer=chosen_buf, chosen_inverter=chosen_inv,
            buffer_wh=buf_wh,
            buffer_delay_table=buf_delay, buffer_slew_table=buf_slew,
            buffer_cap_axis=buf_cap_ax, buffer_slew_axis=buf_slw_ax,
            buf_in_cap=buf_in_cap, inv_in_cap=inv_in_cap,
            xin_buf=xin_buf, xout_buf=xout_buf, xcen_buf=xcen_buf,
            fd_sr=fd_sr, fd_lr=fd_lr, fd_sf=fd_sf, fd_lf=fd_lf,
            fs_sr=fs_sr, fs_lr=fs_lr, fs_sf=fs_sf, fs_lf=fs_lf,
        )

    @staticmethod
    def _build_timing_boundaries(timing, start_points, end_points, device):
        RISE_t, FALL_t, MAX = timing.Rise, timing.Fall, timing.Max

        def _arr(it, rf):   return timing.getPinArrival(it, rf, MAX)
        def _slw(it, rf):   return timing.getPinSlew(it, rf, MAX)
        def _slack(it, rf): return timing.getPinSlack(it, rf, MAX)

        sp_idx, sp_arr, sp_slw = [], [], []
        for sp in start_points:
            it = sp.Pin.db_iterm
            sp_idx.append(sp.idx)
            sp_arr.append([_arr(it, RISE_t), _arr(it, FALL_t)])
            sp_slw.append([_slw(it, RISE_t), _slw(it, FALL_t)])

        ep_idx, ep_arr, ep_slk = [], [], []
        for ep in end_points:
            it = ep.Pin.db_iterm
            ep_idx.append(ep.idx)
            ep_arr.append([_arr(it, RISE_t), _arr(it, FALL_t)])
            ep_slk.append([_slack(it, RISE_t), _slack(it, FALL_t)])

        start_idx_t   = torch.tensor(sp_idx,  dtype=torch.long,    device=device)
        start_arr_t   = torch.tensor(sp_arr,  dtype=torch.float32, device=device)
        start_slw_t   = torch.tensor(sp_slw,  dtype=torch.float32, device=device)
        end_idx_t     = torch.tensor(ep_idx,  dtype=torch.long,    device=device)
        end_arr_t     = torch.tensor(ep_arr,  dtype=torch.float32, device=device)
        end_slk_t     = torch.tensor(ep_slk,  dtype=torch.float32, device=device)
        require       = end_arr_t + end_slk_t  # required time = arrival + slack

        return start_idx_t, start_arr_t, start_slw_t, end_idx_t, require

    @staticmethod
    def _build_power_activity(steiner_net, timing, inpin_idx, outpin_idx, device):
        stps = steiner_net.stp_list
        N = len(stps)
        act_cache: Dict = {}

        def _act(it):
            return act_cache.setdefault(id(it), timing.getPinActivityDensity(it))

        in_act  = [_act(stps[i].Pin.db_iterm) for i in inpin_idx.cpu().tolist()]
        out_act = [_act(stps[i].Pin.db_iterm) for i in outpin_idx.cpu().tolist()]

        stp_act_list    = [0.0] * N
        no_driver_mask  = [1.0] * N
        driver_mask     = [0.0] * N
        for i, stp in enumerate(stps):
            if stp.Pin is not None and stp.Pin.io_type == "OUTPUT":
                no_driver_mask[i] = 0.0
                driver_mask[i]    = 1.0
            net = stp.net
            if net is not None and net.pins:
                stp_act_list[i] = _act(net.pins[0].db_iterm)

        return (
            torch.tensor(in_act,       dtype=torch.float32, device=device),
            torch.tensor(out_act,      dtype=torch.float32, device=device),
            torch.tensor(stp_act_list, dtype=torch.float32, device=device),
            torch.tensor(no_driver_mask, dtype=torch.long, device=device),
            torch.tensor(driver_mask,    dtype=torch.long, device=device),
        )

    @staticmethod
    def _build_pi_compute_indices(steiner_net, device):
        from placeopt.circuit.library import _is_driver_pin
        N = len(steiner_net.stp_list)
        driver_idxs = [
            i for i, stp in enumerate(steiner_net.stp_list)
            if (not stp.prevs) or (_is_driver_pin(stp.Pin) if stp.Pin else False)
        ]
        src = torch.tensor(driver_idxs, dtype=torch.long, device=device)
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[src] = False
        internal = mask.nonzero(as_tuple=False).squeeze(1)
        return src, internal


# ---------------------------------------------------------------------------
# Helper: pick smallest buffer and inverter
# ---------------------------------------------------------------------------

def _pick_smallest_buf_inv(cell_lib, timing, corner0):
    chosen_buf, chosen_inv = None, None
    best_buf_area = best_inv_area = 1e20
    best_buf_cap = -1.0
    best_inv_cap = 1e20

    for master in cell_lib.MasterVec:
        area = master.getHeight() * master.getWidth()
        a_in = master.findMTerm("A")
        if a_in is None:
            continue
        cap = timing.getPortCap(a_in, corner0, timing.Max)
        if is_buffer(master):
            if area < best_buf_area or (area == best_buf_area and cap > best_buf_cap):
                best_buf_area = area
                best_buf_cap  = cap
                chosen_buf    = master
        if is_inverter(master):
            if area < best_inv_area or (area == best_inv_area and cap < best_inv_cap):
                best_inv_area = area
                best_inv_cap  = cap
                chosen_inv    = master

    assert chosen_buf is not None, "No buffer master found in cell library."
    assert chosen_inv is not None, "No inverter master found in cell library."
    print(f"[INFO] buffer : {chosen_buf.getName()}, area={best_buf_area}")
    print(f"[INFO] inverter: {chosen_inv.getName()}, area={best_inv_area}")
    return chosen_buf, chosen_inv


def _buf_pin_coords(master, device):
    def _mpin_centre(mterm):
        for mpin in mterm.getMPins():
            b = mpin.getBBox()
            return (b.xMin() + b.xMax()) / 2.0, (b.yMin() + b.yMax()) / 2.0
        return 0.0, 0.0

    in_x, in_y   = _mpin_centre(master.findMTerm("A"))
    out_x, out_y = _mpin_centre(master.findMTerm("Y"))
    xin  = torch.tensor([in_x, in_y],   dtype=torch.float32, device=device)
    xout = torch.tensor([out_x, out_y], dtype=torch.float32, device=device)
    xcen = (xin + xout) / 2.0
    return xin, xout, xcen
