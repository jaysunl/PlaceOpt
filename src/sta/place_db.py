"""
place_db.py — PlaceDB dataclass and PlaceDBFactory builder.

PlaceDB is the central tensor container for gradient-based timing-aware
placement optimization, modelled after the DREAMPlace PlaceDB concept
but extended with timing state (arrival, slew, pi moments).

PlaceDBFactory assembles a PlaceDB from an OpenROAD design object, a
steiner-tree manager, and a circuit library.
"""

from dataclasses import dataclass, field
import time
import torch
from typing import List, Tuple, Union, Dict, Any
from collections import deque

from src.util.helpers import isDriverPin
from src.util.buffer_ops import isBuffer, isInverter
from src.sta.arc_model import fit_delay_model, fit_slew_model
import openroad as ord


# ---------------------------------------------------------------------------
# Propagation schedule record types
# ---------------------------------------------------------------------------

@dataclass
class NetSweepRecord:
    """Represents one level of net (wire) timing propagation."""
    type: str = "NET"
    level_idx: int = 0
    p_tensor: torch.Tensor = None   # [M] parent node indices
    c_tensor: torch.Tensor = None   # [M] child node indices


@dataclass
class CellSweepRecord:
    """Represents one level of cell (gate) arc timing propagation."""
    type: str = "CELL"
    level_idx: int = 0
    p_e: torch.Tensor = None            # [E] parent stp indices per arc entry
    c_e: torch.Tensor = None            # [E] child stp indices per arc entry
    u_index_group: torch.Tensor = None  # [Group] -> U_flat index
    group_tensor: torch.Tensor = None   # [E] -> group ID
    c_group: torch.Tensor = None        # [Group] -> output stp index


# ---------------------------------------------------------------------------
# PlaceDB — central timing/placement tensor container
# ---------------------------------------------------------------------------

@dataclass
class PlaceDB:
    """Flat tensor database for differentiable timing-aware placement.

    Naming mirrors DREAMPlace conventions where applicable:
      cell_xy / U_flat — optimizable parameters (nn.Parameter)
      stp_*            — Steiner-point indexed arrays
      u_*              — sizing parameter indexed arrays
      execution_plan   — levelized propagation schedule
    """
    # -- Optimizable parameters --
    cell_xy: torch.nn.Parameter       # [G,2] gate lower-left coordinates
    U_flat:  torch.nn.Parameter       # [total_U] continuous sizing weights

    # -- Cell / STP geometry --
    stp_gate_mapping: torch.Tensor    # [N] stp -> gate index (-1 if not a gate pin)
    stp_offset: torch.Tensor          # [N,2] pin offset from gate origin
    stp_parent_idx: torch.Tensor      # [N] parent STP index
    stp_fixed_xy: torch.Tensor        # [N,2] initial STP coordinates
    pos_xy: torch.Tensor              # [N,2] current STP positions
    stp_pin_id: torch.Tensor          # [N] pin name id per STP
    gate_orient: torch.Tensor         # [G] orientation code
    gate_movable_mask: torch.Tensor   # [G] 1=movable, 0=fixed/macro

    # -- Position-to-sizing coupling --
    pos_entry_u_index: torch.Tensor   # [E] U_flat index per pin entry
    pos_entry_stp_idx: torch.Tensor   # [E] STP index per pin entry

    # -- Master cell library --
    u_master_id: torch.Tensor         # [total_U] master cell id
    master_w: torch.Tensor            # [M] master width
    master_h: torch.Tensor            # [M] master height
    cell_box: torch.Tensor            # [G,4] (x,y,w,h) for density

    # -- Placement boundary and blockages --
    boundary: torch.Tensor            # [2,2] [[xmin,ymin],[xmax,ymax]]
    constant_blockage_xy: torch.Tensor; constant_blockage_wh: torch.Tensor
    hard_blockage_xy: torch.Tensor;     hard_blockage_wh: torch.Tensor
    soft_blockage_xy: torch.Tensor;     soft_blockage_wh: torch.Tensor
    macro_xy: torch.Tensor;             macro_wh: torch.Tensor
    init_cell_idx: torch.Tensor        # [Gi] indices for displacement penalty
    init_cell_xy: torch.Tensor         # [Gi,2] initial positions

    # -- Net levelization --
    level_edges: List[Tuple[torch.Tensor, torch.Tensor]]

    # -- Gate arc LUTs --
    delay_table: torch.Tensor          # [A,7,7]
    slew_table: torch.Tensor           # [A,7,7]
    load_index: torch.Tensor           # [A,7]
    slew_index: torch.Tensor           # [A,7]
    r_driver: torch.Tensor             # [A]
    unateness: torch.Tensor            # [A]
    arc_index_map: Dict[Any, int]
    arc_lut_r: torch.Tensor            # [M,P,P] rise arc indices
    arc_lut_f: torch.Tensor            # [M,P,P] fall arc indices

    # -- Sizing index --
    u_gate_len: torch.Tensor           # [G] number of U entries per gate
    u_gate_off: torch.Tensor           # [G+1] cumulative offset into U_flat
    u_gate_id: torch.Tensor            # [total_U] gate index

    # -- Pin capacity index --
    inpin_idx: torch.Tensor            # [P_in] stp indices of input pins
    outpin_idx: torch.Tensor           # [P_out] stp indices of output pins
    pin_entry_u_index: torch.Tensor    # [E] U_flat index per pin-entry
    pin_entry_pin_id: torch.Tensor     # [E] pin name id per entry
    pin_name_id: torch.Tensor          # [P_in] pin name id
    master_pin_cap: torch.Tensor       # [M,P_names] input cap per master

    # -- Buffer insertion --
    buffering_tensor: torch.nn.Parameter  # [N] continuous buffer weights
    chosen_buffer: Any
    chosen_inverter: Any
    inv_in_cap: float
    buffer_wh: torch.Tensor
    buffer_delay_table: torch.Tensor   # [2,7,7] (rise/fall)
    buffer_slew_table: torch.Tensor    # [2,7,7]
    buffer_cap_axis: torch.Tensor      # [2,7]
    buffer_slew_axis: torch.Tensor     # [2,7]
    fd_sr: Any; fd_lr: Any; fd_sf: Any; fd_lf: Any
    fs_sr: Any; fs_lr: Any; fs_sf: Any; fs_lf: Any
    buf_in_cap: float
    xin_buf: torch.Tensor
    xout_buf: torch.Tensor
    xcen_buf: torch.Tensor

    # -- Timing constraints --
    start_idx: torch.Tensor            # [S] source STP indices
    start_arrival_init: torch.Tensor   # [S,2]
    start_slew_init: torch.Tensor      # [S,2]
    end_idx: torch.Tensor              # [E_t] endpoint STP indices
    require: torch.Tensor              # [E_t,2] required arrival times

    # -- Propagation schedule --
    execution_plan: List[Union[NetSweepRecord, CellSweepRecord]]

    # -- Activity for power --
    inpin_activity: torch.Tensor
    outpin_activity: torch.Tensor
    stp_activity: torch.Tensor
    no_driver_mask: torch.Tensor
    driver_mask: torch.Tensor

    # -- Pi-model node indices --
    pi_source_idxs: torch.Tensor
    pi_internal_idxs: torch.Tensor

    # -- Leakage --
    leakage_per_u: torch.Tensor        # [total_U] in Watts
    baseline_leakage_W: float


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _bbox_xywh(bbox):
    if bbox is None:
        return None
    return (float(bbox.xMin()), float(bbox.yMin()),
            float(bbox.xMax() - bbox.xMin()), float(bbox.yMax() - bbox.yMin()))


def _loc_xy(inst):
    for getter in ("getLocation", "getOrigin"):
        if hasattr(inst, getter):
            loc = getattr(inst, getter)()
            if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                return float(loc[0]), float(loc[1])
            for a, b in (("x", "y"), ("getX", "getY")):
                if hasattr(loc, a):
                    xa = getattr(loc, a); ya = getattr(loc, b)
                    return float(xa() if callable(xa) else xa), float(ya() if callable(ya) else ya)
    if hasattr(inst, "getBBox"):
        bb = inst.getBBox()
        return float(bb.xMin()), float(bb.yMin())
    return 0.0, 0.0


def _orient_str(orient):
    if orient is None: return "R0"
    if isinstance(orient, str): return orient
    if hasattr(orient, "name"): return str(orient.name)
    return str(orient)


_ORIENT_CODE = {"R0": 0, "R90": 1, "R180": 2, "R270": 3,
                "MX": 4, "MY": 5, "MXR90": 6, "MYR90": 7}


# ---------------------------------------------------------------------------
# Tensor-building functions (previously idx_tensor_setup.py)
# ---------------------------------------------------------------------------

def build_node_positions(stp_list, device=torch.device("cpu")):
    """Build STP coordinate and parent-index tensors."""
    pos, par = [], []
    for stp in stp_list:
        pos.append([float(stp.x), float(stp.y)])
        if isDriverPin(stp.Pin) or not stp.prevs:
            par.append(int(stp.idx))
        else:
            par.append(int(stp.prevs[0].idx))
    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
    par_t = torch.tensor(par, dtype=torch.long, device=device)
    return pos_t, par_t, pos_t  # pos_xy, stp_parent_idx, stp_fixed_xy


def build_placement_index(stt_mgr, circuitLib, u_gate_off, device=torch.device("cpu")):
    """Build cell placement tensors (cell_xy, gate masks, offsets, etc.)."""
    sig_gates = circuitLib.signal_gates
    masters   = circuitLib.masters
    stp_list  = stt_mgr.stpList
    N = len(stp_list); G = len(sig_gates)

    master_name_to_id = {m.getName(): i for i, m in enumerate(masters)}
    pin_name_to_id: Dict[str, int] = {}
    for gate in sig_gates:
        for pin in gate.Pins.values():
            if pin.name not in pin_name_to_id:
                pin_name_to_id[pin.name] = len(pin_name_to_id)

    gate_idx_of_stp = [-1] * N
    stp_off_list    = [[0.0, 0.0]] * N
    stp_pin_ids     = [-1] * N
    cell_xy_list    = [[0.0, 0.0]] * G
    gate_orient_l   = [0] * G
    gate_movable_l  = [1] * G
    mw = [0.0] * len(masters); mh = [0.0] * len(masters)
    for mi, m in enumerate(masters):
        mw[mi] = float(m.getWidth()); mh[mi] = float(m.getHeight())

    for gate in sig_gates:
        gi   = gate.idx
        if gi < 0: continue
        inst = gate.db_Inst
        mst  = getattr(gate, "db_master", None) or (inst.getMaster() if inst else None)
        is_blk = bool(getattr(mst, "isBlock", lambda: False)()) if mst else False
        gate_movable_l[gi] = 0 if is_blk else 1
        gate_orient_l[gi]  = _ORIENT_CODE.get(_orient_str(
            getattr(inst, "getOrient", lambda: None)()), 0)
        cell_xy_list[gi]   = list(_loc_xy(inst))

    u_master_ids: List[int] = []
    for gate in sig_gates:
        for m in gate.eqvMaster:
            u_master_ids.append(master_name_to_id.get(m.getName(), -1))

    u_off_l = u_gate_off.detach().cpu().tolist()
    pe_u_idx: List[int] = []
    pe_stp_idx: List[int] = []
    for stp in stp_list:
        pin  = getattr(stp, "Pin", None)
        gate = getattr(pin, "Gate", None) if pin else None
        gi   = getattr(gate, "idx", -1)
        gate_idx_of_stp[int(stp.idx)] = int(gi)
        if gi < 0:
            stp_off_list[int(stp.idx)] = [float(stp.x), float(stp.y)]
            continue
        bx, by = cell_xy_list[gi]
        stp_off_list[int(stp.idx)] = [float(stp.x) - bx, float(stp.y) - by]
        pn = stp.Pin.name
        stp_pin_ids[int(stp.idx)] = pin_name_to_id.get(pn, -1)
        for mi in range(len(gate.eqvMaster)):
            pe_u_idx.append(int(u_off_l[gi]) + mi)
            pe_stp_idx.append(int(stp.idx))

    def _t(data, dtype): return torch.tensor(data, dtype=dtype, device=device)
    cell_xy = torch.nn.Parameter(_t(cell_xy_list, torch.float32))
    return (
        cell_xy,
        _t(gate_idx_of_stp, torch.long),
        _t(stp_off_list,    torch.float32),
        _t(pe_u_idx,        torch.long) if pe_u_idx else torch.zeros((0,), dtype=torch.long, device=device),
        _t(pe_stp_idx,      torch.long) if pe_stp_idx else torch.zeros((0,), dtype=torch.long, device=device),
        _t(stp_pin_ids,     torch.long),
        _t(u_master_ids,    torch.long),
        _t(mw, torch.float32),
        _t(mh, torch.float32),
        _t(gate_orient_l,   torch.long),
        _t(gate_movable_l,  torch.float32),
        pin_name_to_id,
    )


def levelize_net_graph(nets, device=torch.device("cpu")):
    """Topological sort of net Steiner points (backward BFS from sinks)."""
    stps = []
    for net in nets:
        if hasattr(net, "steinerPoints") and net.steinerPoints:
            stps.extend(s for s in net.steinerPoints if s is not None)
    if not stps:
        return []

    N   = max(int(s.idx) for s in stps) + 1
    adj = [[] for _ in range(N)]
    par = [[] for _ in range(N)]
    deg = [0] * N

    for stp in stps:
        u    = int(stp.idx)
        u_db = getattr(stp, "Net", None)
        if u_db is None: continue
        u_name = None
        seen   = set()
        for ch in getattr(stp, "nexts", []):
            if ch is None: continue
            ch_net = getattr(ch, "Net", None)
            if ch_net is None: continue
            if ch_net is not u_db:
                if u_name is None: u_name = u_db.db_net.getName()
                if ch_net.db_net.getName() != u_name: continue
            v = int(ch.idx)
            if v in seen: continue
            seen.add(v)
            adj[u].append(v); par[v].append(u)

    for u in range(N): deg[u] = len(adj[u])
    q = deque(i for i in range(N) if deg[i] == 0)
    levels: List[List[int]] = []
    while q:
        cur = []
        for _ in range(len(q)):
            c = q.popleft(); cur.append(c)
            for p in par[c]:
                deg[p] -= 1
                if deg[p] == 0: q.append(p)
        if cur: levels.append(cur)

    placed = sum(len(lv) for lv in levels)
    if placed != N:
        seen_s = set(i for lv in levels for i in lv)
        left   = [i for i in range(N) if i not in seen_s]
        if left: levels.append(left)

    fp, fc, sz = [], [], []
    for lv in levels:
        cnt = 0
        for c in lv:
            ps = par[c]
            if ps:
                fp.extend(ps); fc.extend([c] * len(ps)); cnt += len(ps)
        sz.append(cnt)

    if not fp:
        return [(torch.tensor([], dtype=torch.long, device=device),
                 torch.tensor([], dtype=torch.long, device=device))
                for _ in levels]

    ap = torch.tensor(fp, dtype=torch.long, device=device)
    ac = torch.tensor(fc, dtype=torch.long, device=device)
    return list(zip(torch.split(ap, sz), torch.split(ac, sz)))


def pack_arc_luts(circuitLib, device=torch.device("cpu")):
    """Stack all liberty arc tables into stacked tensors."""
    dtabs, stabs, laxes, saxes, rds, unat = [], [], [], [], [], []
    arc_map: Dict[Any, int] = {}
    cnt = 0
    for master in circuitLib.masters:
        for tbl in circuitLib.get_luts(master).values():
            key = (tbl.in_pin_name, tbl.out_pin_name, tbl.out_rf)
            arc_map[(master.getName(), key)] = cnt
            dt = tbl.delay_table; st = tbl.slew_table
            if dt.shape != (7, 7):
                dt = dt[0, 0].view(1, 1).expand(7, 7)
                st = st[0, 0].view(1, 1).expand(7, 7)
            dtabs.append(dt); stabs.append(st)
            def _pad7(ax):
                return (torch.linspace(ax[0].item(), ax[-1].item(), 7, dtype=ax.dtype, device=ax.device)
                        if ax.numel() != 7 else ax)
            laxes.append(_pad7(tbl.axis_1)); saxes.append(_pad7(tbl.axis_0))
            rds.append(tbl.driver_rd.reshape(()))
            unat.append(1 if tbl.out_rf != tbl.in_rf else 0)
            cnt += 1
    return (torch.stack(dtabs).to(device),  torch.stack(stabs).to(device),
            torch.stack(laxes).to(device),  torch.stack(saxes).to(device),
            torch.stack(rds).to(device),    torch.tensor(unat, dtype=torch.long, device=device),
            arc_map)


def build_arc_dispatch_table(arc_map, masters, pin_name_to_id, device=torch.device("cpu")):
    """Build [M, P, P] arc index tables for rise and fall."""
    m2id = {m.getName(): i for i, m in enumerate(masters)}
    P    = len(pin_name_to_id)
    lut_r = torch.full((len(masters), P, P), -1, dtype=torch.long, device=device)
    lut_f = torch.full((len(masters), P, P), -1, dtype=torch.long, device=device)
    for (mn, (fp, tp, rf)), idx in arc_map.items():
        mi = m2id.get(mn);        fi = pin_name_to_id.get(fp); ti = pin_name_to_id.get(tp)
        if None in (mi, fi, ti): continue
        (lut_r if rf == "^" else lut_f)[mi, fi, ti] = idx
    return lut_r, lut_f


def build_size_index(signal_gates, stp_list, timing, masters,
                     origin_weight=6, device=torch.device("cpu")):
    """Build U_flat sizing parameter tensor and pin index maps."""
    G       = len(signal_gates)
    glenl   = [len(g.eqvMaster) for g in signal_gates]
    glen    = torch.tensor(glenl, dtype=torch.long, device=device)
    goff    = torch.zeros(G + 1, dtype=torch.long, device=device)
    goff[1:] = torch.cumsum(glen, dim=0)
    u_gid   = torch.repeat_interleave(torch.arange(G, dtype=torch.long, device=device), glen)

    u_init = []
    for cnt in glenl:
        if cnt > 0:
            u_init.append(float(origin_weight))
            u_init.extend([1.0] * (cnt - 1))
    U_flat = torch.nn.Parameter(torch.tensor(u_init, dtype=torch.float32, device=device))

    ip_idx, op_idx, ig_idx, ip_names = [], [], [], []
    for gi, g in enumerate(signal_gates):
        for pin in g.inputPins:
            stp = getattr(pin, "steinerPoint", None)
            if stp is None: continue
            ip_idx.append(stp.idx); ig_idx.append(gi); ip_names.append(pin.name)
        for pin in g.outputPins:
            stp = getattr(pin, "steinerPoint", None)
            if stp is None: continue
            op_idx.append(stp.idx)

    ip_t  = torch.tensor(ip_idx, dtype=torch.long, device=device)
    op_t  = torch.tensor(op_idx, dtype=torch.long, device=device)
    pg_t  = torch.tensor(ig_idx, dtype=torch.long, device=device)

    p2id: Dict[str, int] = {}
    pid_list = []
    for nm in ip_names:
        if nm not in p2id: p2id[nm] = len(p2id)
        pid_list.append(p2id[nm])
    pid_t = torch.tensor(pid_list, dtype=torch.long, device=device)

    P    = len(ip_idx)
    plen = glen[pg_t]
    poff = torch.zeros(P + 1, dtype=torch.long, device=device)
    poff[1:] = torch.cumsum(plen, dim=0)
    total = int(poff[-1].item())
    pe_pin_id = torch.repeat_interleave(torch.arange(P, dtype=torch.long, device=device), plen)
    pos_e     = torch.arange(total, dtype=torch.long, device=device)
    local     = pos_e - poff[pe_pin_id]
    gfe       = pg_t[pe_pin_id]
    pe_u_idx  = goff[gfe] + local

    corner0 = timing.getCorners()[0]
    mx_mode = timing.Max
    id2n    = [None] * len(p2id)
    for nm, i in p2id.items(): id2n[i] = nm
    mpc = [[0.0] * len(p2id) for _ in range(len(masters))]
    cache: Dict = {}
    for mi, m in enumerate(masters):
        mn = m.getName()
        for pi, pn in enumerate(id2n):
            ck = (mn, pn)
            if ck not in cache:
                mt   = m.findMTerm(pn)
                cache[ck] = 0.0 if mt is None else timing.getPortCap(mt, corner0, mx_mode)
            mpc[mi][pi] = cache[ck]
    mpc_t = torch.tensor(mpc, dtype=torch.float32, device=device)

    return (U_flat, glen, goff, u_gid, ip_t, op_t, pe_u_idx, pe_pin_id, pid_t, mpc_t)


def extract_constraints(timing, start_pts, end_pts, device=torch.device("cpu")):
    """Extract arrival times and required times from OpenSTA."""
    R, F, MX = timing.Rise, timing.Fall, timing.Max
    arr_c, slw_c, slk_c = {}, {}, {}

    def _arr(it, rf):
        k = (id(it), rf)
        if k not in arr_c: arr_c[k] = timing.getPinArrival(it, rf, MX)
        return arr_c[k]

    def _slw(it, rf):
        k = (id(it), rf)
        if k not in slw_c: slw_c[k] = timing.getPinSlew(it, rf, MX)
        return slw_c[k]

    def _slk(it, rf):
        k = (id(it), rf)
        if k not in slk_c: slk_c[k] = timing.getPinSlack(it, rf, MX)
        return slk_c[k]

    si, sa, ss = [], [], []
    for sp in start_pts:
        it = sp.Pin.db_ITerm
        si.append(sp.idx)
        sa.append([_arr(it, R), _arr(it, F)])
        ss.append([_slw(it, R), _slw(it, F)])

    ei, ea, ek = [], [], []
    for ep in end_pts:
        it = ep.Pin.db_ITerm
        ei.append(ep.idx)
        ea.append([_arr(it, R), _arr(it, F)])
        ek.append([_slk(it, R), _slk(it, F)])

    ea_t = torch.tensor(ea, dtype=torch.float32, device=device)
    ek_t = torch.tensor(ek, dtype=torch.float32, device=device)
    return (torch.tensor(si, dtype=torch.long,    device=device),
            torch.tensor(sa, dtype=torch.float32, device=device),
            torch.tensor(ss, dtype=torch.float32, device=device),
            torch.tensor(ei, dtype=torch.long,    device=device),
            ea_t + ek_t)


def build_activity_index(stt_mgr, timing, ip_t, op_t, device=torch.device("cpu")):
    """Build pin switching activity tensors."""
    stpl = stt_mgr.stpList
    cache: Dict = {}

    def _act(it):
        k = id(it)
        if k not in cache: cache[k] = timing.getPinActivityDensity(it)
        return cache[k]

    act_ip = [_act(stpl[i].Pin.db_ITerm) for i in ip_t.cpu().tolist()]
    act_op = [_act(stpl[i].Pin.db_ITerm) for i in op_t.cpu().tolist()]
    N = len(stpl)
    act_stp = [0.0] * N; no_drv = [1.0] * N; drv = [0.0] * N
    for i, stp in enumerate(stpl):
        if stp.Pin is not None and stp.Pin.IO == "OUTPUT":
            no_drv[i] = 0; drv[i] = 1
        p = stp.Net.Pins[0]
        act_stp[i] = _act(p.db_ITerm)
    return (torch.tensor(act_ip,  dtype=torch.float32, device=device),
            torch.tensor(act_op,  dtype=torch.float32, device=device),
            torch.tensor(act_stp, dtype=torch.float32, device=device),
            torch.tensor(no_drv,  dtype=torch.long,    device=device),
            torch.tensor(drv,     dtype=torch.long,    device=device))


def build_buffer_model(stt_mgr, circuitLib, timing, init_value=-8.0, device=torch.device("cpu")):
    """Select smallest buffer/inverter and build their analytical timing models."""
    buf_w = torch.nn.Parameter(
        torch.full((len(stt_mgr.stpList),), init_value, dtype=torch.float32, device=device))

    corner0 = timing.getCorners()[0]
    best_buf = None; best_area = 1e20; best_cap = 1e-20
    for m in circuitLib.masters:
        if not isBuffer(m): continue
        ar = m.getHeight() * m.getWidth()
        cp = timing.getPortCap(m.findMTerm("A"), corner0, timing.Max)
        if ar < best_area or (ar == best_area and cp > best_cap):
            best_area = ar; best_cap = cp; best_buf = m

    buf_in_cap = timing.getPortCap(best_buf.findMTerm("A"), corner0, timing.Max)
    print(f"[PlaceDB] Buffer: {best_buf.getName()} cap={buf_in_cap:.3e} F area={best_area:.0f}")

    bwh = torch.tensor([float(best_buf.getWidth()), float(best_buf.getHeight())],
                       dtype=torch.float32, device=device)
    tbls = circuitLib.get_luts(best_buf)
    dt_buf = torch.zeros((2, 7, 7), dtype=torch.float32, device=device)
    st_buf = torch.zeros((2, 7, 7), dtype=torch.float32, device=device)
    ca_buf = torch.zeros((2, 7),    dtype=torch.float32, device=device)
    sa_buf = torch.zeros((2, 7),    dtype=torch.float32, device=device)
    dm  = [None, None]; sm_l = [None, None]
    for tbl in tbls.values():
        idx = 0 if tbl.out_rf == "^" else 1
        dt_buf[idx] = tbl.delay_table.to(device=device, dtype=torch.float32)
        st_buf[idx] = tbl.slew_table.to(device=device,  dtype=torch.float32)
        ca_buf[idx] = tbl.axis_1.to(device=device,      dtype=torch.float32)
        sa_buf[idx] = tbl.axis_0.to(device=device,      dtype=torch.float32)
        if dm[idx] is None:
            with torch.no_grad():
                dm[idx]   = fit_delay_model(tbl.delay_table, tbl.axis_0, tbl.axis_1)
                sm_l[idx] = fit_slew_model(tbl.slew_table,  tbl.axis_0, tbl.axis_1)

    _DEFAULTS = {
        "dr": [5.2228e-12, 4.9731e-09, 4.6879e-12, 2.3443e-11, -4.2855e-12],
        "df": [6.2871e-12, 4.1882e-09, 2.7307e-11, 2.1773e-11, -5.9683e-12],
        "sr": [3.8538e-12, 1.0859e-08, 7.2963e-12, 8.7341e-13, -1.2711e-12],
        "sf": [3.8091e-12, 8.4023e-09, 9.4455e-12, 1.0353e-12, -2.3475e-12],
    }
    if any(x is None for x in (*dm, *sm_l)):
        print("[PlaceDB] WARN: regression unavailable for some arcs; using defaults.")

    def _coeff(model, key):
        if model is None: return torch.tensor(_DEFAULTS[key], dtype=torch.float32, device=device)
        return torch.as_tensor(model["coeffs"], dtype=torch.float32, device=device)

    dr_c = _coeff(dm[0], "dr");   df_c = _coeff(dm[1], "df")
    sr_c = _coeff(sm_l[0], "sr"); sf_c = _coeff(sm_l[1], "sf")

    def _ssc(model, ax): return torch.tensor(
        model.get("s_scale", ax.abs().max().clamp_min(1e-12).item()),
        dtype=torch.float32, device=device) if model else ax.abs().max().clamp_min(1e-12)

    def _csc(model, ax): return torch.tensor(
        model.get("c_scale", ax.abs().max().clamp_min(1e-12).item()),
        dtype=torch.float32, device=device) if model else ax.abs().max().clamp_min(1e-12)

    ss_r = _ssc(dm[0], sa_buf[0]); cs_r = _csc(dm[0], ca_buf[0])
    ss_f = _ssc(dm[1], sa_buf[1]); cs_f = _csc(dm[1], ca_buf[1])

    def _mkfn(co, ss, cs):
        def _fs(slew):
            c = co.to(device=slew.device, dtype=slew.dtype)
            sn = slew / ss; sc = torch.sign(sn) * (torch.abs(sn) + 1e-10).pow(1/3)
            return c[2] * sn + c[3] * sc + c[4] * sn * sn
        def _fl(load):
            c = co.to(device=load.device, dtype=load.dtype)
            return c[0] + c[1] * load / cs
        return _fs, _fl

    fd_sr, fd_lr = _mkfn(dr_c, ss_r, cs_r)
    fd_sf, fd_lf = _mkfn(df_c, ss_f, cs_f)
    fs_sr, fs_lr = _mkfn(sr_c, ss_r, cs_r)
    fs_sf, fs_lf = _mkfn(sf_c, ss_f, cs_f)

    it  = best_buf.findMTerm("A"); ot = best_buf.findMTerm("Y")
    for mp in it.getMPins():
        bb = mp.getBBox(); cinx = (bb.xMin() + bb.xMax()) / 2; ciny = (bb.yMin() + bb.yMax()) / 2
    for mp in ot.getMPins():
        bb = mp.getBBox(); coutx = (bb.xMin() + bb.xMax()) / 2; couty = (bb.yMin() + bb.yMax()) / 2

    xin  = torch.tensor([cinx,  ciny],  dtype=torch.float32, device=device)
    xout = torch.tensor([coutx, couty], dtype=torch.float32, device=device)
    xcen = torch.tensor([(cinx + coutx) / 2, (ciny + couty) / 2], dtype=torch.float32, device=device)

    best_inv = None; inv_area = 1e20; inv_cap = 1e20
    for m in circuitLib.masters:
        if not isInverter(m): continue
        ar = m.getHeight() * m.getWidth()
        cp = timing.getPortCap(m.findMTerm("A"), corner0, timing.Max)
        if ar < inv_area or (ar == inv_area and cp < inv_cap):
            inv_area = ar; inv_cap = cp; best_inv = m
    inv_in_cap = timing.getPortCap(best_inv.findMTerm("A"), corner0, timing.Max)
    print(f"[PlaceDB] Inverter: {best_inv.getName()} cap={inv_in_cap:.3e} F")

    return (buf_w, best_buf, bwh, dt_buf, st_buf, ca_buf, sa_buf,
            best_inv, inv_in_cap,
            fd_sr, fd_lr, fd_sf, fd_lf, fs_sr, fs_lr, fs_sf, fs_lf,
            buf_in_cap, xin, xout, xcen)


def compile_sweep_schedule(stt_mgr, u_gate_off, device=torch.device("cpu")):
    """Compile the levelized net/cell propagation sweep schedule."""
    plan: List[Union[NetSweepRecord, CellSweepRecord]] = []
    stpl = stt_mgr.stpList

    for li, layer in enumerate(stt_mgr.levelizedNetwork):
        if len(layer) == 0:
            plan.append(NetSweepRecord(
                level_idx=li,
                p_tensor=torch.zeros(1, dtype=torch.long, device=device),
                c_tensor=torch.zeros(1, dtype=torch.long, device=device),
            ))
            continue

        s0 = layer[0]
        is_cell = s0.Pin is not None and s0.Pin.IO == "OUTPUT"

        if is_cell:
            ps, cs, gs = [], [], []
            for stp in layer:
                if stp.prevs and (stp.prevs[0].Pin.Gate.db_Inst.getName()
                                  == stp.Pin.Gate.db_Inst.getName()):
                    for prev in stp.prevs:
                        ps.append(prev.idx); cs.append(stp.idx); gs.append(stp.Pin.Gate.idx)
            p_t = torch.tensor(ps, dtype=torch.long, device=device)
            c_t = torch.tensor(cs, dtype=torch.long, device=device)
            g_t = torch.tensor(gs, dtype=torch.long, device=device)
            E   = len(cs)
            al  = torch.tensor([len(stpl[c].Pin.Gate.eqvMaster) for c in cs],
                               dtype=torch.long, device=device)
            ao  = torch.zeros(E + 1, dtype=torch.long, device=device)
            ao[1:] = torch.cumsum(al, dim=0)
            T   = int(ao[-1].item())
            ea  = torch.repeat_interleave(torch.arange(E, device=device), al)
            elo = torch.arange(T, device=device) - ao[ea]
            pe  = p_t[ea]; ce  = c_t[ea]; ge  = g_t[ea]
            ui  = u_gate_off[ge] + elo
            gl, ul = ge.tolist(), ui.tolist()
            grp: Dict = {}
            for i in range(len(gl)):
                k = (gl[i], ul[i])
                if k not in grp: grp[k] = []
                grp[k].append(i)
            grp_id = [0] * len(gl); uig = []
            for cnt, (k, idxs) in enumerate(grp.items()):
                uig.append(k[1])
                for ii in idxs: grp_id[ii] = cnt
            grp_t = torch.tensor(grp_id, dtype=torch.long, device=device)
            uig_t = torch.tensor(uig,    dtype=torch.long, device=device)
            cg    = torch.full((len(uig),), -1, dtype=torch.long, device=device)
            cg.scatter_reduce_(0, grp_t, ce, reduce="amax", include_self=True)
            plan.append(CellSweepRecord(
                level_idx=li, p_e=pe, c_e=ce,
                u_index_group=uig_t, group_tensor=grp_t, c_group=cg,
            ))
        else:
            ps, cs = [], []
            for stp in layer:
                if stp.prevs and (stp.prevs[0].Net.db_net.getName()
                                  == stp.Net.db_net.getName()):
                    ps.append(stp.prevs[0].idx); cs.append(stp.idx)
            plan.append(NetSweepRecord(
                level_idx=li,
                p_tensor=torch.tensor(ps, dtype=torch.long, device=device),
                c_tensor=torch.tensor(cs, dtype=torch.long, device=device),
            ))
    return plan


def build_driver_index(stt_mgr, device=torch.device("cpu")):
    """Identify source (driver) and internal STP node index sets."""
    N      = len(stt_mgr.stpList)
    drv    = [i for i, stp in enumerate(stt_mgr.stpList)
              if (not getattr(stp, "prevs", None)) or isDriverPin(getattr(stp, "Pin", None))]
    src_t  = torch.tensor(drv, dtype=torch.long, device=device)
    mask   = torch.ones(N, dtype=torch.bool, device=device)
    mask[src_t] = False
    int_t  = mask.nonzero(as_tuple=False).squeeze(1)
    return src_t, int_t


def build_leakage_index(signal_gates, timing, device=torch.device("cpu")):
    """Build leakage power tensor aligned with U_flat."""
    corner = timing.getCorners()[0]
    lk_cache: Dict[str, float] = {}
    for g in signal_gates:
        mn = g.db_Inst.getMaster().getName()
        if mn not in lk_cache:
            lk_cache[mn] = timing.staticPower(g.db_Inst, corner)
    per_u = []
    for g in signal_gates:
        ref_mn  = g.db_Inst.getMaster().getName()
        ref_lk  = lk_cache.get(ref_mn, 0.0)
        ref_ar  = g.db_Inst.getMaster().getWidth() * g.db_Inst.getMaster().getHeight()
        for m in g.eqvMaster:
            mn = m.getName()
            if mn in lk_cache:
                per_u.append(lk_cache[mn])
            elif ref_ar > 0:
                per_u.append(ref_lk * m.getWidth() * m.getHeight() / ref_ar)
            else:
                per_u.append(ref_lk)
    lk_t = torch.tensor(per_u, dtype=torch.float32, device=device)
    base  = sum(lk_cache.get(g.db_Inst.getMaster().getName(), 0.0) for g in signal_gates)
    return lk_t, base


# ---------------------------------------------------------------------------
# PlaceDB factory
# ---------------------------------------------------------------------------

class PlaceDBFactory:
    """Assembles a PlaceDB from an OpenROAD design, steiner-tree manager, and circuit library."""

    def build(self, origin_gate_w, origin_buf_w, stt_mgr, circuitLib, timing, device) -> PlaceDB:
        print("[PlaceDB] Assembling timing/placement database...")
        t0 = time.perf_counter()
        steps: Dict[str, float] = {}

        design = circuitLib.design
        block  = design.getBlock()
        bbox   = block.getBBox()
        boundary = torch.tensor(
            [[bbox.xMin(), bbox.yMin()], [bbox.xMax(), bbox.yMax()]],
            dtype=torch.float32, device=device)

        hard_xy, hard_wh, soft_xy, soft_wh = [], [], [], []
        for blk in block.getBlockages():
            r = _bbox_xywh(blk.getBBox())
            if r is None: continue
            x0, y0, w, h = r
            if bool(getattr(blk, "isSoft", lambda: False)()):
                soft_xy.append([x0, y0]); soft_wh.append([w, h])
            else:
                hard_xy.append([x0, y0]); hard_wh.append([w, h])

        def _tens(lst): return (torch.tensor(lst, dtype=torch.float32, device=device)
                                if lst else torch.zeros((0, 2), device=device))
        hard_bxy, hard_bwh = _tens(hard_xy), _tens(hard_wh)
        soft_bxy, soft_bwh = _tens(soft_xy), _tens(soft_wh)

        mac_xy, mac_wh = [], []
        for inst in block.getInsts():
            m = inst.getMaster()
            if m is None or not m.isBlock(): continue
            r = _bbox_xywh(inst.getBBox())
            if r is None: continue
            x0, y0, w, h = r
            mac_xy.append([x0, y0]); mac_wh.append([w, h])
        macro_xy_t = _tens(mac_xy); macro_wh_t = _tens(mac_wh)

        if hard_xy or mac_xy:
            cb_xy = torch.tensor(hard_xy + mac_xy, dtype=torch.float32, device=device)
            cb_wh = torch.tensor(hard_wh + mac_wh, dtype=torch.float32, device=device)
        else:
            cb_xy = torch.zeros((0, 2), device=device)
            cb_wh = torch.zeros((0, 2), device=device)

        ts = time.perf_counter()
        pos_xy, stp_par, stp_fix = build_node_positions(stt_mgr.stpList, device=device)
        steps["node_positions"] = time.perf_counter() - ts

        ts = time.perf_counter()
        dt, st, li, si, rd, un, aim = pack_arc_luts(circuitLib, device=device)
        steps["arc_luts"] = time.perf_counter() - ts

        ts = time.perf_counter()
        level_edges = levelize_net_graph(circuitLib.signal_nets, device=device)
        steps["net_graph"] = time.perf_counter() - ts

        ts = time.perf_counter()
        (U_flat, glen, goff, ugid, ip_t, op_t,
         peu_t, pep_t, pn_t, mpc_t) = build_size_index(
            circuitLib.signal_gates, stt_mgr.stpList, timing,
            circuitLib.masters, origin_weight=origin_gate_w, device=device)
        steps["size_index"] = time.perf_counter() - ts

        ts = time.perf_counter()
        (cell_xy, stp_gmap, stp_off, pe_ui, pe_si,
         stp_pid, u_mid, mw, mh, gor, gmo, p2id) = build_placement_index(
            stt_mgr, circuitLib, goff, device=device)
        steps["placement_index"] = time.perf_counter() - ts
        cell_box = cell_xy.new_zeros((cell_xy.shape[0], 4))

        ts = time.perf_counter()
        arc_lut_r, arc_lut_f = build_arc_dispatch_table(aim, circuitLib.masters, p2id, device=device)
        steps["arc_dispatch"] = time.perf_counter() - ts

        ts = time.perf_counter()
        (buf_w, cho_buf, bwh, dt_buf, st_buf, ca_buf, sa_buf,
         cho_inv, inv_cap,
         fd_sr, fd_lr, fd_sf, fd_lf,
         fs_sr, fs_lr, fs_sf, fs_lf,
         buf_ic, xin, xout, xcen) = build_buffer_model(
            stt_mgr, circuitLib, timing, init_value=origin_buf_w, device=device)
        steps["buffer_model"] = time.perf_counter() - ts

        ts = time.perf_counter()
        si_t, sa_t, ss_t, ei_t, req = extract_constraints(
            timing, stt_mgr.startPoints, stt_mgr.endPoints, device=device)
        steps["constraints"] = time.perf_counter() - ts

        ts = time.perf_counter()
        sweep = compile_sweep_schedule(stt_mgr, goff, device=device)
        steps["sweep_plan"] = time.perf_counter() - ts

        ts = time.perf_counter()
        ia_t, oa_t, sa_act, ndm, dm_t = build_activity_index(
            stt_mgr, timing, ip_t, op_t, device=device)
        steps["activity_index"] = time.perf_counter() - ts

        ts = time.perf_counter()
        src_t, int_t = build_driver_index(stt_mgr, device=device)
        steps["driver_index"] = time.perf_counter() - ts

        ts = time.perf_counter()
        lk_t, lk_base = build_leakage_index(circuitLib.signal_gates, timing, device=device)
        steps["leakage_index"] = time.perf_counter() - ts

        for nm, el in steps.items():
            print(f"  -- {nm:<22} {el:.3f}s")
        print(f"[PlaceDB] Done in {time.perf_counter() - t0:.2f}s")

        ic_idx = torch.tensor(getattr(circuitLib, "init_cell_idxs", []),
                              dtype=torch.long, device=device)
        ic_xy  = torch.tensor(getattr(circuitLib, "init_pos", []),
                              dtype=torch.float32, device=device)

        return PlaceDB(
            cell_xy=cell_xy, U_flat=U_flat,
            stp_gate_mapping=stp_gmap, stp_offset=stp_off,
            stp_parent_idx=stp_par, stp_fixed_xy=stp_fix,
            pos_xy=pos_xy, stp_pin_id=stp_pid,
            gate_orient=gor, gate_movable_mask=gmo,
            pos_entry_u_index=pe_ui, pos_entry_stp_idx=pe_si,
            u_master_id=u_mid, master_w=mw, master_h=mh,
            cell_box=cell_box,
            boundary=boundary,
            constant_blockage_xy=cb_xy, constant_blockage_wh=cb_wh,
            hard_blockage_xy=hard_bxy, hard_blockage_wh=hard_bwh,
            soft_blockage_xy=soft_bxy, soft_blockage_wh=soft_bwh,
            macro_xy=macro_xy_t, macro_wh=macro_wh_t,
            init_cell_idx=ic_idx, init_cell_xy=ic_xy,
            level_edges=level_edges,
            delay_table=dt, slew_table=st,
            load_index=li, slew_index=si,
            r_driver=rd, unateness=un, arc_index_map=aim,
            arc_lut_r=arc_lut_r, arc_lut_f=arc_lut_f,
            u_gate_len=glen, u_gate_off=goff, u_gate_id=ugid,
            inpin_idx=ip_t, outpin_idx=op_t,
            pin_entry_u_index=peu_t, pin_entry_pin_id=pep_t,
            pin_name_id=pn_t, master_pin_cap=mpc_t,
            buffering_tensor=buf_w, chosen_buffer=cho_buf,
            chosen_inverter=cho_inv, inv_in_cap=inv_cap,
            buffer_wh=bwh,
            buffer_delay_table=dt_buf, buffer_slew_table=st_buf,
            buffer_cap_axis=ca_buf, buffer_slew_axis=sa_buf,
            fd_sr=fd_sr, fd_lr=fd_lr, fd_sf=fd_sf, fd_lf=fd_lf,
            fs_sr=fs_sr, fs_lr=fs_lr, fs_sf=fs_sf, fs_lf=fs_lf,
            buf_in_cap=buf_ic, xin_buf=xin, xout_buf=xout, xcen_buf=xcen,
            start_idx=si_t, start_arrival_init=sa_t, start_slew_init=ss_t,
            end_idx=ei_t, require=req,
            execution_plan=sweep,
            inpin_activity=ia_t, outpin_activity=oa_t,
            stp_activity=sa_act, no_driver_mask=ndm, driver_mask=dm_t,
            pi_source_idxs=src_t, pi_internal_idxs=int_t,
            leakage_per_u=lk_t, baseline_leakage_W=lk_base,
        )
