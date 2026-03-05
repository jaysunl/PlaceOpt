"""
timing.schedule — Execution plan builder and task dataclasses.

The execution plan is a flat list of ``WireTask`` and ``GateTask`` objects
that the timing engine processes in order during its forward pass.

  * ``WireTask`` — propagate arrival/slew along wire segments within one net.
  * ``GateTask`` — propagate arrival/slew through gate timing arcs.

Building the plan once (at graph-construction time) and caching it avoids
repeated Python-level graph traversal during the hot optimization loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Task dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WireTask:
    """
    Wire propagation step: parent Steiner points → child Steiner points.

    Attributes
    ----------
    level_idx : level index in the circuit DAG.
    p_tensor  : [M] parent STP indices.
    c_tensor  : [M] child  STP indices (one-to-one with p_tensor).
    type      : always "WIRE" (used as a discriminant).
    """
    level_idx: int
    p_tensor:  torch.Tensor
    c_tensor:  torch.Tensor
    type: str = "WIRE"


@dataclass
class GateTask:
    """
    Gate arc propagation step: input pins → output pins of the same cell.

    Attributes
    ----------
    level_idx    : level index in the circuit DAG.
    p_e          : [T] STP indices of arc input pins  (expanded over U variants).
    c_e          : [T] STP indices of arc output pins (expanded over U variants).
    u_index_group: [Group] unique U_flat indices, one per (gate, master) group.
    group_tensor : [T] maps each expanded entry to its group index.
    c_group      : [Group] output-pin STP index for each group.
    type         : always "GATE".
    """
    level_idx:     int
    p_e:           torch.Tensor
    c_e:           torch.Tensor
    u_index_group: torch.Tensor
    group_tensor:  torch.Tensor
    c_group:       torch.Tensor
    type: str = "GATE"


# ---------------------------------------------------------------------------
# Net-traverse schedule (backward moment pass)
# ---------------------------------------------------------------------------

def build_net_traverse_schedule(
    signal_nets,
    device: torch.device = torch.device("cpu"),
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build a backward-leveled edge list for propagating Elmore moments.

    Moments are computed from sinks toward the driver (reverse BFS), so this
    function returns levels in that reverse order.

    Returns
    -------
    List of (parent_tensor, child_tensor) pairs, one per level.
    """
    stps = []
    for net in signal_nets:
        if net.steiner_pts:
            stps.extend([s for s in net.steiner_pts if s is not None])

    if not stps:
        return []

    max_idx = max(int(s.idx) for s in stps)
    N = max_idx + 1

    # Build parent/child adjacency within the same net.
    adj: List[List[int]] = [[] for _ in range(N)]
    parents_of: List[List[int]] = [[] for _ in range(N)]
    out_deg = [0] * N

    for stp in stps:
        u = int(stp.idx)
        u_net = getattr(stp, "net", None)
        if u_net is None:
            continue
        u_db = u_net.db_net
        seen = set()
        for child in getattr(stp, "nexts", []):
            if child is None:
                continue
            c_net = getattr(child, "net", None)
            if c_net is None:
                continue
            # Only follow edges within the same physical net.
            same = (c_net is u_net) or (c_net.db_net.getName() == u_db.getName())
            if not same:
                continue
            v = int(child.idx)
            if v in seen:
                continue
            seen.add(v)
            adj[u].append(v)
            parents_of[v].append(u)

    for u in range(N):
        out_deg[u] = len(adj[u])

    # Backward BFS: start from nodes with out-degree 0 (sinks).
    q = deque([i for i in range(N) if out_deg[i] == 0])
    node_levels: List[List[int]] = []

    while q:
        size = len(q)
        layer = []
        for _ in range(size):
            c = q.popleft()
            layer.append(c)
            for p in parents_of[c]:
                out_deg[p] -= 1
                if out_deg[p] == 0:
                    q.append(p)
        if layer:
            node_levels.append(layer)

    # Handle any disconnected / cycle nodes.
    seen_all = {i for lvl in node_levels for i in lvl}
    leftovers = [i for i in range(N) if i not in seen_all]
    if leftovers:
        node_levels.append(leftovers)

    # Collect all edges flat then split by level (avoids per-level GPU transfers).
    flat_p: List[int] = []
    flat_c: List[int] = []
    level_sizes: List[int] = []

    for lvl in node_levels:
        cnt = 0
        for c in lvl:
            ps = parents_of[c]
            if ps:
                flat_p.extend(ps)
                flat_c.extend([c] * len(ps))
                cnt += len(ps)
        level_sizes.append(cnt)

    if not flat_p:
        return [(torch.tensor([], dtype=torch.long, device=device),
                 torch.tensor([], dtype=torch.long, device=device))
                for _ in node_levels]

    all_p = torch.tensor(flat_p, dtype=torch.long, device=device)
    all_c = torch.tensor(flat_c, dtype=torch.long, device=device)
    p_split = torch.split(all_p, level_sizes)
    c_split = torch.split(all_c, level_sizes)
    return list(zip(p_split, c_split))


# ---------------------------------------------------------------------------
# Main execution plan
# ---------------------------------------------------------------------------

def build_execution_plan(
    steiner_net,
    u_gate_off: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> List:
    """
    Build the forward-propagation execution plan from the levelized network.

    Each level becomes either a ``WireTask`` or a ``GateTask`` depending on
    whether the nodes in that level are routing points or gate output pins.

    Parameters
    ----------
    steiner_net : SteinerNetworkBuilder
    u_gate_off  : [G+1] cumulative gate-offset tensor into U_flat.
    device      : target device for all output tensors.

    Returns
    -------
    execution_plan : list of WireTask / GateTask objects.
    """
    plan = []
    stp_list = steiner_net.stp_list
    off_list = u_gate_off.detach().cpu().tolist()

    for level_idx, layer in enumerate(steiner_net.levelized):
        if not layer:
            plan.append(WireTask(
                level_idx=level_idx,
                p_tensor=torch.zeros(1, dtype=torch.long, device=device),
                c_tensor=torch.zeros(1, dtype=torch.long, device=device),
            ))
            continue

        sample = layer[0]
        is_gate_layer = (sample.Pin is not None and sample.Pin.io_type == "OUTPUT")

        if is_gate_layer:
            plan.append(_build_gate_task(level_idx, layer, stp_list, off_list, device))
        else:
            plan.append(_build_wire_task(level_idx, layer, device))

    return plan


def _build_wire_task(level_idx: int, layer, device) -> WireTask:
    parents, children = [], []
    for stp in layer:
        if not stp.prevs:
            continue
        prev = stp.prevs[0]
        if prev.net is not stp.net:
            continue
        if prev.net is None or prev.net.db_net.getName() != stp.net.db_net.getName():
            continue
        parents.append(prev.idx)
        children.append(stp.idx)
    return WireTask(
        level_idx=level_idx,
        p_tensor=torch.tensor(parents,  dtype=torch.long, device=device),
        c_tensor=torch.tensor(children, dtype=torch.long, device=device),
    )


def _build_gate_task(level_idx: int, layer, stp_list, off_list, device) -> GateTask:
    parents, children, gate_ids = [], [], []
    for stp in layer:
        if not stp.prevs:
            continue
        prev = stp.prevs[0]
        if prev.Pin is None or prev.Pin.gate is None:
            continue
        if prev.Pin.gate is not stp.Pin.gate:
            continue
        parents.append(prev.idx)
        children.append(stp.idx)
        gate_ids.append(stp.Pin.gate.idx)

    p_t = torch.tensor(parents,  dtype=torch.long, device=device)
    c_t = torch.tensor(children, dtype=torch.long, device=device)
    g_t = torch.tensor(gate_ids, dtype=torch.long, device=device)
    E = len(children)

    # Expand over equivalent masters.
    arc_len_list = [len(stp_list[c].Pin.gate.equiv_masters) for c in children]
    arc_len = torch.tensor(arc_len_list, dtype=torch.long, device=device)
    arc_off = torch.zeros(E + 1, dtype=torch.long, device=device)
    arc_off[1:] = torch.cumsum(arc_len, 0)
    T = int(arc_off[-1].item())

    entry_arc   = torch.repeat_interleave(torch.arange(E, device=device), arc_len)
    entry_local = torch.arange(T, device=device) - arc_off[entry_arc]

    p_e   = p_t[entry_arc]
    c_e   = c_t[entry_arc]
    gid_e = g_t[entry_arc]

    # Map each entry to its U_flat index.
    u_index_e = torch.tensor(
        [int(off_list[gid_e[i].item()]) + entry_local[i].item()
         for i in range(T)], dtype=torch.long, device=device)

    # Group entries by (gate_idx, u_index) to allow weight-blended output.
    gid_list = gid_e.tolist()
    uidx_list = u_index_e.tolist()
    group_map: Dict[tuple, List[int]] = {}
    for i in range(T):
        key = (gid_list[i], uidx_list[i])
        group_map.setdefault(key, []).append(i)

    group_list   = [0] * T
    u_idx_group  = []
    for cnt, (key, indices) in enumerate(group_map.items()):
        u_idx_group.append(key[1])
        for i in indices:
            group_list[i] = cnt

    group_t    = torch.tensor(group_list,  dtype=torch.long, device=device)
    u_grp_t    = torch.tensor(u_idx_group, dtype=torch.long, device=device)
    num_groups = len(u_idx_group)

    c_group = torch.full((num_groups,), -1, dtype=torch.long, device=device)
    c_group.scatter_reduce_(0, group_t, c_e, reduce="amax", include_self=True)

    return GateTask(
        level_idx=level_idx,
        p_e=p_e, c_e=c_e,
        u_index_group=u_grp_t,
        group_tensor=group_t,
        c_group=c_group,
    )


# ---------------------------------------------------------------------------
# Buffer regression helpers (used by graph.py)
# ---------------------------------------------------------------------------

def _buf_regression(delay_table, axis_0, axis_1) -> Optional[Dict]:
    """
    Fit a 5-coefficient polynomial model to the buffer delay/slew table.

    Model form:
        f(s, c) = a0 + a1·(c/c_sc) + a2·(s/s_sc) + a3·cbrt(s/s_sc) + a4·(s/s_sc)²

    Returns a dict with keys {"coeffs": [...], "s_scale": float, "c_scale": float}
    or None if fitting fails.
    """
    try:
        from placeopt.timing._regression import fit_buf_model
        return fit_buf_model(delay_table, axis_0, axis_1)
    except Exception:
        return None


def _make_buf_funcs(coeff: torch.Tensor, s_scale, c_scale):
    """
    Return (f_slew, f_load) closure pair for the buffer regression model.

    ``f_slew(slew_tensor)`` → delay/slew contribution from input slew.
    ``f_load(load_tensor)`` → delay/slew contribution from load capacitance.
    """
    def _f_slew(slew):
        c = coeff.to(device=slew.device, dtype=slew.dtype)
        s_n = slew / s_scale
        s_cbrt = torch.sign(s_n) * (s_n.abs() + 1e-10).pow(1.0 / 3.0)
        return c[2] * s_n + c[3] * s_cbrt + c[4] * s_n ** 2

    def _f_load(load):
        c = coeff.to(device=load.device, dtype=load.dtype)
        c_n = load / c_scale
        return c[0] + c[1] * c_n

    return _f_slew, _f_load
