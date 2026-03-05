"""
eda.buffering — Physical buffer-insertion utilities.

This module provides:

* ``is_buffer`` / ``is_inverter`` — Liberty-name heuristics for cell type detection.
* ``insert_buffer`` — low-level OpenDB function to insert one buffer instance
  on a net, splitting it into lhs/rhs sub-nets.
* ``commit_buffering`` — high-level function that reads the optimized
  ``should_buffer`` mask and inserts all required buffers into the OpenDB block.
* ``remove_buffer`` — remove a previously inserted buffer and reconnect its net.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import odb  # type: ignore


# ---------------------------------------------------------------------------
# Cell-type detection
# ---------------------------------------------------------------------------

def is_buffer(master) -> bool:
    """Return True if *master* is a buffer cell (name heuristic)."""
    if master is None:
        return False
    name = master.getName().upper()
    return "BUF" in name or any(x in name for x in ("HB4XP67", "HB3XP67", "HB2XP67", "HB1XP67"))


def is_inverter(master) -> bool:
    """Return True if *master* is an inverter cell (name heuristic)."""
    if master is None:
        return False
    return "INV" in master.getName().upper()


# ---------------------------------------------------------------------------
# Helper data class
# ---------------------------------------------------------------------------

@dataclass
class BufferInsertionResult:
    """Holds the OpenDB objects created by a single buffer insertion."""
    lhs_net:    odb.dbNet
    rhs_net:    odb.dbNet
    buf_in:     odb.dbITerm
    buf_out:    odb.dbITerm


# ---------------------------------------------------------------------------
# Name generators
# ---------------------------------------------------------------------------

_cell_counter = 0
_net_counter  = 0


def _unique_cell_name(block, base: str) -> str:
    global _cell_counter
    while True:
        name = f"{base}_{_cell_counter}"
        _cell_counter += 1
        if block.findInst(name) is None:
            return name


def _unique_net_name(block, base: str) -> str:
    global _net_counter
    while True:
        name = f"{base}_{_net_counter}"
        _net_counter += 1
        if block.findNet(name) is None:
            return name


def _iterm_key(iterm: odb.dbITerm) -> str:
    return f"{iterm.getInst().getName()}/{iterm.getMTerm().getName()}"


def _pick_mterm(master, io_type: str, preferred: List[str]):
    """Find the first matching MTerm by I/O type, preferring names in *preferred*."""
    candidates = [m for m in master.getMTerms() if m.getIoType() == io_type]
    if not candidates:
        return None
    for pref in preferred:
        for m in candidates:
            if m.getName().upper() == pref:
                return m
    return candidates[0]


# ---------------------------------------------------------------------------
# Single buffer insertion
# ---------------------------------------------------------------------------

def insert_buffer(
    lhs_pins: List[odb.dbITerm],
    rhs_pins: List[odb.dbITerm],
    buf_master,
    original_net: odb.dbNet,
    x: float, y: float,
    inst_name: Optional[str] = None,
    new_net_name: Optional[str] = None,
    pin_in: str = "A",
    pin_out: str = "Y",
) -> Optional[BufferInsertionResult]:
    """
    Insert *buf_master* between ``lhs_pins`` and ``rhs_pins`` on ``original_net``.

    The function:
    1. Creates a new net (rhs_net) for the downstream side.
    2. Instantiates the buffer at (x, y).
    3. Moves all rhs pins to rhs_net; keeps lhs pins on original_net.
    4. Connects buffer input → original_net, buffer output → rhs_net.

    Returns ``None`` if the insertion would be invalid (no driver, overlapping
    lhs/rhs, or OpenDB creation failure).

    Parameters
    ----------
    lhs_pins    : pins that remain on the original (upstream) net.
    rhs_pins    : pins to be moved to the new (downstream) net.
    buf_master  : dbMaster for the buffer cell.
    original_net: the net to split.
    x, y        : placement coordinates in DBU.
    """
    if buf_master is None or original_net is None:
        return None

    block = original_net.getBlock()
    if block is None:
        return None

    # Verify exactly one driver on lhs.
    lhs_keys = {_iterm_key(it) for it in lhs_pins if it}
    for it in rhs_pins:
        if it and _iterm_key(it) in lhs_keys:
            return None  # overlap — skip

    driver = None
    for it in lhs_pins:
        if it and it.isOutputSignal():
            if driver is not None:
                return None  # multiple drivers — skip
            driver = it
    if driver is None:
        return None  # no driver — skip

    # Create unique names.
    inst_name     = inst_name     or _unique_cell_name(block, "buf")
    new_net_name  = new_net_name  or _unique_net_name(block, "buf_net")

    if block.findInst(inst_name) is not None:
        inst_name = _unique_cell_name(block, inst_name)
    if block.findNet(new_net_name) is not None:
        new_net_name = _unique_net_name(block, new_net_name)

    rhs_net = odb.dbNet_create(block, new_net_name)
    rhs_net.setSigType(original_net.getSigType())

    inst = odb.dbInst_create(block, buf_master, inst_name)
    if inst is None:
        odb.dbNet_destroy(rhs_net)
        return None

    inst.setLocation(int(x), int(y))
    inst.setOrient("R0")
    inst.setPlacementStatus("PLACED")

    # Find buffer I/O terminals.
    in_mterm  = _pick_mterm(buf_master, "INPUT",  ["A"])
    out_mterm = _pick_mterm(buf_master, "OUTPUT", ["Y"])
    if in_mterm is None or out_mterm is None:
        odb.dbInst_destroy(inst)
        odb.dbNet_destroy(rhs_net)
        return None

    buf_in  = inst.findITerm(in_mterm.getName())
    buf_out = inst.findITerm(out_mterm.getName())

    orig_name = original_net.getName()

    # Move rhs pins to new net.
    for it in rhs_pins:
        if it is None:
            continue
        it_net = it.getNet()
        if it_net is not None and it_net.getName() == orig_name:
            it.disconnect()
            it.connect(rhs_net)

    # Connect buffer.
    buf_in.connect(original_net)
    buf_out.connect(rhs_net)

    # Ensure lhs pins are still on original net.
    for it in lhs_pins:
        if it is None:
            continue
        it_net = it.getNet()
        if it_net is None or it_net.getName() != orig_name:
            it.disconnect()
            it.connect(original_net)

    return BufferInsertionResult(
        lhs_net=original_net, rhs_net=rhs_net,
        buf_in=buf_in, buf_out=buf_out,
    )


# ---------------------------------------------------------------------------
# Topological sort on Steiner points
# ---------------------------------------------------------------------------

def _topo_sort(stps) -> List[int]:
    """Kahn's topological sort on a list of Steiner points (by list index)."""
    n = len(stps)
    pos = {stp: i for i, stp in enumerate(stps)}
    indeg = [0] * n
    for u in range(n):
        for child in stps[u].nexts:
            v = pos.get(child)
            if v is not None:
                indeg[v] += 1

    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for child in stps[u].nexts:
            v = pos.get(child)
            if v is None:
                continue
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != n:
        seen = set(order)
        order.extend(i for i in range(n) if i not in seen)
    return order


def _subtree_pin_set(
    net_obj,
    stps: list,
    rep: Dict[int, Set[str]],
    idx: int,
    driver_key: str,
) -> Set[str]:
    """Compute the set of downstream pin keys for the subtree rooted at stps[idx]."""
    stp = stps[idx]
    rhs: Set[str] = set()
    if stp.Pin is not None:
        key = _iterm_key(stp.Pin.db_iterm)
        if key != driver_key:
            rhs.add(key)
    for child in stp.nexts:
        rhs |= rep.get(child.idx, set())
    return rhs


# ---------------------------------------------------------------------------
# High-level commit function
# ---------------------------------------------------------------------------

def commit_buffering(net_obj, should_buffer: dict, buf_master) -> list:
    """
    Insert all required buffers into *net_obj* according to *should_buffer*.

    Parameters
    ----------
    net_obj       : ``circuit.Net`` wrapper with ``.steiner_pts``.
    should_buffer : dict  stp_global_idx → bool.
    buf_master    : dbMaster for the buffer cell.

    Returns
    -------
    change_list : list of (dbInst, dbMaster, dbNet) triples, one per buffer.
    """
    stps = net_obj.steiner_pts
    # Skip if nothing to buffer.
    if not any(should_buffer.get(stp.idx, False) for stp in stps):
        return []

    # Cache midpoint offsets of buffer I/O pins.
    def _pin_centre(mterm):
        for mpin in mterm.getMPins():
            b = mpin.getBBox()
            return (b.xMin() + b.xMax()) / 2.0, (b.yMin() + b.yMax()) / 2.0
        return 0.0, 0.0

    out_cx, out_cy = _pin_centre(buf_master.findMTerm("Y"))

    # Process Steiner points in reverse topological order (sinks first).
    topo = _topo_sort(stps)
    order = list(reversed(topo))

    current_net = net_obj.db_net
    driver_key  = _iterm_key(net_obj.driver_pin.db_iterm)
    rep: Dict[int, Set[str]] = {}
    change_list = []

    for local_idx in order[:-1]:   # skip driver
        stp = stps[local_idx]
        rhs_set = _subtree_pin_set(net_obj, stps, rep, local_idx, driver_key)

        if not should_buffer.get(stp.idx, False):
            rep[stp.idx] = rhs_set
            continue

        # Resolve live iterms on the current net.
        alive = {_iterm_key(it): it for it in current_net.getITerms()}
        rhs_live = rhs_set & alive.keys()
        lhs_live = set(alive.keys()) - rhs_live

        if not rhs_live or not lhs_live:
            rep[stp.idx] = rhs_set
            continue

        if driver_key in rhs_live:
            rhs_live.discard(driver_key)
            lhs_live.add(driver_key)
            if not rhs_live:
                rep[stp.idx] = rhs_set
                continue

        drv_iterm = alive.get(driver_key, net_obj.driver_pin.db_iterm)
        lhs_list  = [drv_iterm] + [alive[k] for k in sorted(lhs_live) if k != driver_key]
        rhs_list  = [alive[k] for k in sorted(rhs_live)]

        mid_x = (stp.x + stp.prevs[0].x) / 2.0
        mid_y = (stp.y + stp.prevs[0].y) / 2.0
        buf_x = int(mid_x - out_cx)
        buf_y = int(mid_y - out_cy)

        result = insert_buffer(lhs_list, rhs_list, buf_master, current_net,
                               buf_x, buf_y, pin_in="A", pin_out="Y")
        if result is None:
            rep[stp.idx] = rhs_set
            continue

        from placeopt.circuit.components import Gate
        new_gate = Gate(result.buf_in.getInst(), buf_master)
        change_list.append((result.buf_in.getInst(), buf_master, result.rhs_net))

        current_net = result.lhs_net
        rep[stp.idx] = {_iterm_key(result.buf_in)}

    return change_list


# ---------------------------------------------------------------------------
# Buffer removal
# ---------------------------------------------------------------------------

def remove_buffer(inst) -> None:
    """
    Remove a buffer instance and reconnect its downstream net to the upstream net.

    All sinks that were on the buffer's output net are moved back to its
    input net; the output net is then destroyed.
    """
    if inst is None:
        return
    buf_in  = inst.findITerm("A")
    buf_out = inst.findITerm("Y")
    if buf_in is None or buf_out is None:
        return

    out_net  = buf_out.getNet()
    in_net   = buf_in.getNet()
    if out_net is None or in_net is None or out_net is in_net:
        odb.dbInst_destroy(inst)
        return

    # Reconnect all sinks from out_net → in_net.
    for it in list(out_net.getITerms()):
        if it is buf_out or it.getInst() is inst:
            continue
        it.disconnect()
        it.connect(in_net)

    odb.dbInst_destroy(inst)
    odb.dbNet_destroy(out_net)
