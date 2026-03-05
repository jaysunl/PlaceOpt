"""
circuit.library — CellLibrary: builds the circuit model from OpenROAD DB.

This module mirrors ``src.CircuitLibrary`` but uses the cleaner component
classes from ``circuit.components`` and avoids holding mutable shared state
between the Liberty-table models (which are immutable after construction).

Responsibilities
----------------
* Instantiate Gate / Pin / Net wrappers from the OpenDB block.
* Load Liberty NLDM timing tables for every master cell into ``TensorTable``
  objects ready for GPU transfer.
* Identify signal gates, signal nets, circuit I/O (start/end timing points),
  and per-gate equivalent-master sets.
* Track initial cell positions for the displacement penalty.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import openroad as ord  # type: ignore
from openroad import Timing  # type: ignore

from placeopt.circuit.components import Gate, Net, Pin, SteinerPoint
from placeopt.io.design import get_rss_mb
from placeopt.timing.lut import TensorTable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_driver_pin(pin_or_iterm) -> bool:
    """Return True when *pin_or_iterm* drives a signal net."""
    iterm = getattr(pin_or_iterm, "db_iterm", pin_or_iterm)
    net = iterm.getNet()
    if net is None:
        return False
    if net.getSigType() in ("CLOCK", "POWER", "GROUND"):
        return False
    return bool(iterm.isOutputSignal())


def _is_circuit_input(iterm) -> bool:
    net = iterm.getNet()
    if net is None:
        return False
    sig = net.getSigType()
    if sig in ("CLOCK", "POWER", "GROUND"):
        return False
    if iterm.getMTerm().getName().upper().find("CLK") != -1:
        return False
    for bterm in net.getBTerms():
        if bterm.getIoType() in ("INPUT", "INOUT") and bterm.getSigType() not in ("POWER", "GROUND"):
            return True
    return False


def _is_circuit_output(iterm) -> bool:
    net = iterm.getNet()
    if net is None:
        return False
    if net.getSigType() in ("CLOCK", "POWER", "GROUND"):
        return False
    if _is_driver_pin(iterm):
        for peer in iterm.getNet().getITerms():
            if peer != iterm and not _is_driver_pin(peer):
                return False
    for bterm in net.getBTerms():
        if bterm.getIoType() in ("OUTPUT", "INOUT") and bterm.getSigType() not in ("POWER", "GROUND"):
            return True
    return False


def _inst_origin(inst) -> Tuple[float, float]:
    """Return (x, y) origin of an OpenDB instance in DBU."""
    if hasattr(inst, "getLocation"):
        loc = inst.getLocation()
        return float(loc[0]), float(loc[1])
    if hasattr(inst, "getOrigin"):
        loc = inst.getOrigin()
        return float(loc[0]), float(loc[1])
    bbox = inst.getBBox()
    return float(bbox.xMin()), float(bbox.yMin())


# ---------------------------------------------------------------------------
# CellLibrary
# ---------------------------------------------------------------------------

class CellLibrary:
    """
    Builds and caches all circuit-model data needed for timing optimization.

    Parameters
    ----------
    design : openroad.Design
        Loaded OpenROAD design object.

    Notes
    -----
    After construction the following public attributes are available:

    ``signal_gates``   — list of Gate objects participating in optimization
    ``signal_nets``    — list of Net objects with SIGNAL type (non-clock)
    ``start_points``   — timing start points (PI bterms, FF clk pins, FF Q pins)
    ``end_points``     — timing end points (PO bterms, FF D/SI/SE pins)
    ``TensorTableModels``
                       — dict: master_name → {(in_pin, out_pin, rf) → TensorTable}
    """

    def __init__(self, design) -> None:
        self.design = design
        self.timing = Timing(design)

        # Raw lists populated during construction.
        self._all_gates: List[Gate] = []
        self._all_nets: List[Net] = []
        self._pin_dict: Dict[str, Pin] = {}
        self._master_vec: List = []

        # Optimization-relevant subsets.
        self.signal_gates: List[Gate] = []
        self.signal_nets: List[Net] = []
        self.start_points: List[Pin] = []
        self.end_points: List[Pin] = []

        # Liberty timing tables (immutable after construction).
        self.TensorTableModels: Dict[str, Dict] = {}

        # Initial cell positions for displacement penalty.
        self._init_pos: Dict[str, Tuple[float, float]] = {}  # inst_name → (x, y)
        self.init_cell_idxs: List[int] = []
        self.init_pos: List[Tuple[float, float]] = []

        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        block = self.design.getBlock()
        db = ord.get_db()
        timing = self.timing

        # 1. Collect all masters.
        for lib in db.getLibs():
            self._master_vec.extend(lib.getMasters())

        # 2. Build equivalent-cell map (required before Gate.set_equiv_masters).
        timing.makeEquivCells()

        # 3. Load Liberty tables for every master.
        for master in self._master_vec:
            table: Dict = {}
            for c_model in timing.getLibertyCellTableModels(master):
                key = (c_model.in_pin_name, c_model.out_pin_name, c_model.out_rf)
                table[key] = TensorTable(c_model)
            self.TensorTableModels[master.getName()] = table

        # 4. Instantiate Gate wrappers.
        self._all_gates = [Gate(inst, inst.getMaster()) for inst in block.getInsts()]

        # 5. Instantiate Pin wrappers.
        self._pin_dict = {it.getName(): Pin(it) for it in block.getITerms()}

        # 6. Instantiate Net wrappers.
        self._all_nets = [Net(n) for n in block.getNets()]

        # 7. Wire pins ↔ gates.
        for g in self._all_gates:
            gate_pins = [self._pin_dict[it.getName()] for it in g.db_inst.getITerms()]
            g.set_pins(gate_pins)
            for p in gate_pins:
                p.gate = g
            g.set_equiv_masters(timing)
            # Determine unateness from the first available arc.
            tbl = self.TensorTableModels.get(g.db_master.getName(), {})
            first = next(iter(tbl.values()), None)
            if first is not None:
                g.set_unateness(0 if first.in_rf == first.out_rf else 1)

        # 8. Wire pins ↔ nets.
        for n in self._all_nets:
            net_pins = [self._pin_dict[it.getName()] for it in n.db_net.getITerms()]
            n.set_pins(net_pins)
            for p in net_pins:
                p.net = n

        # 9. Extract optimization-relevant subsets.
        self._extract_signal_components()

        # 10. Record initial cell positions.
        self._record_init_positions()

    def _extract_signal_components(self) -> None:
        """Populate signal_gates, signal_nets, start_points, end_points."""
        sig_idx = 0
        for g in self._all_gates:
            if not g.input_pins:
                continue  # TIE cells / power cells — skip
            tbl = self.TensorTableModels.get(g.db_master.getName(), {})
            if not tbl:
                continue  # no timing arcs → not a sizing candidate
            g.idx = sig_idx
            self.signal_gates.append(g)
            sig_idx += 1

        net_idx = 0
        for n in self._all_nets:
            if n.db_net.getSigType() != "SIGNAL":
                continue
            # Filter out nets that are physically on a clock domain.
            if any(p.db_iterm.getMTerm().getName().upper().find("CLK") != -1 for p in n.pins):
                continue
            n.idx = net_idx
            self.signal_nets.append(n)
            net_idx += 1

        for pin in self._pin_dict.values():
            if pin.net is None:
                continue
            if _is_circuit_input(pin.db_iterm):
                self.start_points.append(pin)
            if _is_circuit_output(pin.db_iterm):
                self.end_points.append(pin)
            if not self.design.isSequential(pin.gate.db_inst.getMaster()):
                continue
            n = pin.net
            if n.db_net.getSigType() in ("POWER", "GROUND"):
                continue
            if n.db_net.getSigType() == "CLOCK":
                self.start_points.append(pin)
            elif pin.io_type == "INPUT":
                self.end_points.append(pin)

    def _record_init_positions(self) -> None:
        """Record initial (x, y) of every signal gate for displacement tracking."""
        self._init_pos = {}
        for g in self.signal_gates:
            self._init_pos[g.db_inst.getName()] = _inst_origin(g.db_inst)

        self.init_cell_idxs = []
        self.init_pos = []
        for g in self.signal_gates:
            name = g.db_inst.getName()
            if name in self._init_pos:
                self.init_cell_idxs.append(g.idx)
                self.init_pos.append(self._init_pos[name])

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_table(self, master) -> Dict:
        """Return the timing-table dict for *master* (keyed by arc tuple)."""
        return self.TensorTableModels.get(master.getName(), {})

    # Legacy alias used by sttMgr / graph builder.
    def getTensorTableModels(self, master):  # noqa: N802
        return self.get_table(master)

    @property
    def MasterVec(self):  # noqa: N802
        return self._master_vec

    # ------------------------------------------------------------------
    # Re-initialize after the design has been modified (e.g. after buffering)
    # ------------------------------------------------------------------

    def reinitialize(self) -> None:
        """
        Rebuild the circuit model from the current OpenDB state.

        Called after buffer insertion to refresh gate/net/pin lists while
        keeping the Liberty table models (which are static) and the initial
        position snapshot unchanged.
        """
        t0 = time.perf_counter()
        rss = get_rss_mb()
        if rss:
            print(f"[MEM] CellLibrary.reinitialize start: {rss:.1f} MB")

        block = self.design.getBlock()

        # Rebuild mutable lists only.
        self._all_gates = [Gate(inst, inst.getMaster()) for inst in block.getInsts()]
        self._pin_dict = {it.getName(): Pin(it) for it in block.getITerms()}
        self._all_nets = [Net(n) for n in block.getNets()]

        for g in self._all_gates:
            gate_pins = [self._pin_dict[it.getName()] for it in g.db_inst.getITerms()]
            g.set_pins(gate_pins)
            for p in gate_pins:
                p.gate = g
            g.set_equiv_masters(self.timing)
            tbl = self.TensorTableModels.get(g.db_master.getName(), {})
            first = next(iter(tbl.values()), None)
            if first is not None:
                g.set_unateness(0 if first.in_rf == first.out_rf else 1)

        for n in self._all_nets:
            net_pins = [self._pin_dict[it.getName()] for it in n.db_net.getITerms()]
            n.set_pins(net_pins)
            for p in net_pins:
                p.net = n

        self.signal_gates = []
        self.signal_nets = []
        self.start_points = []
        self.end_points = []
        self._extract_signal_components()

        # Rebuild idx mappings but keep old init_pos snapshot.
        self.init_cell_idxs = []
        self.init_pos = []
        for g in self.signal_gates:
            name = g.db_inst.getName()
            if name in self._init_pos:
                self.init_cell_idxs.append(g.idx)
                self.init_pos.append(self._init_pos[name])

        rss2 = get_rss_mb()
        if rss2:
            print(f"[MEM] CellLibrary.reinitialize end: {rss2:.1f} MB")
        print(f"[INFO] CellLibrary.reinitialize done in {time.perf_counter()-t0:.2f}s")
