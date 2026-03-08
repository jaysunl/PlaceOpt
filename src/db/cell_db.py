import time
from collections import deque

import odb
import torch
import torch.nn as nn
import numpy as np
import openroad as ord
from openroad import Tech, Design, Timing

from .netlist import LogicCell, SignalPin, WireNet
from src.util.helpers import isDriverPin, is_circuit_input, is_circuit_output
from src.sta.arc_model import lut_bilinear_batch


class TimingLUT(nn.Module):
    """Liberty timing arc stored as differentiable delay/slew lookup tables."""

    def __init__(self, c_model):
        super().__init__()
        self.arc_description = c_model.arc_description
        self.in_pin_name  = c_model.in_pin_name
        self.out_pin_name = c_model.out_pin_name
        self.in_rf  = c_model.in_rf
        self.out_rf = c_model.out_rf

        self.register_buffer('axis_0', torch.tensor(list(c_model.table_axis0), dtype=torch.float32))
        self.register_buffer('axis_1', torch.tensor(list(c_model.table_axis1), dtype=torch.float32))
        self.register_buffer('delay_table', torch.tensor(
            [list(r) for r in c_model.delay_table], dtype=torch.float32))
        self.register_buffer('slew_table', torch.tensor(
            [list(r) for r in c_model.slew_table], dtype=torch.float32))

        self.driver_rd = 0.0
        self._estimate_rd()

    def _interp(self, table, axis0_val, axis1_val):
        return lut_bilinear_batch(
            table[None, :, :],
            self.axis_0[None, :],
            self.axis_1[None, :],
            axis0_val[None],
            axis1_val[None],
        )

    def query_delay(self, axis0_value, axis1_value):
        return self._interp(self.delay_table, axis0_value, axis1_value)

    def query_slew(self, axis0_value, axis1_value):
        return self._interp(self.slew_table, axis0_value, axis1_value)

    def _estimate_rd(self):
        def _mid(t):
            return (t[len(t) // 2 + 1] + t[len(t) // 2]) / 2 if len(t) > 2 else t[-1] * 0.75 + t[0] * 0.25

        ms = _mid(self.axis_0)
        mc = _mid(self.axis_1)
        delta_c = 1e-15
        d1 = self.query_delay(ms, mc)
        d2 = self.query_delay(ms, mc + delta_c)
        rd = -torch.log(torch.tensor(0.5)) * torch.abs(d2 - d1) / delta_c
        self.driver_rd = rd if rd >= 0.001 else torch.tensor(0.001, dtype=torch.float32)


class NetlistDB:
    """Manages the full netlist of gates, nets, and pins with timing arc tables."""

    def __init__(self, design):
        self.design = design
        self.timing = Timing(design)

        self.gates   = []
        self.nets    = []
        self.pin_map = {}
        self.masters = []

        self.start_points = []
        self.end_points   = []
        self.signal_nets  = []
        self.signal_gates = []

        self.init_cells            = []
        self.init_pos              = []
        self.init_cell_name_to_pos = {}
        self.init_cell_idxs        = []

        self.arc_luts   = {}
        self.setup_luts = {}

        self._build()

    def _build(self):
        block  = self.design.getBlock()
        libs   = ord.get_db().getLibs()
        timing = self.timing

        for lib in libs:
            self.masters.extend(lib.getMasters())
        timing.makeEquivCells()

        self.gates   = [LogicCell(inst, inst.getMaster()) for inst in block.getInsts()]
        self.pin_map = {it.getName(): SignalPin(it) for it in block.getITerms()}
        self.nets    = [WireNet(net) for net in block.getNets()]

        for master in self.masters:
            table = {
                (m.in_pin_name, m.out_pin_name, m.out_rf): TimingLUT(m)
                for m in timing.getLibertyCellTableModels(master)
            }
            self.arc_luts[master.getName()] = table

        for g in self.gates:
            pins = [self.pin_map[it.getName()] for it in g.db_Inst.getITerms()]
            for p in pins:
                self.pin_map[p.db_ITerm.getName()].Gate = g
            g.setPins(pins)
            g.setEqvMaster(timing)

        for n in self.nets:
            pins = [self.pin_map[it.getName()] for it in n.db_net.getITerms()]
            for p in pins:
                self.pin_map[p.db_ITerm.getName()].Net = n
            n.setPins(pins)

        self._extract_signal_components()
        self._snapshot_cell_positions(refresh=True)

    def reinitialize(self):
        """Rebuild gate/net/pin data after cell swaps, reusing existing arc tables."""
        start = time.time()
        print("[DB] rebuilding netlist data...")

        self.gates   = []
        self.nets    = []
        self.pin_map = {}
        self.masters = []
        self.start_points = []
        self.end_points   = []
        self.signal_nets  = []
        self.signal_gates = []

        block  = self.design.getBlock()
        libs   = ord.get_db().getLibs()
        timing = self.timing

        for lib in libs:
            self.masters.extend(lib.getMasters())

        self.gates   = [LogicCell(inst, inst.getMaster()) for inst in block.getInsts()]
        self.pin_map = {it.getName(): SignalPin(it) for it in block.getITerms()}
        self.nets    = [WireNet(net) for net in block.getNets()]

        for g in self.gates:
            pins = [self.pin_map[it.getName()] for it in g.db_Inst.getITerms()]
            for p in pins:
                self.pin_map[p.db_ITerm.getName()].Gate = g
            g.setPins(pins)
            g.setEqvMaster(timing)

        for n in self.nets:
            pins = [self.pin_map[it.getName()] for it in n.db_net.getITerms()]
            for p in pins:
                self.pin_map[p.db_ITerm.getName()].Net = n
            n.setPins(pins)

        self._extract_signal_components()
        self._snapshot_cell_positions(refresh=False)

        print(f"[DB] rebuild complete in {time.time() - start:.2f}s")

    def get_luts(self, master):
        return self.arc_luts[master.getName()]

    def _extract_signal_components(self):
        cnt = 0
        for g in self.gates:
            g.idx = -1
            if not g.inputPins:
                continue  # TIE cell — skip
            table_models = self.get_luts(g.db_master)
            first = next(iter(table_models.values()), None)
            if first is not None:
                g.set_unateness(0 if first.in_rf == first.out_rf else 1)
            self.signal_gates.append(g)
            g.idx = cnt
            cnt += 1

        cnt = 0
        for n in self.nets:
            for p in n.Pins:
                if isDriverPin(p):
                    n.driver_pin = p
                    break
            if n.db_net.getSigType() != "SIGNAL":
                continue
            if any(p.db_ITerm.getMTerm().getName().upper().find("CLK") != -1 for p in n.Pins):
                continue  # skip clock nets
            self.signal_nets.append(n)
            n.idx = cnt
            cnt += 1

        for pin in self.pin_map.values():
            if pin.Net is None or pin.Gate is None:
                continue
            if is_circuit_input(pin.db_ITerm):
                self.start_points.append(pin)
            if is_circuit_output(pin.db_ITerm):
                self.end_points.append(pin)
            if not self.design.isSequential(pin.Gate.db_Inst.getMaster()):
                continue
            n = pin.Net
            if n.db_net.getSigType() in ("POWER", "GROUND"):
                continue
            if n.db_net.getSigType() == "CLOCK":
                self.start_points.append(pin)
            if pin.IO == "INPUT":
                self.end_points.append(pin)

    def _inst_xy(self, inst):
        if inst is None:
            return None
        for attr in ("getLocation", "getOrigin"):
            if hasattr(inst, attr):
                loc = getattr(inst, attr)()
                return float(loc[0]), float(loc[1])
        if hasattr(inst, "getBBox"):
            bbox = inst.getBBox()
            return (float(bbox.xMin()), float(bbox.yMin())) if bbox else (0.0, 0.0)
        return 0.0, 0.0

    def _snapshot_cell_positions(self, refresh=False):
        if refresh or not self.init_cell_name_to_pos:
            self.init_cell_name_to_pos = {
                g.db_Inst.getName(): self._inst_xy(g.db_Inst)
                for g in self.signal_gates
                if g.db_Inst is not None
            }
        self.init_cells     = []
        self.init_pos       = []
        self.init_cell_idxs = []
        for idx, g in enumerate(self.signal_gates):
            if g.db_Inst is None:
                continue
            name = g.db_Inst.getName()
            if name in self.init_cell_name_to_pos:
                self.init_cells.append(g)
                self.init_pos.append(self.init_cell_name_to_pos[name])
                self.init_cell_idxs.append(idx)
