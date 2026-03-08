import odb
import torch
import torch.nn as nn

from src.util.helpers import isSignalNet


class SteinerNode:
    """A node in the Steiner tree approximating a routed net."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.nexts = []   # downstream SteinerNode list
        self.prevs = []   # upstream SteinerNode list
        self.level = -1

        self.Pin = None   # SignalPin if this node coincides with a pin
        self.Net = None   # WireNet this node belongs to
        self.idx = -1     # index within the net's node list


def map_pins_to_tree(pin_vec, node_vec):
    """Assign circuit pins to the closest Steiner tree nodes by coordinates."""
    for p in pin_vec:
        p.steinerPoint = None

    if not pin_vec or not node_vec:
        return node_vec

    pin_map = {}
    for pin in pin_vec:
        x = pin.db_ITerm.getAvgXY()[1]
        y = pin.db_ITerm.getAvgXY()[2]
        pin_map.setdefault((x, y), []).append(pin)

    for i, node in enumerate(node_vec):
        node.idx = i
        coords = (node.x, node.y)
        if coords in pin_map:
            pins = pin_map[coords]
            if pins:
                pin = pins.pop(0)
                node.Pin = pin
                pin.steinerPoint = node
                if not pins:
                    del pin_map[coords]

    return node_vec


class WireNet:
    """Represents a physical net connecting design pins."""

    def __init__(self, db_net):
        self.Pins = []
        self.steinerPoints = []

        self.db_net = db_net
        self.idx = -1
        self.driver_pin = None

    def setPins(self, pins):
        self.Pins = pins
        for p in pins:
            if p.IO == "OUTPUT":
                self.driver_pin = p
                break


class SignalPin:
    """Wrapper around an OpenROAD dbITerm (instance terminal)."""

    def __init__(self, db_iterm):
        self.db_ITerm = db_iterm
        self.name = db_iterm.getMTerm().getName()
        self.IO = db_iterm.getMTerm().getIoType()  # INPUT, OUTPUT, INOUT

        self.Gate = None
        self.Net = None
        self.steinerPoint = None


class LogicCell:
    """Wrapper around an OpenROAD dbInst (gate instance)."""

    def __init__(self, db_inst, db_master):
        self.db_Inst = db_inst
        self.original_dbMaster = db_master  # master at load time
        self.db_master = db_master          # current master

        self.Pins = {}
        self.inputPins = []
        self.inpin_name_to_idx = {}
        self.outputPins = []
        self.outpin_name_to_idx = {}
        self.eqvMaster = []

        self.idx = -1       # position in signal_gates list
        self.unateness = 0  # 0: non-inverting, 1: inverting

    def setPins(self, pins):
        for p in pins:
            self.Pins[p.name] = p
        self.inputPins = [p for p in self.Pins.values() if p.IO == "INPUT"]
        for i, p in enumerate(self.inputPins):
            self.inpin_name_to_idx[p.name] = i
        self.outputPins = [p for p in self.Pins.values() if p.IO == "OUTPUT"]
        for i, p in enumerate(self.outputPins):
            self.outpin_name_to_idx[p.name] = i

    def set_unateness(self, unateness):
        self.unateness = unateness

    def setEqvMaster(self, timing):
        masters = timing.equivCells(self.db_master)
        self.eqvMaster = [self.db_master]
        for m in masters:
            if m.isBlock():
                break
            if m.getName() == self.db_master.getName():
                continue
            self.eqvMaster.append(m)

    def swapCell(self, new_master):
        if new_master not in self.eqvMaster:
            print(f"[ERROR] cannot swap {self.db_master.getName()} to {new_master.getName()}")
            return False
        self.db_master = new_master
        self.db_Inst.swapMaster(new_master)

    def __str__(self):
        lines = [f"Gate: {self.db_Inst.getName()}", f"Master: {self.db_master.getName()}"]
        for pin in self.Pins.values():
            lines.append(f"  Pin: {pin.name}")
        return "\n".join(lines)
