"""
circuit.components — lightweight wrappers around OpenROAD database objects.

Each class holds a reference to the underlying OpenDB object plus a few
derived attributes that are used repeatedly during optimization (pin I/O
direction, equivalent-master list, Steiner-point pointer, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ---------------------------------------------------------------------------
# Steiner tree node
# ---------------------------------------------------------------------------

class SteinerPoint:
    """
    A node in a rectilinear Steiner minimum tree (RSMT).

    Attributes
    ----------
    x, y    : coordinates in OpenDB units (DBU)
    idx     : global flat index assigned by SteinerNetworkBuilder
    Pin     : attached circuit Pin (None for routing-only Steiner nodes)
    Net     : the Net object this point belongs to
    prevs   : upstream (toward driver) neighbour(s)
    nexts   : downstream (toward sinks) neighbour(s)
    level   : topological level in the circuit graph (-1 until assigned)
    """

    __slots__ = ("x", "y", "idx", "Pin", "Net", "prevs", "nexts", "level")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.idx: int = -1
        self.Pin = None
        self.Net = None
        self.prevs: List[SteinerPoint] = []
        self.nexts: List[SteinerPoint] = []
        self.level: int = -1

    def __repr__(self) -> str:
        pin_name = self.Pin.name if self.Pin is not None else "—"
        return f"SteinerPoint(idx={self.idx}, pin={pin_name}, ({self.x},{self.y}))"


def match_pins_to_steiner_points(
    pins: List["Pin"],
    steiner_pts: List[SteinerPoint],
) -> List[SteinerPoint]:
    """
    Match circuit pins to Steiner-tree branch nodes by (x, y) coordinate.

    Each pin is matched to the nearest unmatched Steiner point at the same
    location.  Unmatched Steiner points (purely topological nodes) are kept
    with ``Pin = None``.

    Parameters
    ----------
    pins        : list of Pin objects for a single net.
    steiner_pts : list of SteinerPoint objects from the Steiner tree builder.

    Returns
    -------
    The same ``steiner_pts`` list, with ``.idx`` and ``.Pin`` fields set.
    """
    for pin in pins:
        pin.steiner_pt = None

    if not pins or not steiner_pts:
        return steiner_pts

    # Build a coord → [pin, …] map to handle duplicate coordinates gracefully.
    coord_map: Dict[tuple, List["Pin"]] = {}
    for pin in pins:
        x = pin.db_iterm.getAvgXY()[1]
        y = pin.db_iterm.getAvgXY()[2]
        coord_map.setdefault((x, y), []).append(pin)

    for i, stp in enumerate(steiner_pts):
        stp.idx = i
        bucket = coord_map.get((stp.x, stp.y))
        if bucket:
            pin = bucket.pop(0)
            stp.Pin = pin
            pin.steiner_pt = stp
            if not bucket:
                del coord_map[(stp.x, stp.y)]

    return steiner_pts


# ---------------------------------------------------------------------------
# Net
# ---------------------------------------------------------------------------

class Net:
    """Wrapper around an OpenDB dbNet with additional derived fields."""

    __slots__ = ("db_net", "pins", "steiner_pts", "driver_pin", "idx")

    def __init__(self, db_net) -> None:
        self.db_net = db_net
        self.pins: List[Pin] = []
        self.steiner_pts: List[SteinerPoint] = []
        self.driver_pin: Optional[Pin] = None
        self.idx: int = -1

    def set_pins(self, pins: List["Pin"]) -> None:
        self.pins = pins
        for p in pins:
            if p.io_type == "OUTPUT":
                self.driver_pin = p
                break

    # Keep old attribute name as an alias for backward compatibility with
    # helper functions that reference `net.steinerPoints`.
    @property
    def steinerPoints(self):  # noqa: N802
        return self.steiner_pts

    @steinerPoints.setter
    def steinerPoints(self, v):  # noqa: N802
        self.steiner_pts = v


# ---------------------------------------------------------------------------
# Pin
# ---------------------------------------------------------------------------

class Pin:
    """Wrapper around an OpenDB dbITerm (instance terminal)."""

    __slots__ = ("db_iterm", "name", "io_type", "gate", "net", "steiner_pt")

    # Alias so callers that use `pin.db_ITerm` still work.
    @property
    def db_ITerm(self):  # noqa: N802
        return self.db_iterm

    def __init__(self, db_iterm) -> None:
        self.db_iterm = db_iterm
        self.name: str = db_iterm.getMTerm().getName()
        self.io_type: str = db_iterm.getMTerm().getIoType()
        self.gate: Optional[Gate] = None
        self.net: Optional[Net] = None
        self.steiner_pt: Optional[SteinerPoint] = None

    # Legacy attribute aliases
    @property
    def IO(self) -> str:  # noqa: N802
        return self.io_type

    @property
    def Net(self):  # noqa: N802
        return self.net

    @Net.setter
    def Net(self, v):  # noqa: N802
        self.net = v

    @property
    def Gate(self):  # noqa: N802
        return self.gate

    @Gate.setter
    def Gate(self, v):  # noqa: N802
        self.gate = v

    @property
    def steinerPoint(self):  # noqa: N802
        return self.steiner_pt

    @steinerPoint.setter
    def steinerPoint(self, v):  # noqa: N802
        self.steiner_pt = v


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class Gate:
    """
    Wrapper around an OpenDB dbInst (cell instance).

    The ``equiv_masters`` list contains all functionally equivalent cell
    masters (drive-strength variants) as returned by OpenROAD's
    ``equivCells()`` API.  Gate sizing selects among these candidates.
    """

    __slots__ = (
        "db_inst", "db_master", "original_master",
        "pins", "input_pins", "output_pins",
        "equiv_masters", "idx", "unateness",
    )

    def __init__(self, db_inst, db_master) -> None:
        self.db_inst = db_inst
        self.db_master = db_master
        self.original_master = db_master
        self.pins: Dict[str, Pin] = {}
        self.input_pins: List[Pin] = []
        self.output_pins: List[Pin] = []
        self.equiv_masters: List = []
        self.idx: int = -1
        self.unateness: int = 0  # 0 = non-inverting, 1 = inverting

    # --- legacy aliases expected by downstream code ---

    @property
    def db_Inst(self):  # noqa: N802
        return self.db_inst

    @property
    def inputPins(self):  # noqa: N802
        return self.input_pins

    @property
    def outputPins(self):  # noqa: N802
        return self.output_pins

    @property
    def eqvMaster(self):  # noqa: N802
        return self.equiv_masters

    # --------------------------------------------------

    def set_pins(self, pins: List[Pin]) -> None:
        self.pins = {p.name: p for p in pins}
        self.input_pins = [p for p in pins if p.io_type == "INPUT"]
        self.output_pins = [p for p in pins if p.io_type == "OUTPUT"]

    # Legacy alias used by circuitMgr
    def setPins(self, pins):  # noqa: N802
        self.set_pins(pins)

    def set_unateness(self, value: int) -> None:
        self.unateness = value

    def set_equiv_masters(self, timing) -> None:
        """Populate ``equiv_masters`` using OpenROAD's equivCells API."""
        self.equiv_masters = [self.db_master]
        for m in timing.equivCells(self.db_master):
            if m.isBlock():
                break
            if m.getName() == self.db_master.getName():
                continue
            self.equiv_masters.append(m)

    # Legacy alias
    def setEqvMaster(self, timing):  # noqa: N802
        self.set_equiv_masters(timing)

    def swap_to(self, new_master) -> bool:
        """Swap this gate to a different drive-strength variant."""
        if new_master not in self.equiv_masters:
            return False
        self.db_master = new_master
        self.db_inst.swapMaster(new_master)
        return True

    # Legacy alias
    def swapCell(self, new_master):  # noqa: N802
        self.swap_to(new_master)

    def __repr__(self) -> str:
        return f"Gate({self.db_inst.getName()}, master={self.db_master.getName()})"
