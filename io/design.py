"""
io.design — OpenROAD design I/O helpers.

This module handles:
* Loading tech/LEF/Liberty/DEF/SDC files into an OpenROAD ``Design`` object.
* Setting the standard wire RC parameters for ASAP7.
* Writing the optimized design back out (.def + .v).
* Memory usage reporting.
"""

from __future__ import annotations

import os
from pathlib import Path

import openroad as ord  # type: ignore
from openroad import Tech, Design, Timing  # type: ignore
import odb  # type: ignore


# ---------------------------------------------------------------------------
# Design loading
# ---------------------------------------------------------------------------

def load_design(design_name: str, design_dir: str, tech_dir: str):
    """
    Load an ISPD26 contest design into OpenROAD.

    Reads, in order:
    * Liberty files from ``tech_dir/lib/``  (stdcell libs first).
    * ``tech_dir/lef/asap7_tech_1x_201209.lef`` (technology LEF).
    * All other ``*.lef`` files in ``tech_dir/lef/``.
    * ``design_dir/contest.v``   (Verilog netlist).
    * ``design_dir/contest.def`` (placed DEF).
    * ``design_dir/contest.sdc`` (timing constraints).

    Then sets per-layer RC values for ASAP7 and establishes power/ground
    global connections.

    Parameters
    ----------
    design_name : top-module name.
    design_dir  : path to the design directory (contains .v, .def, .sdc).
    tech_dir    : path to the technology directory (contains lib/, lef/).

    Returns
    -------
    (tech, design) : (openroad.Tech, openroad.Design)
    """
    tech = Tech()
    lib_dir = Path(tech_dir) / "lib"
    lef_dir = Path(tech_dir) / "lef"

    # Stdcell libs first so that their time-unit settings (ps) take precedence
    # over SRAM/fakeram models.
    def _lib_sort_key(p: Path):
        name = p.name
        if name.startswith("asap7sc7p5t_"):
            return (0, name)
        if name.startswith(("sram_", "fakeram_")):
            return (1, name)
        return (2, name)

    for lib_file in sorted(lib_dir.glob("*.lib"), key=_lib_sort_key):
        tech.readLiberty(lib_file.as_posix())

    # Tech LEF must come before cell LEFs.
    tech_lef = lef_dir / "asap7_tech_1x_201209.lef"
    tech.readLef(tech_lef.as_posix())
    for lef_file in sorted(lef_dir.glob("*.lef"), key=lambda p: p.name):
        if lef_file == tech_lef:
            continue
        tech.readLef(lef_file.as_posix())

    design = Design(tech)
    design.readVerilog(str(Path(design_dir) / "contest.v"))
    design.evalTclString(f"read_def {Path(design_dir) / 'contest.def'}")
    design.evalTclString(f"read_sdc {Path(design_dir) / 'contest.sdc'}")

    _set_asap7_wire_rc(design)
    _connect_power_ground(design)

    return tech, design


def _set_asap7_wire_rc(design: Design) -> None:
    """Apply ASAP7 per-layer wire RC values via TCL."""
    rc_tcl = """
set_layer_rc -layer M1 -resistance 7.04175E-02 -capacitance 1e-10
set_layer_rc -layer M2 -resistance 4.62311E-02 -capacitance 1.84542E-01
set_layer_rc -layer M3 -resistance 3.63251E-02 -capacitance 1.53955E-01
set_layer_rc -layer M4 -resistance 2.03083E-02 -capacitance 1.89434E-01
set_layer_rc -layer M5 -resistance 1.93005E-02 -capacitance 1.71593E-01
set_layer_rc -layer M6 -resistance 1.18619E-02 -capacitance 1.76146E-01
set_layer_rc -layer M7 -resistance 1.25311E-02 -capacitance 1.47030E-01
set_wire_rc -signal -resistance 3.23151E-02 -capacitance 1.73323E-01
set_wire_rc -clock  -resistance 5.13971E-02 -capacitance 1.44549E-01
set_layer_rc -via V1 -resistance 1.72E-02
set_layer_rc -via V2 -resistance 1.72E-02
set_layer_rc -via V3 -resistance 1.72E-02
set_layer_rc -via V4 -resistance 1.18E-02
set_layer_rc -via V5 -resistance 1.18E-02
set_layer_rc -via V6 -resistance 8.20E-03
set_layer_rc -via V7 -resistance 8.20E-03
set_layer_rc -via V8 -resistance 6.30E-03
"""
    design.evalTclString(rc_tcl)


def _connect_power_ground(design: Design) -> None:
    """Create VDD/VSS special nets and add global connect rules."""
    block = design.getBlock()

    for net_name, sig_type in (("VDD", "POWER"), ("VSS", "GROUND")):
        net = block.findNet(net_name) or odb.dbNet_create(block, net_name)
        net.setSpecial()
        net.setSigType(sig_type)

    block.addGlobalConnect(None, ".*", "VDD", block.findNet("VDD"), True)
    block.addGlobalConnect(None, ".*", "VSS", block.findNet("VSS"), True)
    block.globalConnect()


# ---------------------------------------------------------------------------
# Design output
# ---------------------------------------------------------------------------

def write_results(design: Design, output_dir: str, design_name: str) -> None:
    """
    Write the optimized design to ``output_dir/contest.def`` and
    ``output_dir/contest.v``.

    Parameters
    ----------
    design      : openroad.Design with the final netlist.
    output_dir  : directory where output files will be written.
    design_name : Verilog top-module name.
    """
    os.makedirs(output_dir, exist_ok=True)
    def_path = str(Path(output_dir) / "contest.def")
    v_path   = str(Path(output_dir) / "contest.v")

    design.evalTclString(f"write_def {def_path}")
    design.evalTclString(f"write_verilog {v_path}")
    print(f"[INFO] Results written to {output_dir}")


# ---------------------------------------------------------------------------
# OpenROAD write-back helpers
# ---------------------------------------------------------------------------

def apply_gate_sizing(cell_lib, w_hard: "torch.Tensor") -> None:
    """
    Apply the argmax gate-sizing decision from ``w_hard`` (one-hot weights)
    to the OpenDB instances.

    Parameters
    ----------
    cell_lib : CellLibrary
    w_hard   : [total_u]  one-hot weight tensor (after ``discretize()``).
    """
    from placeopt.timing.engine import _segment_softmax
    import torch
    gd = cell_lib  # compatible with either CellLibrary or STAGraph
    # For each gate, the chosen master is the one with weight ≈ 1.
    u = w_hard
    for g in cell_lib.signal_gates:
        gi = g.idx
        off = int(gd.u_gate_off[gi].item())
        end = int(gd.u_gate_off[gi + 1].item())
        w_gate = u[off:end]
        chosen_idx = int(w_gate.argmax().item())
        chosen_master = g.equiv_masters[chosen_idx]
        if chosen_master.getName() != g.db_master.getName():
            g.swap_to(chosen_master)


def apply_cell_positions(cell_lib, cell_xy: "torch.Tensor") -> None:
    """
    Write optimized cell (x, y) positions back to OpenDB.

    Parameters
    ----------
    cell_lib : CellLibrary
    cell_xy  : [G, 2]  gate positions in DBU.
    """
    xy_cpu = cell_xy.detach().cpu()
    for g in cell_lib.signal_gates:
        gi = g.idx
        if gi < 0:
            continue
        x = int(xy_cpu[gi, 0].item())
        y = int(xy_cpu[gi, 1].item())
        g.db_inst.setLocation(x, y)
        g.db_inst.setPlacementStatus("PLACED")


def apply_buffering(
    cell_lib,
    steiner_net,
    b_tensor: "torch.Tensor",
    buf_threshold: float = 0.5,
) -> None:
    """
    Insert buffers into the OpenDB netlist wherever ``sigmoid(b_tensor) > threshold``.

    Parameters
    ----------
    cell_lib      : CellLibrary
    steiner_net   : SteinerNetworkBuilder
    b_tensor      : [N]  buffering logits (before sigmoid).
    buf_threshold : insertion threshold on sigmoid output.
    """
    import torch
    from placeopt.eda.buffering import commit_buffering

    b_flat = torch.sigmoid(b_tensor).detach().cpu()
    should_buffer = {
        stp.idx: bool(b_flat[stp.idx].item() > buf_threshold)
        for stp in steiner_net.stp_list
    }
    buf_master = cell_lib.chosen_buf if hasattr(cell_lib, "chosen_buf") else None
    if buf_master is None:
        print("[WARN] apply_buffering: no chosen buffer master available.")
        return

    for net in cell_lib.signal_nets:
        if any(should_buffer.get(stp.idx, False) for stp in net.steiner_pts):
            commit_buffering(net, should_buffer, buf_master)


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------

def get_rss_mb() -> float | None:
    """Return process resident set size in MB (Linux /proc/self/status)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return None


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

def get_time_unit(design: Design) -> float:
    """Return the STA time unit in seconds (from 'sta::unit_scaled_suffix time')."""
    unit_str = design.evalTclString("sta::unit_scaled_suffix time").strip()
    units = {"ps": 1e-12, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0, "fs": 1e-15}
    if unit_str not in units:
        raise ValueError(f"Unknown STA time unit: {unit_str!r}")
    return units[unit_str]
