# BSD 3-Clause License
#
# Copyright (c) 2024, ASU-VDA-Lab
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.
import openroad as ord
from openroad import Tech, Design
import odb
from pathlib import Path
from collections import defaultdict


_LAST_TOP = None
_LAST_LIBS = None


def load_design(design_name, design_dir, tech_dir):
    """Read liberty/LEF/DEF/verilog/SDC and set up global RC values."""
    global _LAST_TOP, _LAST_LIBS

    tech = Tech()
    lib_dir = Path(tech_dir) / "lib"
    lef_dir = Path(tech_dir) / "lef"

    def _lib_sort_key(path):
        name = path.name
        if name.startswith("asap7sc7p5t_"):
            return (0, name)
        if name.startswith("sram_") or name.startswith("fakeram_"):
            return (1, name)
        return (2, name)

    lib_files = sorted(lib_dir.glob("*.lib"), key=_lib_sort_key)
    for lib_file in lib_files:
        tech.readLiberty(lib_file.as_posix())

    tech_lef = lef_dir / "asap7_tech_1x_201209.lef"
    tech.readLef(tech_lef.as_posix())
    for lef_file in sorted(lef_dir.glob("*.lef"), key=lambda p: p.name):
        if lef_file == tech_lef:
            continue
        tech.readLef(lef_file.as_posix())

    design = Design(tech)
    design.readVerilog(f"{design_dir}/contest.v")
    design.evalTclString(f"read_def {design_dir}/contest.def")
    design.evalTclString(f"read_sdc {design_dir}/contest.sdc")

    design.evalTclString("""
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
    """)

    block = design.getBlock()
    vdd = block.findNet("VDD") or odb.dbNet_create(block, "VDD")
    vdd.setSpecial()
    vdd.setSigType("POWER")
    vss = block.findNet("VSS") or odb.dbNet_create(block, "VSS")
    vss.setSpecial()
    vss.setSigType("GROUND")
    block.addGlobalConnect(None, ".*", "VDD", vdd, True)
    block.addGlobalConnect(None, ".*", "VSS", vss, True)
    block.globalConnect()

    _LAST_TOP = block.getName() if block else design_name
    _LAST_LIBS = [p.as_posix() for p in lib_files]
    return tech, design


def build_libcell_dict(filename):
    """Build a mapping from cell name to equivalence group."""
    id_to_names = defaultdict(list)
    with open(filename) as f:
        for line in f:
            parts = line.split(",")
            id_to_names[parts[1][:-1]].append(parts[0])
    libcell_dict = {}
    for names in id_to_names.values():
        for name in names:
            libcell_dict[name] = names
    return libcell_dict


def get_output_load_pin_cap(pin, corner, timing):
    """Return total downstream load capacitance for an output pin."""
    if not pin.isOutputSignal():
        return -1
    cap = 0
    for net_pin in pin.getNet().getITerms():
        if net_pin.isInputSignal():
            cap += timing.getPortCap(net_pin, corner, timing.Max)
    return cap
