#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path

import openroad as ord
from openroad import Tech, Design, Timing

from src.engine.pipeline import OptPipeline
from src.io.loader import load_design


def parse_args():
    parser = argparse.ArgumentParser(description="Running PlaceOpt flow")
    parser.add_argument("design_name", nargs="?")
    parser.add_argument("tech_dir",    nargs="?")
    parser.add_argument("design_dir",  nargs="?")
    parser.add_argument("output_dir",  nargs="?")
    return parser.parse_args()


class MetricsCollector:
    """Collects baseline timing and power metrics on a fresh design load."""

    NUM_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

    @staticmethod
    def gather(tech_dir, design_name, design_dir):
        tech, design = load_design(design_name, design_dir, tech_dir)
        design.evalTclString("set_ideal_network [all_clocks]")
        design.evalTclString(
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V "
            "-resistance kOhm -distance um -power mW"
        )
        design.evalTclString("set_units -power mW")
        design.evalTclString("set_routing_layers -signal M2-M9 -clock M2-M9")
        design.evalTclString("detailed_placement")

        try:
            design.evalTclString("global_route -skip_large_fanout_nets 300 -allow_congestion")
        except Exception as e:
            print(f"[INFO] global_route failed, retrying: {e}")
            design.evalTclString("detailed_placement")
            design.evalTclString("global_route -skip_large_fanout_nets 300 -allow_congestion")

        design.evalTclString("estimate_parasitics -global_routing")

        macro_names = []
        try:
            for inst in design.getBlock().getInsts():
                if inst.getMaster().isBlock():
                    macro_names.append(inst.getName())
        except Exception as e:
            print(f"[INFO] macro scan failed: {e}")

        macro_list = " ".join([f"{{{n}}}" for n in macro_names])
        macro_tcl = (
            f"set __macro_insts [get_cells -hierarchical {{{macro_list}}}]"
            if macro_names else "set __macro_insts {}"
        )

        tcl = f"""
            sta::redirect_string_begin
            report_tns
            set __tns_out [sta::redirect_string_end]
            sta::redirect_string_begin
            report_power
            set __pwr_out [sta::redirect_string_end]
            sta::redirect_string_begin
            {macro_tcl}
            if {{[llength $__macro_insts] > 0}} {{
                report_power -instances $__macro_insts
            }} else {{
                puts "Total 0 0 0 0"
            }}
            set __macro_pwr_out [sta::redirect_string_end]
            return "$__tns_out\\n__OR_SPLIT__\\n$__pwr_out\\n__OR_SPLIT__\\n$__macro_pwr_out"
        """
        raw = design.evalTclString(tcl)
        return MetricsCollector._parse(raw)

    @staticmethod
    def _parse(raw):
        tns = leakage = switching = internal = 0.0
        macro_leakage = macro_switching = macro_internal = 0.0

        if not raw:
            return tns, leakage, switching, internal, macro_leakage, macro_switching, macro_internal

        parts = raw.split("__OR_SPLIT__")
        tns_report   = parts[0].strip() if len(parts) >= 1 else ""
        power_report = parts[1].strip() if len(parts) >= 2 else ""
        macro_report = parts[2].strip() if len(parts) >= 3 else ""

        num = MetricsCollector.NUM_PATTERN
        m = re.search(rf"^\s*tns\s+\S+\s+({num})", tns_report, re.MULTILINE)
        if not m:
            m = re.search(num, tns_report)
        if m:
            tns = float(m.group(1) if m.lastindex else m.group(0))

        def _parse_power_line(report, prefix):
            for line in report.splitlines():
                s = line.lstrip()
                if s.startswith(prefix):
                    fields = s.split()
                    if len(fields) >= 4:
                        return float(fields[1]), float(fields[2]), float(fields[3])
            return 0.0, 0.0, 0.0

        internal, switching, leakage = _parse_power_line(power_report, "Total")
        mi, ms, ml = _parse_power_line(power_report, "Macro")
        if mi == 0.0 and ms == 0.0:
            mi, ms, ml = _parse_power_line(macro_report, "Total")
        macro_internal, macro_switching, macro_leakage = mi, ms, ml

        return tns, leakage, switching, internal, macro_leakage, macro_switching, macro_internal

    @staticmethod
    def weight_matrix(tns, leakage, switching, internal,
                      macro_leakage, macro_switching, macro_internal):
        """Compute [tns_weight, leakage, switching, internal] for the optimizer."""
        timing_ratio = 0.1
        macro_dynamic = macro_switching + macro_internal
        total_dynamic = switching + internal
        if total_dynamic > 0 and (macro_dynamic / total_dynamic) > 0.5:
            timing_ratio = 0.01
        elif leakage > 0 and (macro_leakage / leakage) > 0.5:
            timing_ratio = 0.01
        print("Timing driven ratio:", timing_ratio)
        tns_clamped = min(tns, -0.5)
        return [tns_clamped * 1e-9 * timing_ratio, leakage, switching, internal]


class FlowRunner:
    """Drives the full PlaceOpt optimization pipeline for one design."""

    def __init__(self, args):
        self.design_name = args.design_name
        self.tech_dir    = args.tech_dir.rstrip("/") + "/"
        self.design_dir  = args.design_dir.rstrip("/") + "/"
        self.output_dir  = args.output_dir.rstrip("/") + "/"

    def execute(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        tns, leakage, switching, internal, macro_leakage, macro_switching, macro_internal = \
            MetricsCollector.gather(self.tech_dir, self.design_name, self.design_dir)
        print(f"[INFO] BASELINE: TNS={tns}  leakage={leakage}  switching={switching}  internal={internal}")
        print(f"[INFO] BASELINE MACRO POWER: leakage={macro_leakage}  switching={macro_switching}  "
              f"internal={macro_internal}")

        matrix = MetricsCollector.weight_matrix(
            tns, leakage, switching, internal, macro_leakage, macro_switching, macro_internal
        )

        pipeline = OptPipeline(self.design_name, self.design_dir, self.tech_dir, self.output_dir)
        pipeline.load_design()
        pipeline.setup_design()
        pipeline.evaluate_design()
        pipeline.initialize()
        pipeline.write_eqv_csv(self.design_dir + "/csv/")

        pipeline.repair_design()
        pipeline.rebuild_network()
        pipeline.run_gradient_opt(
            iteration=100,
            lr=0.5,
            original_buffer_weight=-12,
            matrix=matrix,
        )
        pipeline.evaluate_sta()

        pipeline.rebuild_network()
        pipeline.run_gradient_opt(
            iteration=100,
            lr=0.5,
            original_buffer_weight=-18,
            matrix=matrix,
        )

        pipeline.evaluate_sta()
        pipeline.default_flow()

        pipeline.write_eqv_csv(self.output_dir + "/csv/")
        pipeline.evaluate_design()
        pipeline.output_results()


if __name__ == "__main__":
    args = parse_args()
    FlowRunner(args).execute()
    os._exit(0)
