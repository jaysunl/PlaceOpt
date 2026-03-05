"""
placeopt.main — Entry point for the ISPD26 post-placement optimization flow.

Invocation
----------
::

    python -m placeopt.main \\
        --design <design_name> \\
        --design_dir <path/to/design> \\
        --tech_dir <path/to/tech> \\
        --output_dir <path/to/output> \\
        [--device cuda:0] \\
        [--epochs 150]

Flow summary
------------
1. Load design (LEF/Liberty/DEF/SDC) via OpenROAD.
2. Record baseline TNS + power from OpenSTA.
3. Run OpenROAD's built-in ``repair_design`` for DRV fixing.
4. Build the CellLibrary + SteinerNetworkBuilder.
5. Build STAGraph tensors and construct the TimingEngine.
6. Run two gradient-optimization passes with different buffer biases.
7. Apply discrete gate-sizing + buffering decisions to OpenDB.
8. Run detailed placement (legalization).
9. Write output .def and .v files.
"""

from __future__ import annotations

import argparse
import time
import gc
from typing import Optional, Tuple

import torch
from openroad import Timing  # type: ignore

from placeopt.io.design import (
    load_design, write_results,
    apply_gate_sizing, apply_cell_positions, apply_buffering,
    get_rss_mb,
)
from placeopt.circuit.library import CellLibrary
from placeopt.circuit.steiner import SteinerNetworkBuilder
from placeopt.timing.graph import STAGraphBuilder
from placeopt.timing.engine import TimingEngine
from placeopt.opt.optimizer import GradientOptimizer, OptConfig
from placeopt.eda.placement import run_detailed_placement


# ---------------------------------------------------------------------------
# Baseline measurement
# ---------------------------------------------------------------------------

def measure_baseline(design) -> Tuple[float, float]:
    """
    Run OpenSTA and return (TNS in seconds, dynamic power in Watts).

    Uses OpenROAD's ``report_tns``/``report_power`` TCL commands.
    """
    timing = Timing(design)
    tns_s   = timing.getTNS() or 0.0
    power_w = timing.getPower() or 0.0
    return float(tns_s), float(power_w)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(design, timing, device: torch.device,
                init_gate_w: float = 6.0, init_buf_w: float = -12.0):
    """
    Build CellLibrary → SteinerNetworkBuilder → STAGraph → TimingEngine.

    Returns
    -------
    cell_lib, steiner_net, engine : the three constructed objects.
    """
    rss = get_rss_mb()
    if rss:
        print(f"[MEM] build_graph start: {rss:.1f} MB")

    t0 = time.perf_counter()

    cell_lib = CellLibrary(design)
    print(f"[TIME] CellLibrary: {time.perf_counter()-t0:.2f}s  "
          f"({len(cell_lib.signal_gates)} signal gates, "
          f"{len(cell_lib.signal_nets)} signal nets)")

    t1 = time.perf_counter()
    steiner_net = SteinerNetworkBuilder(design, timing)
    steiner_net.build(cell_lib.signal_nets, cell_lib.signal_gates,
                      cell_lib.TensorTableModels)
    print(f"[TIME] SteinerNetworkBuilder: {time.perf_counter()-t1:.2f}s")

    t1 = time.perf_counter()
    graph = STAGraphBuilder().build(
        init_gate_w, init_buf_w, steiner_net, cell_lib, timing, device)
    print(f"[TIME] STAGraphBuilder: {time.perf_counter()-t1:.2f}s")

    engine = TimingEngine(graph).to(device)
    print(f"[TIME] Total graph build: {time.perf_counter()-t0:.2f}s")

    rss2 = get_rss_mb()
    if rss2:
        print(f"[MEM] build_graph end: {rss2:.1f} MB")

    return cell_lib, steiner_net, engine


# ---------------------------------------------------------------------------
# Write-back helpers
# ---------------------------------------------------------------------------

def _apply_decisions(engine: TimingEngine, cell_lib, steiner_net) -> None:
    """Discretize soft decisions and apply them to OpenDB."""
    engine.discretize()
    gd = engine.gd
    apply_cell_positions(cell_lib, gd.cell_xy)
    # Gate sizing: apply argmax master selection.
    with torch.no_grad():
        from placeopt.timing.engine import _segment_softmax
        w = _segment_softmax(gd.U_flat, gd.u_gate_id, gd.u_gate_len)
    apply_gate_sizing_from_w(cell_lib, gd, w)
    # Buffer insertion.
    _apply_buffering_from_engine(engine, cell_lib, steiner_net)


def apply_gate_sizing_from_w(cell_lib, gd, w: torch.Tensor) -> None:
    """Apply the soft gate-size distribution (argmax) to OpenDB masters."""
    w_cpu = w.detach().cpu()
    u_gate_off = gd.u_gate_off.cpu().tolist()
    for g in cell_lib.signal_gates:
        gi = g.idx
        off = int(u_gate_off[gi])
        end = int(u_gate_off[gi + 1])
        w_gate = w_cpu[off:end]
        idx = int(w_gate.argmax().item())
        new_master = g.equiv_masters[idx]
        if new_master.getName() != g.db_master.getName():
            g.swap_to(new_master)


def _apply_buffering_from_engine(engine: TimingEngine, cell_lib, steiner_net) -> None:
    """Insert buffers into the OpenDB netlist from the optimized buffering_tensor."""
    from placeopt.eda.buffering import commit_buffering
    import torch
    b = torch.sigmoid(engine.buffering_tensor).detach().cpu()
    buf_master = engine.gd.chosen_buffer

    should_buffer = {stp.idx: bool(b[stp.idx].item() > 0.5)
                     for stp in steiner_net.stp_list}

    for net in cell_lib.signal_nets:
        if any(should_buffer.get(stp.idx, False) for stp in net.steiner_pts):
            commit_buffering(net, should_buffer, buf_master)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def run_flow(
    design_name: str,
    design_dir: str,
    tech_dir: str,
    output_dir: str,
    device: torch.device,
    epochs: int = 150,
) -> None:
    """
    Execute the full post-placement buffering and sizing flow.

    Two optimization passes are run:
    * Pass 1: buffer bias = −12  (moderate insertion freedom).
    * Pass 2: buffer bias = −18  (conservative; focuses on sizing refinement).
    """
    t_total = time.perf_counter()

    # ── Step 1: Load design ─────────────────────────────────────────────
    print("[FLOW] Loading design...")
    tech, design = load_design(design_name, design_dir, tech_dir)
    timing = Timing(design)
    design.evalTclString("estimate_parasitics -placement")

    # ── Step 2: Baseline ─────────────────────────────────────────────────
    print("[FLOW] Measuring baseline...")
    baseline_tns, baseline_power = measure_baseline(design)
    print(f"[FLOW] Baseline  TNS={baseline_tns:.3e}  Power={baseline_power:.3e}")

    # ── Step 3: DRV repair ───────────────────────────────────────────────
    print("[FLOW] Running repair_design...")
    design.evalTclString("repair_design")
    design.evalTclString("estimate_parasitics -placement")

    # ── Step 4: Build graph ──────────────────────────────────────────────
    print("[FLOW] Building timing graph...")
    cell_lib, steiner_net, engine = build_graph(design, timing, device)

    # ── Step 5: Optimization pass 1 ──────────────────────────────────────
    print("[FLOW] Optimization pass 1 (buf_bias=-12)...")
    cfg1         = OptConfig()
    cfg1.epochs  = epochs
    opt1         = GradientOptimizer(engine, cfg1, device)
    opt1.run_pass(baseline_tns, baseline_power, buf_bias=-12.0)

    # Apply pass-1 decisions and rebuild the graph.
    _apply_decisions(engine, cell_lib, steiner_net)
    design.evalTclString("estimate_parasitics -placement")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Step 6: Optimization pass 2 ──────────────────────────────────────
    print("[FLOW] Rebuilding graph for pass 2...")
    cell_lib2, steiner_net2, engine2 = build_graph(design, timing, device)

    print("[FLOW] Optimization pass 2 (buf_bias=-18)...")
    cfg2         = OptConfig()
    cfg2.epochs  = epochs
    opt2         = GradientOptimizer(engine2, cfg2, device)
    opt2.run_pass(baseline_tns, baseline_power, buf_bias=-18.0)

    _apply_decisions(engine2, cell_lib2, steiner_net2)

    # ── Step 7: Legalization ─────────────────────────────────────────────
    print("[FLOW] Running detailed placement...")
    rc, failed = run_detailed_placement(design)
    if failed:
        print(f"[WARN] {len(failed)} instances failed DPL.")

    # ── Step 8: Report + write output ────────────────────────────────────
    design.evalTclString("estimate_parasitics -placement")
    tns_final, power_final = measure_baseline(design)
    print(f"[FLOW] Final  TNS={tns_final:.3e}  Power={power_final:.3e}")
    print(f"[FLOW] TNS improvement  : {tns_final - baseline_tns:.3e}")
    print(f"[FLOW] Power improvement: {baseline_power - power_final:.3e}")

    write_results(design, output_dir, design_name)
    print(f"[FLOW] Total time: {time.perf_counter()-t_total:.1f}s")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="placeopt: post-placement gate sizing & buffer insertion")
    p.add_argument("--design",     required=True, help="Top-module name")
    p.add_argument("--design_dir", required=True, help="Path to design directory")
    p.add_argument("--tech_dir",   required=True, help="Path to technology directory")
    p.add_argument("--output_dir", required=True, help="Path for output files")
    _default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device",     default=_default_device,
                   help=f"PyTorch device (default: auto → {_default_device})")
    p.add_argument("--epochs",     type=int, default=150,
                   help="Optimization epochs per pass (default: 150)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dev  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_flow(
        design_name=args.design,
        design_dir=args.design_dir,
        tech_dir=args.tech_dir,
        output_dir=args.output_dir,
        device=dev,
        epochs=args.epochs,
    )
