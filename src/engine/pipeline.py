import re
import gc
import time
from pathlib import Path

import odb
import torch
import openroad as ord
from openroad import Tech, Design, Timing

from src.sta.arc_model import effective_load_cap
from src.sta.arc_model import *
from src.sta.params import *
from src.util.helpers import *
from src.util.buffer_ops import buffering_update
from src.util.placement import detailed_placement
from src.io.loader import load_design

from src.db.netlist import SteinerNode
from src.db.cell_db import NetlistDB
from src.analysis.gsta import GSTA
from src.analysis.tree_builder import TreeBuilder


class OptPipeline:
    """End-to-end optimization pipeline wrapping NetlistDB, GSTA, and TreeBuilder."""

    def __init__(self, design_name, design_dir, tech_dir, output_dir):
        self.design_name = design_name
        self.design_dir  = design_dir
        self.tech_dir    = tech_dir
        self.output_dir  = output_dir

        self.tech   = None
        self.design = None
        self.db     = None
        self.timing = None

        self.cell_db      = None
        self.sta_engine   = None
        self.optimizer    = None
        self.tree_builder = None

    # ------------------------------------------------------------------
    # Design I/O
    # ------------------------------------------------------------------

    def load_design(self):
        self.tech, self.design = load_design(self.design_name, self.design_dir, self.tech_dir)
        self.db     = ord.get_db()
        self.timing = Timing(self.design)

    def write_eqv_csv(self, folder_path):
        if self.design is None:
            raise RuntimeError("Design not loaded. Call load_design() first.")

        out_dir = Path(folder_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        or_utils = Path(__file__).resolve().parents[2] / "src/io/or_utils.tcl"
        if not or_utils.exists():
            raise FileNotFoundError(f"Missing or_utils.tcl: {or_utils}")

        node_file = out_dir / "node.csv"
        net_file  = out_dir / "nets.csv"
        self.design.evalTclString(
            f"source {{{or_utils}}}\n"
            f"write_node_and_net_files {{{node_file}}} {{{net_file}}}\n"
        )

    def output_results(self, file_name=None):
        if file_name is None:
            file_name = self.design_name
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        out_def = self.output_dir + file_name + ".def"
        out_v   = self.output_dir + file_name + ".v"
        print(f"[INFO] write_def     -> {out_def}")
        self.design.evalTclString(f"write_def {out_def}")
        print(f"[INFO] write_verilog -> {out_v}")
        self.design.evalTclString(f"write_verilog {out_v}")

    # ------------------------------------------------------------------
    # Design setup & evaluation
    # ------------------------------------------------------------------

    def setup_design(self):
        self.design.evalTclString("set_ideal_network [all_clocks]")
        self.design.evalTclString(
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um -power mW"
        )
        self.design.evalTclString("set_units -power mW")
        self.design.evalTclString("estimate_parasitics -placement")
        self.design.evalTclString("set_operating_conditions")
        self.design.evalTclString("set_thread_count 8")

    def evaluate_design(self):
        self.design.evalTclString("remove_parasitics -all")
        self.design.evalTclString("estimate_parasitics -placement")
        self.design.evalTclString("report_units")
        self.design.evalTclString("report_tns")
        self.design.evalTclString("report_wns")
        self.design.evalTclString("report_power")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        t = time.time()
        self.cell_db = NetlistDB(self.design)
        print(f"[INFO] NetlistDB initialized in {time.time() - t:.2f}s")

        t = time.time()
        self.tree_builder = TreeBuilder(self.design, self.timing)
        print(f"[INFO] TreeBuilder initialized in {time.time() - t:.2f}s")

        t = time.time()
        self.sta_engine = GSTA(self.design, self.timing)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")
        self.sta_engine.to(device)
        self.sta_engine.setLibrary(self.cell_db)
        self.sta_engine.setSTPNetwork(self.tree_builder)
        print(f"[INFO] GSTA initialized in {time.time() - t:.2f}s")

    def rebuild_network(self):
        if self.sta_engine is not None:
            self.sta_engine.release()
        gc.collect()
        self.cell_db.reinitialize()
        self.tree_builder.build_network(
            self.cell_db.signal_nets,
            self.cell_db.signal_gates,
            self.cell_db.arc_luts,
        )
        if self.sta_engine is not None:
            self.sta_engine.setLibrary(self.cell_db)
            self.sta_engine.setSTPNetwork(self.tree_builder)

    # ------------------------------------------------------------------
    # Optimization passes
    # ------------------------------------------------------------------

    def repair_design(self):
        t = time.time()
        print("[INFO] repair_design")
        self.design.evalTclString("repair_design")
        print(f"[INFO] repair_design done in {time.time() - t:.2f}s")

    def repair_timing(self):
        print("[INFO] repair_timing -setup -skip_gate_cloning -skip_pin_swap")
        self.design.evalTclString("repair_timing -setup -skip_gate_cloning -skip_pin_swap")

    def default_flow(self):
        print("[INFO] detailed_placement")
        rc, failed = detailed_placement(self.design, self.output_dir)
        if rc != 0:
            print(f"[WARN] detailed_placement failed; {len(failed)} instances reported.")

    def run_gradient_opt(
        self,
        iteration=20,
        quantified_penalty_ratio=0.001,
        movement_penalty_ratio=0.01,
        density_penalty_ratio=75.0,
        lr=0.5,
        decay_lr=0.988,
        matrix=None,
        original_gate_weight=6.0,
        original_buffer_weight=-8.0,
    ):
        if matrix is None:
            matrix = [1.0, 1.0, 1.0, 1.0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sta_engine.to(device)
        self.sta_engine.tensor_init(
            origin_gate_weights=original_gate_weight,
            origin_buffer_weights=original_buffer_weight,
            device=device,
        )

        graph_data = getattr(self.sta_engine, "graph_data", None)
        if graph_data is not None:
            bbox = getattr(graph_data, "boundary", None)
            if bbox is not None and bbox.numel() >= 4:
                b = bbox.detach().cpu()
                print(f"[INFO] placement boundary (DBU): "
                      f"x_min={b[0,0]:.0f}, y_min={b[0,1]:.0f}, "
                      f"x_max={b[1,0]:.0f}, y_max={b[1,1]:.0f}")

        size_opt = torch.optim.Adam([self.sta_engine.U_flat],           lr=lr)
        pos_opt  = torch.optim.Adam([self.sta_engine.cell_xy],          lr=lr * 1000)
        buf_opt  = torch.optim.Adam([self.sta_engine.buffering_tensor], lr=lr)

        size_sched = torch.optim.lr_scheduler.StepLR(size_opt, step_size=1, gamma=decay_lr)
        xy_sched   = torch.optim.lr_scheduler.StepLR(pos_opt,  step_size=1, gamma=decay_lr)
        buf_sched  = torch.optim.lr_scheduler.StepLR(buf_opt,  step_size=1, gamma=decay_lr)

        b_tns  = matrix[0]
        b_leak = matrix[1]  
        b_sp   = matrix[2]
        density_threshold = 0.6

        b_leak_W = getattr(self.sta_engine.graph_data, "baseline_leakage_W", None)
        if b_leak_W is None or b_leak_W == 0.0:
            b_leak_W = max(b_leak * 1e-3, 1e-12)  # fallback: convert mW baseline
        b_leak_t = torch.tensor(b_leak_W, dtype=torch.float32, device=device)

        exe_time = time.time()
        with torch.no_grad():
            init_tns, _, init_sp, init_leak, _, b_d = self.sta_engine()
            self.report_density(b_d)
            # leakage is over-estimated so weight it less
            obj_term = 80 * init_tns / b_tns + 40 * init_sp / b_sp + 20 * init_leak / b_leak
            b_ovfl = torch.tensor(DENSITY_BINS * DENSITY_BINS * density_threshold, device=device)
            print(f"[INFO] initial overflow sum: {b_ovfl.item():.4f}")
            # print(f"[INFO] baseline leakage: {init_leak.item()*1e3:.4f}mW  (graph)  "
            #       f"{b_leak:.4f}mW  (report_power)")
            penalty_scale = quantified_penalty_ratio * obj_term

        penalty_tensor = torch.tensor([penalty_scale], device=device)
        penalty_inc    = torch.tensor([1.0 / decay_lr], device=device)

        for epoch in range(iteration):
            tns, wns, switching_power, leakage_power, quantified_penalty, density = self.sta_engine()
            ovfl = (density - density_threshold).clamp_min(0).square().sum()
            disp_avg, disp_max = self.sta_engine.forward_movement_displacement()
            disp_pen = movement_penalty_ratio * disp_avg

            loss = (
                80 * (tns - b_tns) / b_tns
                + 40 * (switching_power - b_sp) / b_sp
                + 20 * (leakage_power - b_leak_t) / b_leak_t
                + penalty_tensor * quantified_penalty
                + density_penalty_ratio * ovfl / b_ovfl
                + disp_pen
            )
            actual_obj = (80 * (tns - b_tns) / b_tns
                          + 40 * (switching_power - b_sp) / b_sp
                          + 20 * (leakage_power - b_leak_t) / b_leak_t)

            print(
                f"[INFO] Iteration {epoch + 1}: "
                f"TNS={tns.item()*1e9:.2f}ns  WNS={wns.item()*1e9:.6f}ns  "
                f"DynPow={switching_power.item()*1e3:.4f}mW  "
                f"LeakPow={leakage_power.item()*1e3:.4f}mW  "
                f"Overflow={ovfl.item():.4f}  Penalty={quantified_penalty.item():.4f}  "
                f"Loss={loss.item():.4f}  lr={size_sched.get_last_lr()[0]:.4f}  "
                f"t={time.time()-exe_time:.1f}s"
            )

            loss.backward()

            grad = self.sta_engine.cell_xy.grad
            mask = getattr(self.sta_engine.graph_data, "gate_movable_mask", None)
            if grad is not None and mask is not None and mask.numel() == grad.shape[0]:
                grad.mul_(mask.to(device=grad.device, dtype=grad.dtype).unsqueeze(1))

            for opt in (size_opt, pos_opt, buf_opt):
                opt.step()
                opt.zero_grad()
            for sched in (size_sched, xy_sched, buf_sched):
                sched.step()

            penalty_scale *= 1.0 / decay_lr
            penalty_tensor.mul_(penalty_inc)

        print(f"[INFO] run_gradient_opt done in {time.time() - exe_time:.2f}s")

        self._commit_to_db()
        self.design.evalTclString("remove_parasitics -all")
        self.design.evalTclString("estimate_parasitics -placement")
        self.design.evalTclString("report_tns")
        self.design.evalTclString("report_wns")
        self.design.evalTclString("report_power")

        return size_sched.get_last_lr()[0]

    # ------------------------------------------------------------------
    # OpenROAD database writeback
    # ------------------------------------------------------------------

    def _commit_to_db(self):
        self._sync_cell_types()
        self._sync_cell_positions()
        self._sync_buffering()

    def _sync_cell_types(self):
        t = time.time()
        U   = self.sta_engine.U_flat.detach().cpu()
        off = self.sta_engine.graph_data.u_gate_off.detach().cpu()
        swapped = 0
        for gi, g in enumerate(self.cell_db.signal_gates):
            s, e = int(off[gi]), int(off[gi + 1])
            new_master = g.eqvMaster[int(torch.argmax(U[s:e]).item())]
            if g.db_master.getName() != new_master.getName():
                g.swapCell(new_master)
                swapped += 1
        print(f"[INFO] cell type sync: {swapped} swapped in {time.time()-t:.2f}s")

    def _sync_cell_positions(self):
        t = time.time()
        if self.sta_engine is None or self.cell_db is None:
            return
        cell_xy = self.sta_engine.cell_xy.detach().cpu()
        if cell_xy.ndim != 2 or cell_xy.shape[1] < 2:
            print(f"[WARN] unexpected cell_xy shape {tuple(cell_xy.shape)}")
            return
        moved = skipped = 0
        for g in self.cell_db.signal_gates:
            idx = getattr(g, "idx", -1)
            if idx < 0 or idx >= cell_xy.shape[0]:
                continue
            inst = g.db_Inst
            if inst is None or (inst.getMaster() and inst.getMaster().isBlock()):
                skipped += 1
                continue
            status = inst.getPlacementStatus()
            if status in ("LOCKED", "FIRM", "COVER"):
                skipped += 1
                continue
            x = int(round(float(cell_xy[idx, 0].item())))
            y = int(round(float(cell_xy[idx, 1].item())))
            inst.setLocation(x, y)
            if status in ("NONE", "UNPLACED", "SUGGESTED"):
                inst.setPlacementStatus("PLACED")
            moved += 1
        print(f"[INFO] cell position sync: {moved} moved, {skipped} skipped in {time.time()-t:.2f}s")

    def _sync_buffering(self):
        t = time.time()
        should_buf = self.sta_engine.buffering_tensor.detach().cpu() > 0
        inserted = should_buf.sum().item()
        for net in self.cell_db.signal_nets:
            if net.driver_pin is None:
                continue
            buffering_update(net, should_buf, self.sta_engine.graph_data.chosen_buffer)
        print(f"[INFO] buffering sync: {inserted} inserted in {time.time()-t:.2f}s")

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_sta(self):
        self.sta_engine.discretize()
        with torch.no_grad():
            tns, wns, switching_power, leakage_power, quantified_penalty, density = self.sta_engine()
            print(
                f"[INFO] Eval STA: TNS={tns.item():.4e}  WNS={wns.item():.4e}  "
                f"Power={switching_power.item():.4e}  Leakage={leakage_power.item():.4e}  "
                f"Penalty={quantified_penalty.item():.4f}  "
                f"DensitySum={density.sum().item():.4f}"
            )
        return tns, wns, switching_power, quantified_penalty

    def report_density(self, density_map):
        mx  = density_map.max()
        avg = density_map.mean()
        std = density_map.std()
        med = density_map.median()
        fhm = density_map[density_map >= med].mean()
        print(f"[INFO] Density: max={mx.item():.4f}  avg={avg.item():.4f}  "
              f"std={std.item():.4f}  median={med.item():.4f}  upper_half_mean={fhm.item():.4f}")
