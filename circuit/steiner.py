"""
circuit.steiner — SteinerNetworkBuilder

Builds the complete directed timing propagation graph by:

1. Running a Steiner minimum tree builder (OpenROAD FLUTE/PDT) on each net.
2. Levelizing each net tree from driver to sinks.
3. Stitching net trees together through cell timing arcs to form a single
   circuit-level DAG (``stpList`` / ``levelizedNetwork``).
4. Handling flip-flop clock-to-Q arcs as explicit start-point edges.

The resulting ``stpList`` feeds the ``STAGraphBuilder`` in ``timing.graph``.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from placeopt.circuit.components import Gate, Net, Pin, SteinerPoint, match_pins_to_steiner_points
from placeopt.io.design import get_rss_mb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_driver_pin(pin_or_iterm) -> bool:
    iterm = getattr(pin_or_iterm, "db_iterm", pin_or_iterm)
    net = iterm.getNet()
    if net is None:
        return False
    if net.getSigType() in ("CLOCK", "POWER", "GROUND"):
        return False
    return bool(iterm.isOutputSignal())


def _is_signal(net_type: str) -> bool:
    return net_type not in ("CLOCK", "POWER", "GROUND")


def _replace_in_list(lst: list, old, new) -> None:
    for k in range(len(lst)):
        if lst[k] is old:
            lst[k] = new


# ---------------------------------------------------------------------------
# Topological levelize of the full circuit DAG
# ---------------------------------------------------------------------------

def levelize_circuit(design, stp_list: List[SteinerPoint]):
    """
    Topological sort (Kahn's BFS) over all Steiner points in ``stp_list``.

    The result is a flat ``levelized_network`` where alternating layers
    represent net-level propagation (wire delay) and gate-level propagation
    (cell arc delay).

    Returns
    -------
    start_points, end_points, gate_input_pts, gate_output_pts, levelized_network
    """
    in_degree: Dict[int, int] = defaultdict(int)
    for u in stp_list:
        for v in u.nexts:
            in_degree[v.idx] += 1

    q = deque([stp for stp in stp_list if in_degree[stp.idx] == 0])
    raw_levels: List[List[SteinerPoint]] = []

    while q:
        level_size = len(q)
        layer = []
        for _ in range(level_size):
            u = q.popleft()
            layer.append(u)
            for v in u.nexts:
                in_degree[v.idx] -= 1
                if in_degree[v.idx] == 0:
                    q.append(v)
        if layer:
            raw_levels.append(layer)

    # Interleave net and gate layers.
    levelized: List[List[SteinerPoint]] = []
    for layer in raw_levels:
        wire_layer = [s for s in layer if not (s.Pin is not None and s.Pin.io_type == "OUTPUT")]
        gate_layer = [s for s in layer if s.Pin is not None and s.Pin.io_type == "OUTPUT"]
        levelized.append(wire_layer)
        levelized.append(gate_layer)

    # Classify points.
    start_pts: List[SteinerPoint] = []
    end_pts: List[SteinerPoint] = []
    gate_in_pts: List[SteinerPoint] = []
    gate_out_pts: List[SteinerPoint] = []

    for stp in stp_list:
        if stp.Pin is None:
            continue
        if _is_circuit_input(stp.Pin.db_iterm):
            start_pts.append(stp)
        if _is_circuit_output(stp.Pin.db_iterm):
            end_pts.append(stp)
        if design.isSequential(stp.Pin.gate.db_inst.getMaster()):
            n = stp.Pin.net
            if n is None or n.db_net.getNet() is None:
                pass
            sig = n.db_net.getSigType()
            if sig in ("POWER", "GROUND"):
                continue
            mname = stp.Pin.db_iterm.getMTerm().getName().upper()
            if sig == "CLOCK" or "CLK" in mname:
                start_pts.append(stp)
            elif stp.Pin.io_type == "INPUT":
                end_pts.append(stp)
            continue
        if stp.Pin.io_type == "INPUT":
            gate_in_pts.append(stp)
        if stp.Pin.io_type == "OUTPUT":
            gate_out_pts.append(stp)

    for i, layer in enumerate(levelized):
        for stp in layer:
            stp.level = i

    return start_pts, end_pts, gate_in_pts, gate_out_pts, levelized


def _is_circuit_input(iterm) -> bool:
    net = iterm.getNet()
    if net is None:
        return False
    if net.getSigType() in ("CLOCK", "POWER", "GROUND"):
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


# ---------------------------------------------------------------------------
# SteinerNetworkBuilder
# ---------------------------------------------------------------------------

class SteinerNetworkBuilder:
    """
    Constructs the joint Steiner-tree timing graph for all signal nets.

    Usage
    -----
    ::

        builder = SteinerNetworkBuilder(design, timing)
        builder.build(cell_lib.signal_nets, cell_lib.signal_gates,
                      cell_lib.TensorTableModels)
        # builder.stp_list  ← flat list of all SteinerPoints
        # builder.levelized ← leveled network for forward propagation

    Attributes
    ----------
    stp_list      : flat list of all SteinerPoints (globally indexed)
    levelized     : 2D list: levelized_network[level][stp]
    start_points  : timing start points (PI/CLK pins)
    end_points    : timing end points (PO/D/SI/SE pins)
    gate_input_pts  : non-sequential gate input Steiner points
    gate_output_pts : non-sequential gate output Steiner points
    cell_arc_segs : list of (parent_stp, child_stp) across gate boundaries
    net_arc_segs  : list of (parent_stp, child_stp) within the same net
    """

    def __init__(self, design, timing) -> None:
        self.design = design
        self.timing = timing
        self._tree_builder = design.getSteinerTreeBuilder()
        self._reset()

    def _reset(self) -> None:
        self.stp_list: List[SteinerPoint] = []
        self.cell_arc_segs: List[Tuple[SteinerPoint, SteinerPoint]] = []
        self.net_arc_segs: List[Tuple[SteinerPoint, SteinerPoint]] = []
        self.levelized: List[List[SteinerPoint]] = []
        self.start_points: List[SteinerPoint] = []
        self.end_points: List[SteinerPoint] = []
        self.gate_input_pts: List[SteinerPoint] = []
        self.gate_output_pts: List[SteinerPoint] = []

    # Legacy attribute names for downstream compatibility
    @property
    def stpList(self):  # noqa: N802
        return self.stp_list

    @property
    def levelizedNetwork(self):  # noqa: N802
        return self.levelized

    # ------------------------------------------------------------------

    def build(
        self,
        signal_nets: List[Net],
        signal_gates: List[Gate],
        table_models: Dict,
    ) -> None:
        """
        Run the full Steiner-tree construction and circuit levelize pipeline.

        Steps
        -----
        1. Build per-net Steiner trees.
        2. Prepare FF clock-to-Q start edges.
        3. Stitch net trees via cell arcs.
        4. Assign global flat indices.
        5. Topological levelize the whole circuit.
        """
        t0 = time.perf_counter()
        rss = get_rss_mb()
        if rss:
            print(f"[MEM] SteinerNetworkBuilder.build start: {rss:.1f} MB")

        self._reset()

        # Step 1: per-net trees.
        for net in signal_nets:
            self._build_net_tree(net)
        print(f"[TIME] per-net Steiner trees: {time.perf_counter()-t0:.3f}s")

        # Step 2: FF CLK → Q edges (start points).
        t1 = time.perf_counter()
        clk_stps = self._build_ff_clk_edges(signal_gates)
        print(f"[TIME] FF clk edges: {time.perf_counter()-t1:.3f}s")

        # Step 3: stitch via cell arcs.
        t1 = time.perf_counter()
        for net in signal_nets:
            self._stitch_cell_arcs(net, table_models)
        print(f"[TIME] stitch cell arcs: {time.perf_counter()-t1:.3f}s")

        # Step 4: flatten & index.
        t1 = time.perf_counter()
        self.stp_list = clk_stps
        for net in signal_nets:
            self.stp_list.extend(net.steiner_pts)

        idx = 0
        for stp in self.stp_list:
            stp.idx = idx
            idx += 1

        for stp in self.stp_list:
            for prev in stp.prevs:
                seg = (prev, stp)
                if prev.net is stp.net and prev.net is not None:
                    self.net_arc_segs.append(seg)
                else:
                    self.cell_arc_segs.append(seg)
        print(f"[TIME] flatten & index: {time.perf_counter()-t1:.3f}s")

        # Step 5: levelize.
        t1 = time.perf_counter()
        (self.start_points,
         self.end_points,
         self.gate_input_pts,
         self.gate_output_pts,
         self.levelized) = levelize_circuit(self.design, self.stp_list)
        print(f"[TIME] levelize circuit: {time.perf_counter()-t1:.3f}s")

        rss2 = get_rss_mb()
        if rss2:
            print(f"[MEM] SteinerNetworkBuilder.build end: {rss2:.1f} MB")
        print(f"[INFO] Steiner network built in {time.perf_counter()-t0:.2f}s"
              f"  ({len(self.stp_list)} nodes, {len(signal_nets)} nets)")

    # ------------------------------------------------------------------
    # Per-net tree construction
    # ------------------------------------------------------------------

    def _build_net_tree(self, net: Net) -> None:
        """Build the Steiner tree for a single net and attach it to ``net``."""
        for p in net.pins:
            p.steiner_pt = None
        net.steiner_pts = []

        db_net = net.db_net
        sig = db_net.getSigType()
        if sig in ("CLOCK", "POWER", "GROUND"):
            return

        iterms = list(db_net.getITerms())
        if len(iterms) <= 1:
            return

        # Collect pin coordinates and find the driver index.
        xs, ys, driver_idx = [], [], -1
        for k, it in enumerate(iterms):
            if _is_driver_pin(it):
                driver_idx = k
            avg = it.getAvgXY()
            xs.append(avg[1])
            ys.append(avg[2])

        if driver_idx == -1:
            # No driver → all are sinks (e.g., primary-input net with no driver)
            for i, (x, y) in enumerate(zip(xs, ys)):
                stp = SteinerPoint(x, y)
                stp.net = net
                stp.idx = i
                if i < len(net.pins):
                    stp.Pin = net.pins[i]
                    net.pins[i].steiner_pt = stp
                net.steiner_pts.append(stp)
            return

        tree = self._tree_builder.makeSteinerTree(xs, ys, driver_idx, 0.3)
        n_branches = tree.branchCount()
        if n_branches <= 0:
            return

        pts = [SteinerPoint(br.x, br.y) for br in tree.branch]
        net.steiner_pts = match_pins_to_steiner_points(net.pins, pts)
        for stp in net.steiner_pts:
            stp.net = net

        self._levelize_net_tree(net, tree, driver_idx)

    def _levelize_net_tree(self, net: Net, tree, driver_idx: int) -> None:
        """BFS levelize the Steiner tree from the driver pin."""
        n = tree.branchCount()

        # Build adjacency list from branch connectivity.
        adj: List[List[int]] = [[] for _ in range(n)]
        for i, br in enumerate(tree.branch):
            j = br.n
            if i == j:
                continue
            if j not in adj[i]:
                adj[i].append(j)
            if i not in adj[j]:
                adj[j].append(i)

        # Find the Steiner node that is the driver pin.
        driver_stp = next((s for s in net.steiner_pts if s.Pin is not None
                           and _is_driver_pin(s.Pin.db_iterm)), None)
        if driver_stp is None:
            return
        root = driver_stp.idx  # local index within this net

        levels = [-1] * n
        parent = [-1] * n
        q: deque[int] = deque([root])
        levels[root] = 0

        while q:
            u = q.popleft()
            for v in adj[u]:
                if levels[v] == -1:
                    levels[v] = levels[u] + 1
                    parent[v] = u
                    q.append(v)

        # Assign parent ↔ child relationships.
        for stp in net.steiner_pts:
            stp.prevs.clear()
            stp.nexts.clear()

        for i in range(min(n, len(net.steiner_pts))):
            if levels[i] == -1:
                continue
            stp = net.steiner_pts[i]
            p = parent[i]
            if p != -1 and p < len(net.steiner_pts):
                stp.prevs.append(net.steiner_pts[p])
                net.steiner_pts[p].nexts.append(stp)

        # Split sink pins sitting on branch nodes (not leaves) and the driver
        # pin if it has multiple direct children — this ensures every pin
        # behaves as an endpoint in the tree.
        self._split_branch_pins(net, driver_stp)

    def _split_branch_pins(self, net: Net, driver_stp: SteinerPoint) -> None:
        """
        Insert zero-length branch nodes to ensure every pin is a leaf or
        a node with exactly one net-child.
        """
        orig = list(net.steiner_pts)

        # Sink pins that have net-children need a proxy branch node.
        for stp in orig:
            if stp.Pin is None or stp.Pin is driver_stp.Pin:
                continue
            if not stp.nexts or not stp.prevs:
                continue
            branch = SteinerPoint(stp.x, stp.y)
            branch.net = net
            parent_stp = stp.prevs[0]
            old_children = list(stp.nexts)

            _replace_in_list(parent_stp.nexts, stp, branch)
            branch.prevs = [parent_stp]
            branch.nexts = [stp] + old_children
            stp.prevs = [branch]
            stp.nexts = []
            for c in old_children:
                _replace_in_list(c.prevs, stp, branch)
            net.steiner_pts.append(branch)

        # Driver with multiple children: insert a single fanout branch.
        if len(driver_stp.nexts) > 1:
            branch = SteinerPoint(driver_stp.x, driver_stp.y)
            branch.net = net
            old_children = list(driver_stp.nexts)
            driver_stp.nexts = [branch]
            branch.prevs = [driver_stp]
            branch.nexts = old_children
            for c in old_children:
                _replace_in_list(c.prevs, driver_stp, branch)
            net.steiner_pts.append(branch)

    # ------------------------------------------------------------------
    # Flip-flop clock edges
    # ------------------------------------------------------------------

    def _build_ff_clk_edges(self, signal_gates: List[Gate]) -> List[SteinerPoint]:
        """
        Create CLK→Q edges for every sequential cell and return the list of
        CLK Steiner points that become timing start points.
        """
        clk_stps: List[SteinerPoint] = []
        for g in signal_gates:
            if not self.design.isSequential(g.db_inst.getMaster()):
                continue

            q_pins = []
            clk_pin: Optional[Pin] = None
            for pin in g.pins.values():
                if pin.db_iterm.getNet() is None:
                    continue
                mname = pin.db_iterm.getMTerm().getName().upper()
                if "Q" in mname:
                    q_pins.append(pin)
                n_sig = pin.db_iterm.getNet().getSigType()
                if n_sig != "CLOCK" and "CLK" not in mname:
                    continue
                clk_pin = pin
                pt = SteinerPoint(pin.db_iterm.getAvgXY()[1], pin.db_iterm.getAvgXY()[2])
                pt.Pin = pin
                pt.net = pin.net
                pin.steiner_pt = pt
                if pin.net is not None:
                    pin.net.steiner_pts.append(pt)
                clk_stps.append(pt)

            if clk_pin is None:
                continue
            for qpin in q_pins:
                if qpin.steiner_pt is not None and clk_pin.steiner_pt is not None:
                    qpin.steiner_pt.prevs.append(clk_pin.steiner_pt)
                    clk_pin.steiner_pt.nexts.append(qpin.steiner_pt)

        return clk_stps

    # ------------------------------------------------------------------
    # Stitch net trees via cell arcs
    # ------------------------------------------------------------------

    def _stitch_cell_arcs(self, net: Net, table_models: Dict) -> None:
        """
        Connect input Steiner points on this net to the corresponding output
        Steiner points of the same gate, using Liberty arc tables to find
        valid (in_pin → out_pin) pairs.
        """
        if not net.steiner_pts:
            return

        for stp in [s for s in net.steiner_pts if s.Pin is not None]:
            pin = stp.Pin
            if self.design.isSequential(pin.gate.db_inst.getMaster()):
                if pin.db_iterm.getNet() is not None and pin.db_iterm.getNet().getSigType() == "CLOCK":
                    for out_pin in pin.gate.output_pins:
                        out_stp = out_pin.steiner_pt
                        if out_stp is not None:
                            if stp not in out_stp.prevs:
                                out_stp.prevs.append(stp)
                            if out_stp not in stp.nexts:
                                stp.nexts.append(out_stp)
                continue

            master_name = pin.gate.db_inst.getMaster().getName()
            for (in_pin_name, out_pin_name, _rf), _table in table_models.get(master_name, {}).items():
                if pin.name == in_pin_name:
                    out_pin = pin.gate.pins.get(out_pin_name)
                    if out_pin and out_pin.steiner_pt:
                        if out_pin.steiner_pt not in stp.nexts:
                            stp.nexts.append(out_pin.steiner_pt)
                        if stp not in out_pin.steiner_pt.prevs:
                            out_pin.steiner_pt.prevs.append(stp)
                elif pin.name == out_pin_name:
                    in_pin = pin.gate.pins.get(in_pin_name)
                    if in_pin and in_pin.steiner_pt:
                        if in_pin.steiner_pt not in stp.prevs:
                            stp.prevs.append(in_pin.steiner_pt)
                        if stp not in in_pin.steiner_pt.nexts:
                            in_pin.steiner_pt.nexts.append(stp)
