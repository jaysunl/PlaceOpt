from collections import deque, defaultdict
import time

import odb
import openroad as ord
from openroad import Tech, Design, Timing

from src.util.helpers import isDriverPin, isSignalNet, is_circuit_input, is_circuit_output
from src.db.netlist import SteinerNode, WireNet, map_pins_to_tree


class TreeBuilder:
    """Builds and levelizes Steiner tree networks for timing propagation."""

    def __init__(self, design, timing):
        self.design = design
        self.timing = timing
        self.treeBuilder = design.getSteinerTreeBuilder()
        self._reset_state()

    def _reset_state(self):
        self.stpList = []
        self.cellArcSegments = []
        self.netArcSegments  = []
        self.levelizedNetwork = []
        self.startPoints      = []
        self.endPoints        = []
        self.gateInputPoints  = []
        self.gateOutputPoints = []

    # ------------------------------------------------------------------
    # Steiner tree construction
    # ------------------------------------------------------------------

    def build_tree_node(self, net):
        for pin in net.Pins:
            pin.steinerPoint = None
        net.steinerPoints = []

        db_net   = net.db_net
        drvr_idx = -1
        xs, ys   = deque(), deque()

        for i, it in enumerate(db_net.getITerms()):
            if isDriverPin(it):
                drvr_idx = i
            bb = it.getBBox()
            if not bb:
                continue
            xs.append(it.getAvgXY()[1])
            ys.append(it.getAvgXY()[2])

        if drvr_idx == -1 or len(xs) == 1:
            for i in range(len(xs)):
                node = SteinerNode(xs[i], ys[i])
                node.Net = net
                node.idx = i
                if net.Pins and i < len(net.Pins):
                    node.Pin = net.Pins[i]
                    net.Pins[i].steinerPoint = node
                net.steinerPoints.append(node)
            return

        if len(xs) <= 1:
            return

        sig = db_net.getSigType()
        if sig in ("CLOCK", "POWER", "GROUND"):
            return

        tree = self.treeBuilder.makeSteinerTree(xs, ys, drvr_idx, 0.3)
        branch_cnt = tree.branchCount()

        tree_nodes = []
        for br in tree.branch:
            node = SteinerNode(br.x, br.y)
            node.Net = net
            tree_nodes.append(node)

        net.steinerPoints = map_pins_to_tree(net.Pins, list(tree_nodes))
        self.levelize_tree(net, tree)

    def build_tree_nodes(self, net_vec):
        for net in net_vec:
            self.build_tree_node(net)

    def levelize_tree(self, net, tree):
        branch_cnt = tree.branchCount()
        if branch_cnt <= 0:
            return

        adj = [[] for _ in range(branch_cnt)]
        for i, br in enumerate(tree.branch):
            j = br.n
            if j == i:
                continue
            if j not in adj[i]:
                adj[i].append(j)
            if i not in adj[j]:
                adj[j].append(i)

        driver_pin = None
        for pin in net.Pins:
            if pin.db_ITerm.isOutputSignal():
                driver_pin = pin
                break
        if driver_pin is None or not net.steinerPoints:
            return

        root_idx = next(
            (stp.idx for stp in net.steinerPoints if stp.Pin is driver_pin),
            None,
        )
        if root_idx is None:
            return

        levels = [-1] * branch_cnt
        parent = [-1] * branch_cnt
        q = deque([root_idx])
        levels[root_idx] = 0

        while q:
            u = q.popleft()
            for v in adj[u]:
                if levels[v] == -1:
                    levels[v] = levels[u] + 1
                    parent[v] = u
                    q.append(v)

        for stp in net.steinerPoints:
            stp.prevs.clear()
            stp.nexts.clear()

        max_level = max((l for l in levels if l >= 0), default=-1)
        net.levels = [[] for _ in range(max_level + 1)] if max_level >= 0 else []

        for i in range(min(branch_cnt, len(net.steinerPoints))):
            if levels[i] == -1:
                continue
            stp = net.steinerPoints[i]
            stp.level = levels[i]
            net.levels[stp.level].append(i)
            p = parent[i]
            if p != -1 and p < len(net.steinerPoints):
                stp.prevs.append(net.steinerPoints[p])
                net.steinerPoints[p].nexts.append(stp)

        def _replace(lst, old, new):
            for k in range(len(lst)):
                if lst[k] is old:
                    lst[k] = new

        driver_stp = next((s for s in net.steinerPoints if s.Pin is driver_pin), None)

        for stp in list(net.steinerPoints):
            if stp.Pin is None or stp.Pin is driver_pin or not stp.nexts or not stp.prevs:
                continue
            branch = SteinerNode(stp.x, stp.y)
            branch.Net = net
            parent_stp = stp.prevs[0]
            old_children = list(stp.nexts)

            _replace(parent_stp.nexts, stp, branch)
            branch.prevs = [parent_stp]
            branch.nexts = [stp] + old_children

            stp.prevs = [branch]
            stp.nexts = []
            for c in old_children:
                _replace(c.prevs, stp, branch)

            net.steinerPoints.append(branch)

        if driver_stp is not None and len(driver_stp.nexts) > 1:
            branch = SteinerNode(driver_stp.x, driver_stp.y)
            branch.Net = net
            old_children = list(driver_stp.nexts)
            driver_stp.nexts = [branch]
            branch.prevs = [driver_stp]
            branch.nexts = old_children
            for c in old_children:
                _replace(c.prevs, driver_stp, branch)
            net.steinerPoints.append(branch)

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def build_network(self, net_vec, gate_vec, table_list=None):
        t_all = time.perf_counter()

        self._reset_state()
        self.build_tree_nodes(net_vec)
        clk_stps = self.prepare_clock_points(net_vec, gate_vec)
        self.rectilinearize(net_vec)
        for net in net_vec:
            self.merge_trees(net, table_list)

        self.stpList = clk_stps[:]
        for net in net_vec:
            if not getattr(net, "steinerPoints", None):
                continue
            self.stpList.extend(net.steinerPoints)

        self.cellArcSegments = []
        self.netArcSegments  = []
        for new_idx, stp in enumerate(self.stpList):
            stp.idx = new_idx
            for prev in stp.prevs:
                if prev.Net.db_net.getName() == stp.Net.db_net.getName():
                    self.netArcSegments.append((prev, stp))
                else:
                    self.cellArcSegments.append((prev, stp))

        (
            self.startPoints,
            self.endPoints,
            self.gateInputPoints,
            self.gateOutputPoints,
            self.levelizedNetwork,
        ) = _levelize_circuit(self.design, self.stpList)

        print(f"[INFO] Steiner network: {len(net_vec)} nets, {len(self.stpList)} nodes "
              f"in {time.perf_counter() - t_all:.2f}s")

    def rectilinearize(self, nets):
        pass  # placeholder for future rectilinearization

    def prepare_clock_points(self, net_vec, gate_vec):
        clk_stps = []
        for g in gate_vec:
            if not self.design.isSequential(g.db_Inst.getMaster()):
                continue
            q_pins  = []
            clk_pin = None
            for pin in g.Pins.values():
                if pin.db_ITerm.getNet() is None:
                    continue
                if "Q" in pin.db_ITerm.getMTerm().getName().upper():
                    q_pins.append(pin)
                sig = pin.db_ITerm.getNet().getSigType()
                clk_in_name = "CLK" in pin.db_ITerm.getMTerm().getName().upper()
                if sig != "CLOCK" and not clk_in_name:
                    continue
                clk_pin = pin
                node = SteinerNode(pin.db_ITerm.getAvgXY()[1], pin.db_ITerm.getAvgXY()[2])
                node.Pin = pin
                node.Net = pin.Net
                node.idx = -1
                pin.steinerPoint = node
                pin.Net.steinerPoints.append(node)
                clk_stps.append(node)
            if clk_pin is None:
                continue
            for qpin in q_pins:
                if qpin.steinerPoint is not None:
                    qpin.steinerPoint.prevs.append(clk_pin.steinerPoint)
                    clk_pin.steinerPoint.nexts.append(qpin.steinerPoint)
        return clk_stps

    def merge_trees(self, net: WireNet, table_list=None):
        if not getattr(net, "steinerPoints", None):
            return

        for stp in [s for s in net.steinerPoints if s.Pin is not None]:
            pin = stp.Pin
            if self.design.isSequential(pin.Gate.db_Inst.getMaster()):
                if pin.db_ITerm.getNet() and pin.db_ITerm.getNet().getSigType() == "CLOCK":
                    for out_pin in pin.Gate.outputPins:
                        out_stp = out_pin.steinerPoint
                        if out_stp and stp not in out_stp.prevs:
                            out_stp.prevs.append(stp)
                            stp.nexts.append(out_stp)
                continue

            master = pin.Gate.db_Inst.getMaster()
            tables = table_list[master.getName()] if table_list else {}
            for table in tables.values():
                if pin.name == table.in_pin_name:
                    next_pin = pin.Gate.Pins.get(table.out_pin_name)
                    if next_pin and next_pin.steinerPoint:
                        ns = next_pin.steinerPoint
                        if ns not in stp.nexts:
                            stp.nexts.append(ns)
                        if stp not in ns.prevs:
                            ns.prevs.append(stp)
                elif pin.name == table.out_pin_name:
                    prev_pin = pin.Gate.Pins.get(table.in_pin_name)
                    if prev_pin and prev_pin.steinerPoint:
                        ps = prev_pin.steinerPoint
                        if ps not in stp.prevs:
                            stp.prevs.append(ps)
                        if stp not in ps.nexts:
                            ps.nexts.append(stp)


def _levelize_circuit(design, stp_list):
    """Topological sort of Steiner points; classify start/end/gate points."""
    in_degree = defaultdict(int)
    for u in stp_list:
        for v in u.nexts:
            in_degree[v.idx] += 1

    q = deque(stp for stp in stp_list if in_degree[stp.idx] == 0)
    raw_network = []
    while q:
        level = []
        for _ in range(len(q)):
            u = q.popleft()
            level.append(u)
            for v in u.nexts:
                in_degree[v.idx] -= 1
                if in_degree[v.idx] == 0:
                    q.append(v)
        if level:
            raw_network.append(level)

    leveled = []
    for level in raw_network:
        net_lvl  = [s for s in level if s.Pin is None or s.Pin.IO != "OUTPUT"]
        gate_lvl = [s for s in level if s.Pin is not None and s.Pin.IO == "OUTPUT"]
        leveled.append(net_lvl)
        leveled.append(gate_lvl)

    for i, level in enumerate(leveled):
        for stp in level:
            stp.level = i

    start_pts = []
    end_pts   = []
    gate_in   = []
    gate_out  = []

    for stp in stp_list:
        if stp.Pin is None:
            continue
        if is_circuit_input(stp.Pin.db_ITerm):
            start_pts.append(stp)
        if is_circuit_output(stp.Pin.db_ITerm):
            end_pts.append(stp)
        if design.isSequential(stp.Pin.Gate.db_Inst.getMaster()):
            net = stp.Pin.db_ITerm.getNet()
            if net is None:
                continue
            sig = net.getSigType()
            if not isSignalNet(sig) and sig != "CLOCK":
                continue
            if stp.Pin.db_ITerm.getIoType() == "OUTPUT":
                continue
            pin_name = stp.Pin.db_ITerm.getMTerm().getName().upper()
            if sig == "CLOCK" or "CLK" in pin_name:
                start_pts.append(stp)
            else:
                end_pts.append(stp)
            continue
        if stp.Pin.IO == "INPUT":
            gate_in.append(stp)
        if stp.Pin.IO == "OUTPUT":
            gate_out.append(stp)

    return start_pts, end_pts, gate_in, gate_out, leveled
