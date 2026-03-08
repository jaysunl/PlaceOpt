from dataclasses import dataclass
from collections import deque
import time

from src.db.netlist import LogicCell, SignalPin, WireNet, SteinerNode
import openroad as ord
import odb


def isInverter(master):
    if master is None:
        print("[WARN] isInverter: master is None")
        return False
    return "INV" in master.getName().upper()


def isBuffer(master):
    if master is None:
        print("[WARN] isBuffer: master is None")
        return False
    name = master.getName().upper()
    return "BUF" in name or any(x in name for x in ("HB4XP67", "HB3XP67", "HB2XP67", "HB1XP67"))


def iterm_key(iterm):
    return f"{iterm.getInst().getName()}/{iterm.getMTerm().getName()}"


def topo_sort_indices(stps):
    """Kahn topological sort based on `stp.nexts` pointers; returns list indices."""
    n = len(stps)
    pos = {stp: i for i, stp in enumerate(stps)}
    indeg = [0] * n
    for u in range(n):
        for child in stps[u].nexts:
            v = pos.get(child)
            if v is not None:
                indeg[v] += 1

    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for child in stps[u].nexts:
            v = pos.get(child)
            if v is None:
                continue
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != n:
        seen = set(order)
        order.extend([i for i in range(n) if i not in seen])
    return order


def build_rhs_rep_set(net_obj, stps, rep, idx, driver_key: str):
    """Compute set of iterm_keys for the subtree rooted at idx on the remaining net."""
    stp = stps[idx]
    rhs: set[str] = set()
    if stp.Pin is not None:
        key = iterm_key(stp.Pin.db_ITerm)
        if key != driver_key:
            rhs.add(key)
    for c in stp.nexts:
        rhs |= rep.get(c.idx, set())
    return rhs


@dataclass
class BufferingData:
    lhs_net: odb.dbNet
    rhs_net: odb.dbNet
    buffer_in: odb.dbITerm
    buffer_out: odb.dbITerm


cell_idx = 0
net_idx  = 0


def _unique_name(block, base, kind):
    global cell_idx, net_idx
    while True:
        if kind == "net":
            name = f"{base}_{net_idx}"
            net_idx += 1
            if block.findNet(name) is None:
                return name
        else:
            name = f"{base}_{cell_idx}"
            cell_idx += 1
            if block.findInst(name) is None:
                return name


def _pick_mterm(master, io_type, preferred):
    candidates = [m for m in master.getMTerms() if m.getIoType() == io_type]
    if not candidates:
        return None
    for pref in preferred:
        for m in candidates:
            if m.getName().upper() == pref:
                return m
    return candidates[0]


def insert_buffer(lhs_pins, rhs_pins, buffer_master, original_net, x, y,
                  inst_name=None, new_net_name=None, pin_in=None, pin_out=None):
    if buffer_master is None or original_net is None:
        return None

    block = original_net.getBlock()
    if block is None:
        return None

    lhs_keys = {iterm_key(it) for it in lhs_pins if it}
    for iterm in rhs_pins:
        if iterm and iterm_key(iterm) in lhs_keys:
            print(f"[WARNING] skip - lhs/rhs overlap on iterm {iterm.getName()}")
            return None

    driver = None
    for iterm in lhs_pins:
        if iterm and iterm.isOutputSignal():
            if driver is not None:
                print(f"[WARNING] skip - multiple drivers on net {original_net.getName()}")
                return None
            driver = iterm
    if driver is None:
        print(f"[WARNING] skip - no driver on net {original_net.getName()}")
        return None

    if inst_name is None:
        inst_name = _unique_name(block, "buffering", "inst")
    elif block.findInst(inst_name) is not None:
        inst_name = _unique_name(block, inst_name, "inst")

    if new_net_name is None:
        new_net_name = _unique_name(block, "net", "net")
    elif block.findNet(new_net_name) is not None:
        new_net_name = _unique_name(block, new_net_name, "net")

    rhs_net = odb.dbNet_create(block, new_net_name)
    rhs_net.setSigType(original_net.getSigType())

    inst = odb.dbInst_create(block, buffer_master, inst_name)
    if inst is None:
        print(f"[WARNING] skip - buffer {inst_name} creation failed")
        return None

    if x is not None and y is not None:
        inst.setLocation(int(x), int(y))
        inst.setOrient("R0")
        inst.setPlacementStatus("PLACED")

    if pin_in is None or pin_out is None:
        in_term  = _pick_mterm(buffer_master, "INPUT",  ["A"])
        out_term = _pick_mterm(buffer_master, "OUTPUT", ["Y"])
        if pin_in is None and in_term is not None:
            pin_in = in_term.getName()
        if pin_out is None and out_term is not None:
            pin_out = out_term.getName()

    pin_in_iterm  = inst.findITerm(pin_in)  if pin_in  else None
    pin_out_iterm = inst.findITerm(pin_out) if pin_out else None
    if pin_in_iterm is None or pin_out_iterm is None:
        print(f"[WARNING] skip - buffer {inst_name} pin not found")
        return None

    orig_name = original_net.getName()
    for iterm in rhs_pins:
        if iterm is None:
            continue
        iterm_net = iterm.getNet()
        if iterm_net is None or iterm_net.getName() != orig_name:
            continue
        iterm.disconnect()
        iterm.connect(rhs_net)

    pin_in_iterm.connect(original_net)
    pin_out_iterm.connect(rhs_net)

    for iterm in lhs_pins:
        if iterm is None:
            continue
        iterm_net = iterm.getNet()
        if iterm_net is None or iterm_net.getName() != orig_name:
            iterm.disconnect()
            iterm.connect(original_net)

    return BufferingData(
        lhs_net=original_net,
        rhs_net=rhs_net,
        buffer_in=pin_in_iterm,
        buffer_out=pin_out_iterm,
    )


def buffering_update(net_obj, should_buffer, buf_master):
    """Insert buffers at Steiner points marked by `should_buffer`."""
    buffer_cnt = sum(1 for stp in net_obj.steinerPoints if should_buffer[stp.idx])
    if buffer_cnt == 0:
        return []

    stps = net_obj.steinerPoints
    topo = topo_sort_indices(stps)
    order = list(reversed(topo))

    current_db_net = net_obj.db_net
    rep: dict[int, set[str]] = {}

    driver_iterm = net_obj.driver_pin.db_ITerm
    driver_key   = iterm_key(driver_iterm)

    newAddedBUFs = []
    newdbNets    = []
    change_list  = []

    for _ in range(len(order) - 1):
        idx = order[_]
        stp = stps[idx]

        rhs_set = build_rhs_rep_set(net_obj, stps, rep, idx, driver_key)
        do_buf  = should_buffer[stp.idx]
        if not do_buf:
            rep[stp.idx] = rhs_set
            continue

        alive_iterms = list(current_db_net.getITerms())
        alive_by_key = {iterm_key(it): it for it in alive_iterms}
        alive_keys   = set(alive_by_key.keys())

        rhs_keys_live = rhs_set & alive_keys
        lhs_keys_live = alive_keys - rhs_keys_live

        if not rhs_keys_live or not lhs_keys_live:
            rep[stp.idx] = rhs_set
            continue

        if driver_key in rhs_keys_live:
            rhs_keys_live.remove(driver_key)
            lhs_keys_live.add(driver_key)
            if not rhs_keys_live:
                rep[stp.idx] = rhs_set
                continue

        driver_iterm_live = alive_by_key.get(driver_key, driver_iterm)
        lhs_list = [driver_iterm_live] + [alive_by_key[k] for k in sorted(lhs_keys_live) if k != driver_key]
        rhs_list = [alive_by_key[k] for k in sorted(rhs_keys_live)]

        x = (stp.x + stp.prevs[0].x) / 2
        y = (stp.y + stp.prevs[0].y) / 2
        in_term  = buf_master.findMTerm("A")
        out_term = buf_master.findMTerm("Y")

        for mpin in in_term.getMPins():
            bbox = mpin.getBBox()
            cinx  = (bbox.xMin() + bbox.xMax()) / 2.0
            ciny  = (bbox.yMin() + bbox.yMax()) / 2.0
        for mpin in out_term.getMPins():
            bbox = mpin.getBBox()
            coutx = (bbox.xMin() + bbox.xMax()) / 2.0
            couty = (bbox.yMin() + bbox.yMax()) / 2.0

        X_buf = int(x - coutx)
        Y_buf = int(y - couty)

        data = insert_buffer(lhs_list, rhs_list, buf_master, current_db_net, X_buf, Y_buf,
                             pin_in="A", pin_out="Y")

        if data is None:
            rep[stp.idx] = rhs_set
            continue

        new_db_inst = data.buffer_in.getInst()
        change_list.append((new_db_inst, buf_master, data.rhs_net))

        current_db_net = data.lhs_net
        newdbNets.append(data.rhs_net)
        rep[stp.idx] = {iterm_key(data.buffer_in)}

        newbuf = LogicCell(data.buffer_in.getInst(), buf_master)
        newAddedBUFs.append(newbuf)

    newdbNets.append(current_db_net)
    return change_list


def remove_buffer(inst):
    if inst is None:
        return

    pin_in  = inst.findITerm("A")
    pin_out = inst.findITerm("Y")
    if pin_in is None or pin_out is None:
        return

    new_db_net   = pin_out.getNet()
    front_db_net = pin_in.getNet()
    if new_db_net is None or front_db_net is None:
        return
    if new_db_net == front_db_net:
        odb.dbInst_destroy(inst)
        return

    sinks = [it for it in list(new_db_net.getITerms()) if it is not None]
    for iterm in sinks:
        if iterm == pin_out or iterm.getInst() == inst:
            continue
        iterm.disconnect()
        iterm.connect(front_db_net)
    odb.dbInst_destroy(inst)
    odb.dbNet_destroy(new_db_net)
