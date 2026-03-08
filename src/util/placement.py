from src.util.buffer_ops import *
from openroad import Tech, Design, Timing

def detailed_placement(
    design,
    output_dir,
):
    dpl_rc = _run_detailed_placement(design)
    failed_names = _collect_dpl_failures(design) if str(dpl_rc).strip() != "0" else []
    removed_buffers = 0
    timing = Timing(design)
    if failed_names:
        block = design.getBlock()
        for name in failed_names:
            inst = block.findInst(name)
            if inst is None:
                continue
            if isBuffer(inst.getMaster()):
                remove_buffer(inst)
                removed_buffers += 1
                print(f"[INFO] removed buffer instance {name} for re-placement.")
            else:
                print(f"[WARN] detailed_placement failed to place non-buffer instance {name}.")
                eqv = timing.equivCells(inst.getMaster())
                for alt_master in eqv:
                    print(f"[WARN]   equivalent cell: {alt_master.getName()}")

    if removed_buffers > 0:
        dpl_rc = _run_detailed_placement(design)
        failed_names = _collect_dpl_failures(design) if str(dpl_rc).strip() != "0" else []

    try:
        rc_int = int(str(dpl_rc).strip())
    except ValueError:
        rc_int = 1
    return rc_int, failed_names


def _run_detailed_placement(design):
    tcl = "set dpl_rc [catch { detailed_placement } dpl_err]\n" \
          "if { $dpl_rc != 0 } { puts \"[WARN] detailed_placement failed: $dpl_err\" }\n" \
          "return $dpl_rc"
    return design.evalTclString(tcl)


def _collect_dpl_failures(design):
    block = design.getBlock() if design is not None else None
    if block is None:
        return []

    tool_category = block.findMarkerCategory("DPL")
    if tool_category is None:
        return []

    fail_category = tool_category.findMarkerCategory("Placement_failures")
    if fail_category is None:
        return []

    failed_names = []
    for marker in fail_category.getMarkers():
        name = marker.getName()
        if name.startswith("Layer: "):
            parts = name.split(" ", 2)
            name = parts[2] if len(parts) > 2 else ""
        for item in name.split(","):
            item = item.strip()
            if item:
                failed_names.append(item)

    unique = []
    seen = set()
    for name in failed_names:
        if name not in seen:
            seen.add(name)
            unique.append(name)

    if unique:
        print(f"[WARN] detailed_placement failed to place {len(unique)} instances.")
        print(f"[WARN] failed instances: {', '.join(unique)}")

    return unique
