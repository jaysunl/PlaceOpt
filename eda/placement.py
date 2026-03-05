"""
eda.placement — Detailed placement and legalization.

Runs OpenROAD's detailed placer (DPL) on the current design.  If any
instances fail to legalize, buffer instances are removed and the placer
is retried once; non-buffer failures generate a warning.
"""

from __future__ import annotations

from typing import List, Tuple

from placeopt.eda.buffering import remove_buffer, is_buffer


def run_detailed_placement(design) -> Tuple[int, List[str]]:
    """
    Run detailed placement and return (return_code, [failed_instance_names]).

    On failure the function:
    1. Collects the list of failed instance names from OpenROAD DPL markers.
    2. Removes any failed *buffer* instances (they can be re-inserted later).
    3. Reruns DPL once more with the reduced set of instances.

    Parameters
    ----------
    design : openroad.Design

    Returns
    -------
    rc           : 0 = success, non-zero = failure.
    failed_names : list of instance names that could not be placed.
    """
    from openroad import Timing  # type: ignore

    rc = _run_dpl(design)
    failed = _collect_dpl_failures(design) if str(rc).strip() != "0" else []

    if failed:
        block   = design.getBlock()
        timing  = Timing(design)
        n_removed = 0
        for name in failed:
            inst = block.findInst(name)
            if inst is None:
                continue
            if is_buffer(inst.getMaster()):
                remove_buffer(inst)
                n_removed += 1
                print(f"[INFO] Removed unplaceable buffer: {name}")
            else:
                print(f"[WARN] Non-buffer instance failed DPL: {name}")
                for alt in timing.equivCells(inst.getMaster()):
                    print(f"[WARN]   equivalent cell: {alt.getName()}")

        if n_removed > 0:
            rc     = _run_dpl(design)
            failed = _collect_dpl_failures(design) if str(rc).strip() != "0" else []

    try:
        rc_int = int(str(rc).strip())
    except ValueError:
        rc_int = 1
    return rc_int, failed


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _run_dpl(design) -> object:
    tcl = (
        "set dpl_rc [catch { detailed_placement } dpl_err]\n"
        "if { $dpl_rc != 0 } { puts \"[WARN] detailed_placement: $dpl_err\" }\n"
        "return $dpl_rc"
    )
    return design.evalTclString(tcl)


def _collect_dpl_failures(design) -> List[str]:
    block = design.getBlock() if design is not None else None
    if block is None:
        return []

    cat = block.findMarkerCategory("DPL")
    if cat is None:
        return []
    fail_cat = cat.findMarkerCategory("Placement_failures")
    if fail_cat is None:
        return []

    names: List[str] = []
    for marker in fail_cat.getMarkers():
        raw = marker.getName()
        if raw.startswith("Layer: "):
            raw = raw.split(" ", 2)[-1]
        for item in raw.split(","):
            item = item.strip()
            if item:
                names.append(item)

    # Deduplicate while preserving order.
    seen: set = set()
    unique = [n for n in names if not (n in seen or seen.add(n))]
    if unique:
        print(f"[WARN] DPL failed for {len(unique)} instance(s): {', '.join(unique)}")
    return unique
