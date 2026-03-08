



#Danger buffer testcsae
from pathlib import Path

def _bbox_to_xyxy(bbox):
    if bbox is None:
        return None
    return (
        float(bbox.xMin()),
        float(bbox.yMin()),
        float(bbox.xMax()),
        float(bbox.yMax()),
    )


def build_danger_buffer_input(design, stt_mgr, circuitLib, include_zero_len=False, include_soft=False, include_macro=True):
    _ = circuitLib
    block = design.getBlock()
    boundary = _bbox_to_xyxy(block.getBBox())

    blockages = []
    for blockage in block.getBlockages():
        if not include_soft and hasattr(blockage, "isSoft") and blockage.isSoft():
            continue
        rect = _bbox_to_xyxy(blockage.getBBox())
        if rect is not None:
            blockages.append(list(rect))

    if include_macro:
        for inst in block.getInsts():
            master = inst.getMaster()
            if master is None or not master.isBlock():
                continue
            rect = _bbox_to_xyxy(inst.getBBox())
            if rect is not None:
                blockages.append(list(rect))

    segments = []
    edges = getattr(stt_mgr, "netArcSegments", None)
    if not edges:
        edges = []
        for stp in stt_mgr.stpList:
            if not getattr(stp, "prevs", None):
                continue
            for prev in stp.prevs:
                if getattr(prev, "Net", None) is None or getattr(stp, "Net", None) is None:
                    continue
                if prev.Net.db_net.getName() != stp.Net.db_net.getName():
                    continue
                edges.append((prev, stp))
    for prev, stp in edges:
        sp_idx = int(prev.idx)
        ep_idx = int(stp.idx)
        if not include_zero_len and sp_idx == ep_idx:
            continue
        segments.append(
            (
                (sp_idx, float(prev.x), float(prev.y)),
                (ep_idx, float(stp.x), float(stp.y)),
            )
        )

    return boundary, blockages, segments


def format_danger_buffer_input(boundary, blockages, segments):
    def _fmt(v):
        return f"{v:g}"
    lines = ["boundary"]
    lines.append(f"{_fmt(boundary[0])} {_fmt(boundary[1])} {_fmt(boundary[2])} {_fmt(boundary[3])}")
    lines.append("blockage")
    for blk in blockages:
        lines.append(f"{_fmt(blk[0])} {_fmt(blk[1])} {_fmt(blk[2])} {_fmt(blk[3])}")
    lines.append("segment")
    for (sp_idx, sp_x, sp_y), (ep_idx, ep_x, ep_y) in segments:
        lines.append(f"{sp_idx},{_fmt(sp_x)},{_fmt(sp_y)} {ep_idx},{_fmt(ep_x)},{_fmt(ep_y)}")
    return "\n".join(lines)


def write_danger_buffer_testcase_file(
    design,
    stt_mgr,
    circuitLib,
    file_path,
    include_zero_len=False,
    include_soft=False,
    include_macro=True,
):
    boundary, blockages, segments = build_danger_buffer_input(
        design,
        stt_mgr,
        circuitLib,
        include_zero_len=include_zero_len,
        include_soft=include_soft,
        include_macro=include_macro,
    )
    text = format_danger_buffer_input(boundary, blockages, segments)
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return str(out_path)
