"""
Diagram format support: .bpmn, .drawio, .svg, .xml, .uml, .puml.

Provides BPMN/drawio XML parsing and conversion to images for VLM processing.
"""
import logging
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# Constants

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
CONVERT_TO_IMAGE_EXTENSIONS = {".svg", ".uml", ".puml"}
PARSE_XML_EXTENSIONS = {".bpmn", ".drawio"}

SUPPORTED_EXTENSIONS = (
    IMAGE_EXTENSIONS
    | CONVERT_TO_IMAGE_EXTENSIONS
    | PARSE_XML_EXTENSIONS
    | {".xml", ".uml", ".puml"}
)

BPMN_TASK_TAGS = (
    "task", "userTask", "serviceTask", "sendTask", "scriptTask",
    "manualTask", "receiveTask", "businessRuleTask",
)


# Helpers

def _local_tag(elem: ET.Element) -> str:
    """Get local tag name"""
    tag = elem.tag or ""
    return tag.split("}")[-1] if "}" in tag else tag


def _strip_html(value: str) -> str:
    """Remove HTML tags and decode common entities from a string"""
    if not value:
        return ""
    return re.sub(r"<[^>]+>", " ", value).replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").strip()


def _format_steps_output(
    step_order: list,
    get_name: Callable,
    get_role: Callable,
) -> list[str]:
    """Format steps as 'Step | Role' or 'Step' lines"""
    has_roles = any(get_role(s) for s in step_order)
    header = ["Шаг\t|\tРоль"] if has_roles else ["Шаг"]
    if has_roles:
        max_len = max(len(get_name(s)) for s in step_order)
        for i, step in enumerate(step_order, 1):
            name = get_name(step)
            role = get_role(step)
            pad = "\t" * max(1, (max_len - len(name)) // 4 + 1)
            header.append(f"{i}. {name}{pad}|\t{role}")
    else:
        for i, step in enumerate(step_order, 1):
            header.append(f"{i}. {get_name(step)}")
    return header


# BPMN parsing

def is_bpmn_xml(path: Path) -> bool:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        tag = (root.tag or "").lower()
        return "definitions" in tag or "bpmn" in tag
    except Exception:
        return False


def is_drawio_xml(path: Path) -> bool:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        tag = (root.tag or "").lower()
        return "mxfile" in tag
    except Exception:
        return False


def _bpmn_collect_participants_and_lanes(root: ET.Element) -> tuple[dict[str, str], dict[str, str]]:
    """Collect participant (processRef -> role) and flowNodeRef -> lane mappings"""
    participant_by_process: dict[str, str] = {}
    flow_node_to_lane: dict[str, str] = {}
    lane_names: dict[str, str] = {}

    for elem in root.iter():
        tag = _local_tag(elem)
        if tag == "participant":
            pref, name = elem.get("processRef"), (elem.get("name") or "").strip()
            if pref and name:
                participant_by_process[pref] = name
        elif tag == "lane":
            lid, lname = elem.get("id"), (elem.get("name") or "").strip()
            if lid:
                lane_names[lid] = lname
            for ref in elem:
                if _local_tag(ref) == "flowNodeRef" and ref.text and lid:
                    flow_node_to_lane[ref.text] = lane_names.get(lid, lname)

    return participant_by_process, flow_node_to_lane


def _bpmn_walk_tree(
    root: ET.Element,
) -> tuple[dict[str, tuple[str, str]], list[tuple[str, str]], dict[str, str], set[str], set[str]]:
    """
    Walk BPMN tree and collect flow nodes, sequence flows, node-to-process, start events, subprocess nodes
    """
    flow_nodes: dict[str, tuple[str, str]] = {}
    sequence_flows: list[tuple[str, str]] = []
    node_to_process: dict[str, str] = {}
    start_event_ids: set[str] = set()
    node_inside_subprocess: set[str] = set()

    def walk(elem: ET.Element, current_process: str, root_process: str, inside_sub: bool) -> None:
        tag = _local_tag(elem)
        eid = elem.get("id")
        name = (elem.get("name") or "").strip()
        proc, root_p, in_sub = current_process, root_process, inside_sub

        if tag == "process" and eid:
            proc, root_p, in_sub = eid, eid, False

        elif tag == "startEvent" and eid:
            start_event_ids.add(eid)

        elif tag == "subProcess":
            if eid and name:
                flow_nodes[eid] = (name, "subProcess")
                node_to_process[eid] = root_p
            if eid:
                proc, in_sub = eid, True

        elif tag in BPMN_TASK_TAGS:
            if eid and name:
                flow_nodes[eid] = (name, "task")
                node_to_process[eid] = root_p
                if in_sub:
                    node_inside_subprocess.add(eid)

        elif tag == "sequenceFlow":
            src, tgt = elem.get("sourceRef"), elem.get("targetRef")
            if src and tgt:
                sequence_flows.append((src, tgt))

        for child in elem:
            walk(child, proc, root_p, in_sub)

    walk(root, "", "", False)
    return flow_nodes, sequence_flows, node_to_process, start_event_ids, node_inside_subprocess


def _bpmn_compute_step_order(
    flow_nodes: dict[str, tuple[str, str]],
    sequence_flows: list[tuple[str, str]],
    start_event_ids: set[str],
    node_inside_subprocess: set[str],
) -> list[str]:
    """Compute topological order of steps from sequence flows"""
    out_edges: dict[str, list[str]] = {}
    for src, tgt in sequence_flows:
        out_edges.setdefault(src, []).append(tgt)

    targets = {t for _, t in sequence_flows}
    no_incoming = set(flow_nodes) - targets

    if no_incoming:
        start_ids = list(no_incoming)
    else:
        start_ids = [
            tgt for src, tgt in sequence_flows
            if src in start_event_ids and tgt in flow_nodes
        ]
        start_ids.sort(key=lambda nid: (nid in node_inside_subprocess, nid))
        if not start_ids:
            start_ids = list(flow_nodes.keys())[:1]

    seen: set[str] = set()
    order: list[str] = []

    def dfs(nid: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        if nid in flow_nodes:
            order.append(nid)
        for next_id in out_edges.get(nid, []):
            dfs(next_id)

    for sid in start_ids:
        dfs(sid)
    for nid in flow_nodes:
        if nid not in seen:
            dfs(nid)

    step_order = [
        nid for nid in order
        if nid in flow_nodes and flow_nodes[nid][1] in ("task", "subProcess")
    ]
    return step_order if step_order else order


def parse_bpmn(path):
    """
    Extract steps and roles from BPMN file

    Returns text description of the algorithm/process
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    participant_by_process, flow_node_to_lane = _bpmn_collect_participants_and_lanes(root)
    flow_nodes, sequence_flows, node_to_process, start_event_ids, node_inside_subprocess = (
        _bpmn_walk_tree(root)
    )

    node_to_role = {
        nid: flow_node_to_lane.get(nid) or participant_by_process.get(node_to_process.get(nid, ""), "")
        for nid in flow_nodes
    }

    step_order = _bpmn_compute_step_order(
        flow_nodes, sequence_flows, start_event_ids, node_inside_subprocess
    )

    lines = _format_steps_output(
        step_order,
        get_name=lambda nid: flow_nodes[nid][0],
        get_role=lambda nid: node_to_role.get(nid, ""),
    )
    return "\n".join(lines) if lines else None


# Draw.io parsing

def _drawio_collect_cells_and_swimlanes(root: ET.Element) -> tuple[list[tuple[str, str, str, float, float]], dict[str, str]]:
    """Collect mxCell text cells and swimlane labels from draw.io XML"""
    cells: list[tuple[str, str, str, float, float]] = []
    swimlanes: dict[str, str] = {}
    skip = {"ИСУ", "Шаг", "Роль"}

    for elem in root.iter():
        tag = _local_tag(elem)
        if tag != "mxCell":
            continue

        eid = elem.get("id")
        value = elem.get("value") or ""
        style = (elem.get("style") or "").lower()
        parent = elem.get("parent") or ""
        is_edge = "edge=1" in style or elem.get("edge") == "1"

        if not eid:
            continue

        if "swimlane" in style and value:
            swimlanes[eid] = _strip_html(value)

        x, y = 0.0, 0.0
        geom = elem.find("mxGeometry") or elem.find("{*}mxGeometry")
        if geom is not None:
            x = float(geom.get("x") or 0)
            y = float(geom.get("y") or 0)

        if value and not is_edge:
            text = _strip_html(value)
            if text and len(text) < 500 and text not in skip:
                cells.append((eid, text, parent, x, y))

    return cells, swimlanes


def parse_drawio(path: Path) -> str | None:
    """
    Extract steps from draw.io XML

    Roles are derived from swimlane containers.
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    if "mxfile" not in (root.tag or "").lower():
        return None

    cells, swimlanes = _drawio_collect_cells_and_swimlanes(root)

    # Deduplicate by text, sort by position
    seen: set[str] = set()
    by_xy: list[tuple[float, float, str, str]] = []
    for eid, text, parent, x, y in cells:
        if not text or text in seen:
            continue
        seen.add(text)
        role = swimlanes.get(parent, "")
        by_xy.append((y, x, text, role))

    by_xy.sort(key=lambda t: (t[0], t[1]))
    unique_steps = [(t[2], t[3]) for t in by_xy]
    if not unique_steps:
        return None

    lines = _format_steps_output(
        unique_steps,
        get_name=lambda s: s[0],
        get_role=lambda s: s[1],
    )
    return "\n".join(lines)


# SVG conversion

def _is_cairo_related_error(exc: Exception) -> bool:
    """Check if exception is due to missing/unavailable Cairo library"""
    msg = str(exc).lower()
    return "cairo" in msg or "libcairo" in msg or "cannot load library" in msg


def _convert_svg_cairosvg(svg_path: Path, png_path: Path) -> None:
    """Convert SVG to PNG via cairosvg (requires system Cairo library)"""
    import cairosvg
    cairosvg.convert_file(url=str(svg_path.resolve()), write_to=str(png_path))


def _convert_svg_resvg(svg_path: Path, png_path: Path) -> None:
    """Convert SVG to PNG via resvg_py (works on Windows without Cairo)"""
    import resvg_py
    svg_str = svg_path.read_bytes().decode("utf-8", errors="replace")
    png_path.write_bytes(resvg_py.svg_to_bytes(svg_string=svg_str))


def convert_svg_to_png(svg_path: Path) -> Path:
    """
    Convert SVG to PNG in a temporary file.

    Tries cairosvg first; falls back to resvg_py when Cairo is unavailable.
    """
    fd, png_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    png_path = Path(png_path)

    try:
        _convert_svg_cairosvg(svg_path, png_path)
        return png_path
    except ImportError:
        pass
    except (OSError, Exception) as e:
        if not _is_cairo_related_error(e):
            raise

    try:
        _convert_svg_resvg(svg_path, png_path)
        return png_path
    except ImportError:
        raise ImportError(
            "For .svg to PNG conversion install one of:\n"
            "  pip install cairosvg   (requires system Cairo library)\n"
            "  pip install resvg-py   (works on Windows without Cairo)"
        )


# PlantUML rendering

def _plantuml_render_via_library(plantuml_source: str) -> bytes | None:
    """Render PlantUML via plantuml Python package"""
    try:
        import plantuml
        # URL must end with /img/ for PNG; package appends encoded text
        p = plantuml.PlantUML(url="https://www.plantuml.com/plantuml/img/")
        result = p.processes(plantuml_source)
        if result and len(result) > 20 and result[:4] == b"\x89PNG":
            return result
        # Server returned non-PNG (e.g. HTML error page)
        preview = (result[:200].decode("utf-8", errors="replace") if result else "empty")
        logger.warning("PlantUML library: server returned non-PNG (len=%s), preview: %s", len(result or b""), preview[:80])
        return None
    except ImportError:
        logger.warning("PlantUML library: import failed (pip install plantuml)")
        return None
    except Exception as e:
        logger.warning("PlantUML library: %s", e)
        return None


def _plantuml_fetch_png_hex(plantuml_source: str) -> bytes | None:
    """GET request to PlantUML server with HEX encoding"""
    try:
        hex_encoded = plantuml_source.encode("utf-8").hex()
        url = f"https://www.plantuml.com/plantuml/png/~h{hex_encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "diagrams-service/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                logger.warning("PlantUML fetch: HTTP %s", resp.status)
                return None
            data = resp.read()
            if not data or len(data) < 20:
                return None
            if data[:8] == b"<html>" or data[:4] != b"\x89PNG":
                preview = data[:200].decode("utf-8", errors="replace")
                logger.warning("PlantUML fetch: server returned non-PNG, preview: %s", preview[:80])
                return None
            return data
    except Exception as e:
        logger.warning("PlantUML fetch: %s", e)
        return None


def _write_temp_png(png_bytes: bytes) -> Path:
    """Write bytes to a temporary PNG file and return path"""
    fd, out = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Path(out).write_bytes(png_bytes)
    return Path(out)


def render_plantuml_from_string(plantuml_source: str) -> Path | None:
    """Render PlantUML source code to PNG. Returns path to temporary PNG file or None on failure"""
    if not plantuml_source or "@startuml" not in plantuml_source.lower():
        logger.warning("PlantUML render: invalid source (empty or missing @startuml)")
        return None

    png_bytes = _plantuml_render_via_library(plantuml_source)
    if not png_bytes:
        png_bytes = _plantuml_fetch_png_hex(plantuml_source)
    if not png_bytes:
        logger.warning("PlantUML render: both library and fetch failed. Source snippet: %s", plantuml_source[:300])
        return None

    return _write_temp_png(png_bytes)


def render_plantuml_to_png(path: Path) -> Path | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    return render_plantuml_from_string(text)
