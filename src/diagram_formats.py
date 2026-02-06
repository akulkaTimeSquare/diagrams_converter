"""
Поддержка форматов диаграмм: .bpmn, .drawio, .svg, .xml, .uml.
Парсинг BPMN/drawio или конвертация в изображение для VLM.
"""
import re
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

# Расширения, которые обрабатываются как изображения (VLM)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Расширения, требующие конвертации в изображение
CONVERT_TO_IMAGE_EXTENSIONS = {".svg", ".uml", ".puml"}

# Расширения с парсингом XML (без VLM)
PARSE_XML_EXTENSIONS = {".bpmn", ".drawio"}

def is_bpmn_xml(path: Path) -> bool:
    """Проверить, что XML файл похож на BPMN (корень definitions с bpmn)."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        tag = (root.tag or "").lower()
        return "definitions" in tag or "bpmn" in tag
    except Exception:
        return False


SUPPORTED_EXTENSIONS = (
    IMAGE_EXTENSIONS
    | CONVERT_TO_IMAGE_EXTENSIONS
    | PARSE_XML_EXTENSIONS
    | {".xml", ".uml", ".puml"}
)


def _strip_html(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"<[^>]+>", " ", value).replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").strip()


def parse_bpmn(path: Path) -> str | None:
    """Извлечь шаги и роли из BPMN 2.0 XML. Возвращает текст в формате Шаг | Роль или None."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    ns = "http://www.omg.org/spec/BPMN/20100524/MODEL"
    def local(elem):
        return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

    # Участники: processRef -> имя роли
    participant_by_process: dict[str, str] = {}
    for elem in root.iter():
        if local(elem) == "participant":
            pref = elem.get("processRef")
            name = (elem.get("name") or "").strip()
            if pref and name:
                participant_by_process[pref] = name

    # Lane: flowNodeRef -> lane name
    lane_names: dict[str, str] = {}
    flow_node_to_lane: dict[str, str] = {}
    for elem in root.iter():
        if local(elem) == "lane":
            lid = elem.get("id")
            lname = (elem.get("name") or "").strip()
            if lid:
                lane_names[lid] = lname
            for ref in elem:
                if local(ref) == "flowNodeRef" and ref.text and lid:
                    flow_node_to_lane[ref.text] = lane_names.get(lid, lname)

    # Обход дерева: current_process — текущий контейнер (process/subProcess), root_process — process с участником (для роли)
    flow_nodes: dict[str, tuple[str, str]] = {}
    sequence_flows: list[tuple[str, str]] = []
    node_to_process: dict[str, str] = {}  # для роли: участник = participant_by_process[node_to_process[nid]]
    start_event_ids: set[str] = set()
    node_inside_subprocess: set[str] = set()  # id узлов, находящихся внутри subProcess (для порядка вывода)

    def walk_tree(elem, current_process: str, root_process: str, inside_sub: bool = False) -> None:
        tag_local = local(elem)
        eid = elem.get("id")
        name = (elem.get("name") or "").strip()
        proc = current_process
        root = root_process
        in_sub = inside_sub
        if tag_local == "process" and eid:
            proc = eid
            root = eid
            in_sub = False
        elif tag_local == "startEvent" and eid:
            start_event_ids.add(eid)
        elif tag_local == "subProcess":
            if eid and name:
                flow_nodes[eid] = (name, "subProcess")
                node_to_process[eid] = root
            if eid:
                proc = eid
                in_sub = True
        elif tag_local in ("task", "userTask", "serviceTask", "sendTask", "scriptTask", "manualTask", "receiveTask", "businessRuleTask"):
            if eid and name:
                flow_nodes[eid] = (name, "task")
                node_to_process[eid] = root
                if in_sub:
                    node_inside_subprocess.add(eid)
        elif tag_local == "sequenceFlow":
            src, tgt = elem.get("sourceRef"), elem.get("targetRef")
            if src and tgt:
                sequence_flows.append((src, tgt))
        for child in elem:
            walk_tree(child, proc, root, in_sub)

    walk_tree(root, "", "", False)

    node_to_role = {}
    for nid in flow_nodes:
        proc_id = node_to_process.get(nid, "")
        role = flow_node_to_lane.get(nid) or participant_by_process.get(proc_id) or ""
        node_to_role[nid] = role

    task_ids = [nid for nid, (_, t) in flow_nodes.items() if t in ("task", "subProcess")]
    if not task_ids:
        task_ids = list(flow_nodes.keys())

    out_edges: dict[str, list[str]] = {}
    for src, tgt in sequence_flows:
        out_edges.setdefault(src, []).append(tgt)
    no_incoming = set(flow_nodes) - {t for _, t in sequence_flows}
    if no_incoming:
        start_ids = list(no_incoming)
    else:
        # Порядок от начала процесса: узлы, следующие за startEvent; сначала верхний уровень, потом внутри subProcess
        start_ids = [tgt for src, tgt in sequence_flows if src in start_event_ids and tgt in flow_nodes]
        start_ids.sort(key=lambda nid: (nid in node_inside_subprocess, nid))
        if not start_ids:
            start_ids = list(flow_nodes.keys())[:1]

    seen = set()
    order: list[str] = []

    def walk(nid: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        if nid in flow_nodes:
            order.append(nid)
        for next_id in out_edges.get(nid, []):
            walk(next_id)

    for sid in start_ids:
        walk(sid)
    for nid in flow_nodes:
        if nid not in seen:
            walk(nid)

    step_order = [nid for nid in order if nid in flow_nodes and flow_nodes[nid][1] in ("task", "subProcess")]
    if not step_order:
        step_order = order

    has_roles = any(node_to_role.get(nid) for nid in step_order)
    lines = ["Шаг\t|\tРоль"] if has_roles else ["Шаг"]
    if has_roles:
        max_len = max(len(flow_nodes[nid][0]) for nid in step_order)
        for i, nid in enumerate(step_order, 1):
            name = flow_nodes[nid][0]
            role = node_to_role.get(nid, "")
            pad = "\t" * max(1, (max_len - len(name)) // 4 + 1)
            lines.append(f"{i}. {name}{pad}|\t{role}")
    else:
        for i, nid in enumerate(step_order, 1):
            lines.append(f"{i}. {flow_nodes[nid][0]}")

    return "\n".join(lines) if lines else None


def parse_drawio(path: Path) -> str | None:
    """Извлечь шаги из draw.io (diagrams.net) XML. Роли по swimlane."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    if "mxfile" not in (root.tag or "").lower():
        return None

    # Клетки с текстом: (eid, text, parent, x, y)
    cells: list[tuple[str, str, str, float, float]] = []
    swimlanes: dict[str, str] = {}

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag != "mxCell":
            continue
        eid = elem.get("id")
        value = elem.get("value") or ""
        style = (elem.get("style") or "").lower()
        parent = elem.get("parent") or ""
        edge = "edge=1" in style or elem.get("edge") == "1"
        if not eid:
            continue

        if "swimlane" in style and value:
            swimlanes[eid] = _strip_html(value)

        x, y = 0.0, 0.0
        geom = elem.find("mxGeometry") or elem.find("{*}mxGeometry")
        if geom is not None:
            x = float(geom.get("x") or 0)
            y = float(geom.get("y") or 0)

        if value and not edge:
            text = _strip_html(value)
            if text and len(text) < 500:
                cells.append((eid, text, parent, x, y))

    # Исключаем короткие метки-роли и дубликаты; сортируем по y, x
    skip = {"ИСУ", "Шаг", "Роль"}
    by_xy: list[tuple[float, float, str, str]] = []
    seen = set()
    for eid, text, parent, x, y in cells:
        if not text or text in skip or text in seen:
            continue
        seen.add(text)
        role = swimlanes.get(parent, "")
        by_xy.append((y, x, text, role))

    by_xy.sort(key=lambda t: (t[0], t[1]))
    unique_steps: list[tuple[str, str]] = [(t[2], t[3]) for t in by_xy]
    if not unique_steps:
        return None

    has_roles = any(r for _, r in unique_steps)
    lines = ["Шаг\t|\tРоль"] if has_roles else ["Шаг"]
    if has_roles:
        max_len = max(len(t) for t, _ in unique_steps)
        for i, (text, role) in enumerate(unique_steps, 1):
            pad = "\t" * max(1, (max_len - len(text)) // 4 + 1)
            lines.append(f"{i}. {text}{pad}|\t{role}")
    else:
        for i, (text, _) in enumerate(unique_steps, 1):
            lines.append(f"{i}. {text}")

    return "\n".join(lines)


def _convert_svg_cairosvg(svg_path: Path, png_path: Path) -> None:
    """Конвертировать SVG в PNG через cairosvg (требует системную библиотеку Cairo)."""
    import cairosvg
    cairosvg.convert_file(url=str(svg_path.resolve()), write_to=str(png_path))


def _convert_svg_resvg(svg_path: Path, png_path: Path) -> None:
    """Конвертировать SVG в PNG через resvg_py (работает на Windows без Cairo)."""
    import resvg_py
    svg_bytes = svg_path.read_bytes()
    svg_str = svg_bytes.decode("utf-8", errors="replace")
    png_bytes = resvg_py.svg_to_bytes(svg_string=svg_str)
    png_path.write_bytes(png_bytes)


def convert_svg_to_png(svg_path: Path) -> Path:
    """Конвертировать SVG в PNG во временный файл.
    Сначала пробует cairosvg; при отсутствии Cairo (например на Windows) — resvg_py."""
    fd, png_path = tempfile.mkstemp(suffix=".png")
    import os
    os.close(fd)
    png_path = Path(png_path)
    # 1) cairosvg (на Linux часто уже есть Cairo)
    try:
        _convert_svg_cairosvg(svg_path, png_path)
        return png_path
    except ImportError:
        pass
    except (OSError, Exception) as e:
        err_msg = str(e).lower()
        if "cairo" in err_msg or "libcairo" in err_msg or "cannot load library" in err_msg:
            pass  # Cairo не найден — пробуем resvg
        else:
            raise
    # 2) resvg_py (Windows/все ОС без системного Cairo)
    try:
        _convert_svg_resvg(svg_path, png_path)
        return png_path
    except ImportError:
        raise ImportError(
            "Для конвертации .svg в PNG установите один из вариантов:\n"
            "  pip install cairosvg   (нужна системная библиотека Cairo)\n"
            "  pip install resvg-py    (работает на Windows без Cairo)"
        )


def render_plantuml_to_png(path: Path) -> Path | None:
    """Отрендерить PlantUML (.uml/.puml) в PNG. Требует plantuml (сервер или jar)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return render_plantuml_from_string(text)


def _plantuml_fetch_png_hex(plantuml_source: str) -> bytes | None:
    """Запасной рендер: GET к серверу PlantUML с HEX-кодированием (поддерживает UTF-8/кириллицу)."""
    try:
        import urllib.request
        hex_encoded = plantuml_source.encode("utf-8").hex()
        url = f"https://www.plantuml.com/plantuml/png/~h{hex_encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "diagrams-service/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return None
            data = resp.read()
            if not data or len(data) < 20:
                return None
            if data[:8] == b"<html>" or data[:4] != b"\x89PNG":
                return None
            return data
    except Exception:
        return None


def render_plantuml_from_string(plantuml_source: str) -> Path | None:
    """Отрендерить исходный код PlantUML в PNG. Возвращает путь к временному PNG или None."""
    if not plantuml_source or "@startuml" not in plantuml_source.lower():
        return None
    png_bytes = None
    try:
        import plantuml
        p = plantuml.PlantUML(url="https://www.plantuml.com/plantuml")
        png_bytes = p.processes(plantuml_source)
    except ImportError:
        pass
    if not png_bytes:
        png_bytes = _plantuml_fetch_png_hex(plantuml_source)
    if not png_bytes:
        return None
    fd, out = tempfile.mkstemp(suffix=".png")
    import os
    os.close(fd)
    Path(out).write_bytes(png_bytes)
    return Path(out)


def resolve_to_image_path(path: Path) -> tuple[Path, bool]:
    """
    Если путь — не растровое изображение, конвертировать во временный PNG.
    Возвращает (путь к изображению, нужно_удалить_временный_файл).
    """
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return path, False
    if ext == ".svg":
        png = convert_svg_to_png(path)
        return png, True
    if ext in (".uml", ".puml"):
        png = render_plantuml_to_png(path)
        if png:
            return png, True
    raise ValueError(f"Не удалось получить изображение для {path.suffix}")
