"""Test BPMN and drawio parsing."""
import io
import shutil
import sys
import tempfile
from pathlib import Path

# Fix Windows console encoding for Cyrillic output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.diagram_formats import is_drawio_xml, parse_bpmn, parse_drawio


def main() -> None:
    """Run format parsing tests on available sample files."""
    # Use data from train set (data/test/ may not have bpmn/drawio)
    bpmn = ROOT / "data" / "train" / "diagrams" / "БиблиотечныйСервис_Диаграммы" / "BPMN" / "Process_Booking.bpmn"
    if not bpmn.exists():
        bpmn = ROOT / "data" / "train" / "diagrams" / "diagram.bpmn"
    if bpmn.exists():
        result = parse_bpmn(bpmn)
        print("BPMN:", "None" if result is None else result[:600])
    else:
        print("No BPMN file found")

    drawio = ROOT / "data" / "train" / "diagrams" / "ИСУ_Диаграммы" / "BPMN" / "BPMN.drawio"
    if not drawio.exists():
        drawio = ROOT / "data" / "train" / "diagrams" / "Notion_Диаграммы" / "BPMN" / "bpmn.drawio"
    if drawio.exists():
        result = parse_drawio(drawio)
        print("Drawio:", "None" if result is None else result[:400])
        # Test .xml fallback: copy drawio to temp .xml and verify is_drawio_xml + parse
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
            shutil.copy(drawio, tmp.name)
            tmp_path = Path(tmp.name)
        try:
            assert is_drawio_xml(tmp_path), "is_drawio_xml should recognize drawio content in .xml"
            xml_result = parse_drawio(tmp_path)
            assert xml_result == result, "parse_drawio(.xml) should match parse_drawio(.drawio)"
            print("Drawio .xml fallback: OK")
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        print("No drawio file found")


if __name__ == "__main__":
    main()
