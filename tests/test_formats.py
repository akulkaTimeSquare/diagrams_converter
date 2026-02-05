"""Quick test for diagram format parsers."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.diagram_formats import parse_bpmn, parse_drawio, is_bpmn_xml

def main():
    bpmn = ROOT / "Диаграммы" / "diagram.bpmn"
    if bpmn.exists():
        r = parse_bpmn(bpmn)
        print("BPMN (diagram.bpmn):", (r or "None")[:600])
    drawio = ROOT / "Диаграммы" / "ИСУ_Диаграммы 2" / "BPMN" / "BPMN.drawio"
    if drawio.exists():
        r2 = parse_drawio(drawio)
        print("Drawio (first 400):", (r2 or "None")[:400])

if __name__ == "__main__":
    main()
