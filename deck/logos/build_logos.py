"""Rasterise simple-icons SVGs recoloured to official brand hexes."""
import json, pathlib, cairosvg

HERE = pathlib.Path(__file__).parent
ICONS = HERE.parent / "node_modules" / "simple-icons" / "icons"
hexes = json.load(open(HERE / "hexes.json"))

# near-black brand marks are invisible on white; deepen to INK navy instead
OVERRIDE = {"nextdotjs": "10243A", "numpy": "4B73C9"}

for name, hexcode in hexes.items():
    hexcode = OVERRIDE.get(name, hexcode)
    svg = (ICONS / f"{name}.svg").read_text()
    svg = svg.replace("<svg ", f'<svg fill="#{hexcode}" ', 1)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(HERE / f"{name}.png"),
                     output_width=256, output_height=256)
    print(name, hexcode)
