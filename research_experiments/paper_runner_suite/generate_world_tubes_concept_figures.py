from __future__ import annotations

"""Generate deterministic, result-free concept figures for the World Tubes paper.

The figures in this module explain the method and implementation contract. They
do not read benchmark artifacts, import Torch, or encode measured values. The
same source strings always produce the same UTF-8 SVG bytes.
"""

import argparse
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = (
    ROOT / "research_notes" / "gauged_uvt_trace_atlas" / "paper" / "figures"
)
FIGURE_FILENAMES = (
    "world_tubes_system_overview.svg",
    "world_tubes_projective_compiler.svg",
)

SYSTEM_REQUIRED_LABELS = (
    "Same learned world",
    "Known camera program",
    "Per-frame replay",
    "Camera-program compiler",
    "Projective trace atlas",
    "Certified tile-time cells",
    "Metal interval forward",
    "Metal interval VJP",
    "Compiler VJP",
    "World gradients",
    "unavoidable F × H × W outputs",
)
PROJECTIVE_REQUIRED_LABELS = (
    "Homogeneous camera trace",
    "h_i(t) = P(t) X_i^h(t)",
    "u_i = h_u / h_z",
    "Candidate bounded interval",
    "Certify denominator",
    "Fit / certify trace",
    "Certify support",
    "Certify visibility",
    "Lower interval record",
    "Split interval",
    "Explicit fallback",
    "precompiled cell order",
    "no arbitrary 360° / 720° claim",
)


CSS = """
.background { fill: #ffffff; }
.title { fill: #0f172a; font-family: Helvetica, Arial, sans-serif; font-size: 28px; font-weight: 700; }
.subtitle { fill: #475569; font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 400; }
.lane-title { fill: #0f172a; font-family: Helvetica, Arial, sans-serif; font-size: 21px; font-weight: 700; }
.node-title { fill: #0f172a; font-family: Helvetica, Arial, sans-serif; font-size: 19px; font-weight: 700; }
.node-title-compact { fill: #0f172a; font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 700; }
.node-title-tight { fill: #0f172a; font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 700; }
.body { fill: #334155; font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 400; }
.small { fill: #475569; font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 400; }
.mono { fill: #172554; font-family: Menlo, Consolas, monospace; font-size: 18px; font-weight: 600; }
.badge { fill: #ffffff; font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 700; }
.arrow { fill: none; stroke: #334155; stroke-width: 2.4; }
.arrow-blue { fill: none; stroke: #2563eb; stroke-width: 2.8; }
.arrow-red { fill: none; stroke: #be123c; stroke-width: 2.4; }
.arrow-green { fill: none; stroke: #047857; stroke-width: 2.8; }
.loop { fill: none; stroke: #7c3aed; stroke-width: 2.5; stroke-dasharray: 8 6; }
.boundary { fill: #fff7ed; stroke: #c2410c; stroke-width: 1.8; }
.input { fill: #f8fafc; stroke: #64748b; stroke-width: 1.8; }
.replay { fill: #fff1f2; stroke: #fb7185; stroke-width: 1.6; }
.compile { fill: #eff6ff; stroke: #60a5fa; stroke-width: 1.8; }
.cert { fill: #ecfdf5; stroke: #34d399; stroke-width: 1.8; }
.decision { fill: #fffbeb; stroke: #f59e0b; stroke-width: 2; }
.fallback { fill: #fef2f2; stroke: #ef4444; stroke-width: 2; }
.note { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1.4; }
""".strip()

PATH_ATTRIBUTES = {
    "arrow": 'fill="none" stroke="#334155" stroke-width="2.4"',
    "arrow-blue": 'fill="none" stroke="#2563eb" stroke-width="2.8"',
    "arrow-red": 'fill="none" stroke="#be123c" stroke-width="2.4"',
    "arrow-green": 'fill="none" stroke="#047857" stroke-width="2.8"',
    "loop": 'fill="none" stroke="#7c3aed" stroke-width="2.5" stroke-dasharray="8 6"',
}
ARROW_COLORS = {
    "arrow": "#334155",
    "arrow-blue": "#2563eb",
    "arrow-red": "#be123c",
    "arrow-green": "#047857",
    "loop": "#7c3aed",
}


@dataclass
class Svg:
    width: int
    height: int
    title: str
    description: str
    elements: list[str] = field(default_factory=list)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        css_class: str,
        rx: float = 14,
    ) -> None:
        self.elements.append(
            f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
            f'rx="{rx:g}" class="{css_class}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        css_class: str = "arrow",
    ) -> None:
        self.elements.append(
            f'<path d="M {x1:g} {y1:g} L {x2:g} {y2:g}" '
            f'class="{css_class}" {PATH_ATTRIBUTES[css_class]}/>'
        )
        self._arrowhead((x1, y1), (x2, y2), css_class=css_class)

    def path(self, points: tuple[tuple[float, float], ...], *, css_class: str) -> None:
        commands = " ".join(
            ("M" if index == 0 else "L") + f" {x:g} {y:g}"
            for index, (x, y) in enumerate(points)
        )
        self.elements.append(
            f'<path d="{commands}" class="{css_class}" '
            f'{PATH_ATTRIBUTES[css_class]}/>'
        )
        self._arrowhead(points[-2], points[-1], css_class=css_class)

    def _arrowhead(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        css_class: str,
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = hypot(dx, dy)
        if length == 0:
            raise ValueError("arrowhead requires a non-zero final segment")
        ux, uy = dx / length, dy / length
        base_x, base_y = end[0] - 10 * ux, end[1] - 10 * uy
        perp_x, perp_y = -uy, ux
        points = (
            end,
            (base_x + 4.5 * perp_x, base_y + 4.5 * perp_y),
            (base_x - 4.5 * perp_x, base_y - 4.5 * perp_y),
        )
        encoded = " ".join(f"{x:g},{y:g}" for x, y in points)
        self.elements.append(
            f'<polygon points="{encoded}" fill="{ARROW_COLORS[css_class]}"/>'
        )

    def polygon(self, points: tuple[tuple[float, float], ...], *, css_class: str) -> None:
        encoded = " ".join(f"{x:g},{y:g}" for x, y in points)
        self.elements.append(f'<polygon points="{encoded}" class="{css_class}"/>')

    def text(
        self,
        x: float,
        y: float,
        lines: tuple[str, ...],
        *,
        css_class: str = "body",
        anchor: str = "start",
        line_height: int = 19,
    ) -> None:
        spans = []
        for index, line in enumerate(lines):
            dy = 0 if index == 0 else line_height
            spans.append(
                f'<tspan x="{x:g}" dy="{dy:g}">{escape(line)}</tspan>'
            )
        self.elements.append(
            f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" '
            f'class="{css_class}">{"".join(spans)}</text>'
        )

    def badge(self, x: float, y: float, label: str, *, fill: str) -> None:
        width = max(80, len(label) * 10 + 24)
        self.elements.append(
            f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="30" '
            f'rx="15" fill="{fill}"/>'
        )
        self.text(
            x + width / 2,
            y + 21,
            (label,),
            css_class="badge",
            anchor="middle",
        )

    def render(self) -> str:
        content = "\n".join(self.elements)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" '
            'role="img" aria-labelledby="figure-title figure-description">\n'
            f'<title id="figure-title">{escape(self.title)}</title>\n'
            f'<desc id="figure-description">{escape(self.description)}</desc>\n'
            f"<style>{CSS}</style>\n{content}\n</svg>\n"
        )


def system_overview_svg() -> str:
    svg = Svg(
        1200,
        700,
        "World Tubes camera-program compiler and compiled adjoint",
        (
            "The Same learned world and Known camera program enter Per-frame replay "
            "or World Tubes compilation. A Camera-program compiler produces a "
            "Projective trace atlas and Certified tile-time cells for the Metal "
            "interval forward, Metal interval VJP, Compiler VJP, and World gradients."
        ),
    )
    svg.rect(0, 0, 1200, 700, css_class="background", rx=0)
    svg.text(40, 43, ("World Tubes: compile world-side work across a camera program",), css_class="title")
    svg.text(
        40,
        72,
        ("One learned world and camera program; two routes to the same sensor samples",),
        css_class="subtitle",
    )

    svg.rect(35, 118, 225, 88, css_class="input")
    svg.text(147, 150, ("Same learned world θ",), css_class="node-title", anchor="middle")
    svg.text(147, 181, ("motion · opacity · color",), css_class="body", anchor="middle")
    svg.rect(35, 238, 225, 88, css_class="input")
    svg.text(147, 260, ("Known camera", "program Γ"), css_class="node-title", anchor="middle", line_height=22)
    svg.text(147, 310, ("poses · intrinsics · times",), css_class="body", anchor="middle")

    svg.rect(300, 100, 860, 185, css_class="replay", rx=18)
    svg.badge(320, 116, "BASELINE", fill="#be123c")
    svg.text(430, 139, ("Per-frame replay",), css_class="lane-title")
    replay_nodes = (
        (330, "Evaluate world", ("at t_f",)),
        (535, "Project + bin", ("support and order",)),
        (740, "Composite", ("one image I_f",)),
        (945, "Repeat", ("f = 1 … F",)),
    )
    for x, title, body in replay_nodes:
        svg.rect(x, 170, 170, 82, css_class="input", rx=11)
        svg.text(x + 85, 201, (title,), css_class="node-title", anchor="middle")
        svg.text(x + 85, 230, body, css_class="small", anchor="middle")
    for start, stop in ((500, 535), (705, 740), (910, 945)):
        svg.line(start, 211, stop - 10, 211, css_class="arrow-red")

    svg.path(((260, 162), (285, 162), (285, 193), (320, 193)), css_class="arrow-red")
    svg.path(((260, 282), (280, 282), (280, 230), (320, 230)), css_class="arrow-red")

    svg.rect(300, 315, 860, 315, css_class="compile", rx=18)
    svg.badge(320, 332, "WORLD TUBES", fill="#2563eb")
    svg.text(468, 355, ("Compile once; evaluate many times",), css_class="lane-title")

    compiled_nodes = (
        (330, ("Camera-program", "compiler"), ("certify / split / fallback",)),
        (535, ("Projective trace", "atlas"), ("traces + intervals",)),
        (740, ("Certified tile-time", "cells"), ("support + order",)),
        (945, ("Metal interval", "forward"), ("slice at t_f",)),
    )
    for x, title, body in compiled_nodes:
        svg.rect(x, 386, 170, 96, css_class="cert" if x == 740 else "input", rx=12)
        svg.text(
            x + 85,
            408,
            title,
            css_class="node-title-tight",
            anchor="middle",
            line_height=21,
        )
        svg.text(x + 85, 464, body, css_class="small", anchor="middle")
    for start, stop in ((500, 535), (705, 740), (910, 945)):
        svg.line(start, 431, stop - 10, 431, css_class="arrow-blue")

    svg.path(((260, 162), (282, 162), (282, 410), (320, 410)), css_class="arrow-blue")
    svg.path(((260, 282), (290, 282), (290, 452), (320, 452)), css_class="arrow-blue")

    backward_nodes = (
        (945, "Metal interval VJP"),
        (740, "Compiler VJP"),
        (535, "World gradients"),
    )
    for x, title in backward_nodes:
        svg.rect(x, 505, 170, 60, css_class="input", rx=11)
        svg.text(x + 85, 542, (title,), css_class="node-title-tight", anchor="middle")
    svg.line(945, 535, 920, 535, css_class="arrow-green")
    svg.line(740, 535, 715, 535, css_class="arrow-green")
    svg.path(((535, 535), (285, 535), (285, 190), (260, 190)), css_class="arrow-green")
    svg.text(1144, 542, ("λ",), css_class="mono", anchor="middle")
    svg.line(1132, 535, 1120, 535, css_class="arrow-green")

    svg.rect(330, 585, 785, 30, css_class="note", rx=8)
    svg.text(722, 606, ("Shared metadata: traces · support · bins · visibility · backward tape",), css_class="small", anchor="middle")
    svg.rect(35, 652, 1125, 34, css_class="boundary", rx=8)
    svg.text(
        597,
        675,
        ("Claim boundary: world-side metadata is shared; unavoidable F × H × W outputs and shading remain.",),
        css_class="small",
        anchor="middle",
    )
    return svg.render()


def projective_compiler_svg() -> str:
    svg = Svg(
        1200,
        760,
        "Projective gauge certification, splitting, lowering, and fallback",
        (
            "A Homogeneous camera trace enters a Candidate bounded interval. "
            "Denominator, fit, support, and visibility checks accept, split, or "
            "explicitly fall back before interval records reach Metal."
        ),
    )
    svg.rect(0, 0, 1200, 760, css_class="background", rx=0)
    svg.text(40, 43, ("Projective gauge domain: certify before dividing or reusing",), css_class="title")
    svg.text(
        40,
        72,
        ("Bounded camera-program segments only · no arbitrary 360° / 720° claim",),
        css_class="subtitle",
    )

    svg.rect(35, 100, 1130, 110, css_class="compile", rx=16)
    svg.text(65, 133, ("Homogeneous camera trace",), css_class="lane-title")
    svg.text(65, 169, ("h_i(t) = P(t) X_i^h(t)",), css_class="mono")
    svg.text(65, 197, ("u_i = h_u / h_z",), css_class="mono")
    svg.text(
        545,
        148,
        ("Keep numerator and denominator separate", "until projective validity is certified"),
        css_class="body",
        line_height=27,
    )
    svg.text(
        930,
        165,
        ("Pole = chart boundary",),
        css_class="small",
    )

    nodes = (
        (25, ("Candidate bounded", "interval"), ("normalize time",)),
        (255, ("Certify denominator",), ("pole + near plane",)),
        (485, ("Fit / certify trace",), ("UV / depth residual",)),
        (715, ("Certify support",), ("footprint + tiles",)),
        (945, ("Certify visibility",), ("cell order",)),
    )
    for x, title, body in nodes:
        svg.rect(x, 250, 210, 100, css_class="cert" if x > 25 else "input", rx=12)
        svg.text(
            x + 105,
            274 if len(title) > 1 else 282,
            title,
            css_class="node-title-compact",
            anchor="middle",
            line_height=21,
        )
        svg.text(x + 105, 328, body, css_class="small", anchor="middle")
    for start, stop in ((235, 255), (465, 485), (695, 715), (925, 945)):
        svg.line(start, 300, stop - 8, 300, css_class="arrow-blue")

    svg.polygon(((1050, 390), (1115, 440), (1050, 490), (985, 440)), css_class="decision")
    svg.text(1050, 433, ("all", "certified?"), css_class="node-title", anchor="middle", line_height=20)
    svg.line(1050, 350, 1050, 385, css_class="arrow-blue")

    svg.rect(900, 525, 265, 108, css_class="cert", rx=13)
    svg.text(1032, 557, ("Lower interval record",), css_class="node-title", anchor="middle")
    svg.text(1032, 588, ("tiles + precompiled cell order",), css_class="small", anchor="middle")
    svg.text(1032, 616, ("Metal forward / VJP",), css_class="small", anchor="middle")
    svg.path(((1115, 440), (1175, 440), (1175, 505), (1032, 505), (1032, 515)), css_class="arrow-green")

    svg.rect(500, 470, 250, 96, css_class="decision", rx=13)
    svg.text(625, 504, ("Split interval",), css_class="node-title", anchor="middle")
    svg.text(625, 536, ("root or midpoint",), css_class="small", anchor="middle")
    svg.path(((985, 440), (770, 440), (770, 518), (760, 518)), css_class="loop")
    svg.path(((500, 518), (130, 518), (130, 360)), css_class="loop")

    svg.rect(390, 605, 330, 82, css_class="fallback", rx=13)
    svg.text(555, 637, ("Explicit fallback",), css_class="node-title", anchor="middle")
    svg.text(555, 668, ("reason-labelled reference route",), css_class="small", anchor="middle")
    svg.path(((625, 566), (625, 586), (555, 586), (555, 595)), css_class="arrow-red")

    svg.rect(25, 590, 320, 108, css_class="note", rx=13)
    svg.text(185, 622, ("Certificate boundary",), css_class="node-title", anchor="middle")
    svg.text(185, 653, ("continuous denominator", "sampled general residual"), css_class="small", anchor="middle", line_height=27)

    svg.rect(25, 716, 1140, 32, css_class="boundary", rx=9)
    svg.text(
        595,
        738,
        ("Any denominator, fit, support, or visibility uncertainty causes a split or explicit fallback.",),
        css_class="small",
        anchor="middle",
    )
    return svg.render()


def expected_figures() -> dict[str, str]:
    return {
        FIGURE_FILENAMES[0]: system_overview_svg(),
        FIGURE_FILENAMES[1]: projective_compiler_svg(),
    }


def write_figures(out_dir: Path = DEFAULT_OUT_DIR) -> tuple[Path, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, source in expected_figures().items():
        path = out_dir / filename
        path.write_text(source, encoding="utf-8", newline="\n")
        written.append(path)
    return tuple(written)


def verify_figure_dir(out_dir: Path = DEFAULT_OUT_DIR) -> list[str]:
    errors: list[str] = []
    expected = expected_figures()
    required = {
        FIGURE_FILENAMES[0]: SYSTEM_REQUIRED_LABELS,
        FIGURE_FILENAMES[1]: PROJECTIVE_REQUIRED_LABELS,
    }
    for filename, source in expected.items():
        path = out_dir / filename
        if not path.is_file():
            errors.append(f"missing figure: {path}")
            continue
        actual_bytes = path.read_bytes()
        if actual_bytes != source.encode("utf-8"):
            errors.append(f"figure bytes drifted: {path}")
        try:
            actual = actual_bytes.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"figure is not valid UTF-8: {path}")
            continue
        for label in required[filename]:
            if label not in actual:
                errors.append(f"{filename} is missing semantic label: {label}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic World Tubes concept/system SVGs."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--verify-dir",
        type=Path,
        help="Verify deterministic bytes and semantic labels instead of writing.",
    )
    args = parser.parse_args()
    if args.verify_dir is not None:
        errors = verify_figure_dir(args.verify_dir)
        if errors:
            raise SystemExit("figure verification failed:\n- " + "\n- ".join(errors))
        print(f"verified {args.verify_dir}")
        return
    for path in write_figures(args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
