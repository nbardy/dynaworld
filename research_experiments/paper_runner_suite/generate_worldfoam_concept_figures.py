#!/usr/bin/env python3
from __future__ import annotations

"""Generate deterministic, result-free concept figures for the WorldFoam paper.

These figures explain representation and compiler contracts.  They never read
benchmark artifacts and intentionally contain no measured quality, memory, or
timing values.  Exact UTF-8 SVG bytes are the reproducibility contract.
"""

import argparse
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "research_notes" / "worldfoam_paper" / "figures"
FIGURE_FILENAMES = (
    "worldfoam_representation_split.svg",
    "worldfoam_ray_fiber_atlas.svg",
)

REPRESENTATION_REQUIRED_LABELS = (
    "Same learned dynamic world",
    "Known camera program",
    "World Tubes",
    "Depth marginal",
    "Schur complement",
    "WorldFoam",
    "Retain ray depth z",
    "Ordered cell-path word",
    "Translated optical-depth measure",
    "Exact affine transfer",
    "Constant-state ordered reverse",
    "shared compiled world-side work",
    "linear sample / output slice",
    "No measured memory or speed claim in this figure",
)
ATLAS_REQUIRED_LABELS = (
    "Sensor-time base b = (u, v, t)",
    "Ray-depth fiber z",
    "physical ray-length Jacobian",
    "Stable owner word",
    "Order event",
    "split chart",
    "explicit fallback",
    "Translated measure (kappa, nu)",
    "Exact quotient (beta, m)",
    "I = m + beta I_bg",
    "Stream sample bars into node bars",
    "One ordered-word VJP per compiler node",
    "No F x word reverse tape",
    "Event, chart, and rank choices are not differentiated",
)

FORBIDDEN_RESULT_TOKENS = (
    "PSNR",
    "SSIM",
    "LPIPS",
    "dB",
    " MiB",
    " GiB",
    " GB",
    " ms",
    "frames/s",
)

CSS = """
.background { fill: #ffffff; }
.title { fill: #0f172a; font-family: Helvetica, Arial, sans-serif; font-size: 32px; font-weight: 700; }
.subtitle { fill: #475569; font-family: Helvetica, Arial, sans-serif; font-size: 20px; font-weight: 400; }
.lane { fill: #0f172a; font-family: Helvetica, Arial, sans-serif; font-size: 23px; font-weight: 700; }
.node { fill: #0f172a; font-family: Helvetica, Arial, sans-serif; font-size: 20px; font-weight: 700; }
.body { fill: #334155; font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 400; }
.mono { fill: #172554; font-family: Menlo, Consolas, monospace; font-size: 18px; font-weight: 600; }
.badge { fill: #ffffff; font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 700; }
.input { fill: #f8fafc; stroke: #64748b; stroke-width: 2; }
.tubes { fill: #eff6ff; stroke: #2563eb; stroke-width: 2; }
.foam { fill: #ecfdf5; stroke: #059669; stroke-width: 2; }
.measure { fill: #f5f3ff; stroke: #7c3aed; stroke-width: 2; }
.event { fill: #fff7ed; stroke: #ea580c; stroke-width: 2; }
.fallback { fill: #fff1f2; stroke: #e11d48; stroke-width: 2; }
.boundary { fill: #fffbeb; stroke: #d97706; stroke-width: 2; }
.ray { fill: none; stroke: #475569; stroke-width: 3; }
.arrow-blue { fill: none; stroke: #2563eb; stroke-width: 3; }
.arrow-green { fill: none; stroke: #059669; stroke-width: 3; }
.arrow-purple { fill: none; stroke: #7c3aed; stroke-width: 3; }
.dashed { fill: none; stroke: #ea580c; stroke-width: 2.5; stroke-dasharray: 8 6; }
""".strip()

PATH_STYLES = {
    "ray": ("#475569", 3, None),
    "arrow-blue": ("#2563eb", 3, None),
    "arrow-green": ("#059669", 3, None),
    "arrow-purple": ("#7c3aed", 3, None),
    "dashed": ("#ea580c", 2.5, "8 6"),
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

    def circle(self, x: float, y: float, radius: float, *, fill: str) -> None:
        self.elements.append(
            f'<circle cx="{x:g}" cy="{y:g}" r="{radius:g}" fill="{fill}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        css_class: str,
    ) -> None:
        color, width, dash = PATH_STYLES[css_class]
        if dash is None:
            self._filled_segment((x1, y1), (x2, y2), width=width, color=color)
            self._arrowhead((x1, y1), (x2, y2), color=color)
        else:
            self._dashed_segment((x1, y1), (x2, y2), width=width, color=color)

    def polyline(
        self,
        points: tuple[tuple[float, float], ...],
        *,
        css_class: str,
    ) -> None:
        color, width, dash = PATH_STYLES[css_class]
        for start, end in zip(points, points[1:]):
            if dash is None:
                self._filled_segment(start, end, width=width, color=color)
            else:
                self._dashed_segment(start, end, width=width, color=color)
        if dash is None:
            self._arrowhead(points[-2], points[-1], color=color)

    def _filled_segment(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        width: float,
        color: str,
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = hypot(dx, dy)
        if length == 0:
            return
        px, py = -dy / length * width / 2, dx / length * width / 2
        points = (
            (start[0] + px, start[1] + py),
            (end[0] + px, end[1] + py),
            (end[0] - px, end[1] - py),
            (start[0] - px, start[1] - py),
        )
        encoded = " ".join(f"{x:g},{y:g}" for x, y in points)
        self.elements.append(f'<polygon points="{encoded}" fill="{color}"/>')

    def _dashed_segment(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        width: float,
        color: str,
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        cursor = 0.0
        while cursor < length:
            stop = min(cursor + 8.0, length)
            dash_start = (start[0] + cursor * ux, start[1] + cursor * uy)
            dash_end = (start[0] + stop * ux, start[1] + stop * uy)
            self._filled_segment(dash_start, dash_end, width=width, color=color)
            cursor += 14.0

    def _arrowhead(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str,
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = hypot(dx, dy)
        if length == 0:
            raise ValueError("arrowhead requires a non-zero final segment")
        ux, uy = dx / length, dy / length
        base_x, base_y = end[0] - 11 * ux, end[1] - 11 * uy
        perp_x, perp_y = -uy, ux
        points = (
            end,
            (base_x + 5 * perp_x, base_y + 5 * perp_y),
            (base_x - 5 * perp_x, base_y - 5 * perp_y),
        )
        encoded = " ".join(f"{x:g},{y:g}" for x, y in points)
        self.elements.append(f'<polygon points="{encoded}" fill="{color}"/>')

    def text(
        self,
        x: float,
        y: float,
        lines: tuple[str, ...],
        *,
        css_class: str = "body",
        anchor: str = "start",
        line_height: int = 25,
    ) -> None:
        spans = []
        for index, line in enumerate(lines):
            dy = 0 if index == 0 else line_height
            spans.append(f'<tspan x="{x:g}" dy="{dy:g}">{escape(line)}</tspan>')
        self.elements.append(
            f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" '
            f'class="{css_class}">{"".join(spans)}</text>'
        )

    def badge(self, x: float, y: float, label: str, *, fill: str) -> None:
        width = max(105, len(label) * 10 + 28)
        self.elements.append(
            f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="34" '
            f'rx="17" fill="{fill}"/>'
        )
        self.text(
            x + width / 2,
            y + 24,
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


def representation_split_svg() -> str:
    svg = Svg(
        1200,
        760,
        "World Tubes and WorldFoam retain different ray information",
        (
            "The same dynamic world and camera program branch into a "
            "depth-marginalized Gaussian trace representation or a retained-depth "
            "ordered-transfer representation. The figure states structural, not "
            "measured, memory and work boundaries."
        ),
    )
    svg.rect(0, 0, 1200, 760, css_class="background", rx=0)
    svg.text(
        42,
        48,
        ("One camera-ray bundle, two representation choices",),
        css_class="title",
    )
    svg.text(
        42,
        80,
        ("World Tubes marginalizes depth; WorldFoam retains ordered depth transfer.",),
        css_class="subtitle",
    )

    svg.rect(50, 112, 1100, 100, css_class="input", rx=16)
    svg.text(
        600,
        145,
        ("Same learned dynamic world", "Known camera program  Γ : (u, v, t, z) -> (x, t)"),
        css_class="node",
        anchor="middle",
        line_height=30,
    )
    svg.polyline(((600, 212), (600, 238), (305, 238), (305, 265)), css_class="arrow-blue")
    svg.polyline(((600, 212), (600, 238), (895, 238), (895, 265)), css_class="arrow-green")

    svg.rect(50, 275, 510, 350, css_class="tubes", rx=18)
    svg.badge(76, 294, "WORLD TUBES", fill="#2563eb")
    svg.text(305, 361, ("Gaussian closure under marginalization",), css_class="lane", anchor="middle")
    svg.rect(88, 390, 434, 82, css_class="input")
    svg.text(
        305,
        420,
        ("Depth marginal via Schur complement", "UVT footprint + conditional-depth packet"),
        css_class="body",
        anchor="middle",
        line_height=27,
    )
    svg.rect(88, 493, 434, 82, css_class="input")
    svg.text(
        305,
        523,
        ("Certified support and representative order", "splat-compatible trace evaluator"),
        css_class="body",
        anchor="middle",
        line_height=27,
    )
    svg.text(
        305,
        588,
        ("Best when depth can be summarized", "without changing the required transfer law."),
        css_class="body",
        anchor="middle",
        line_height=25,
    )

    svg.rect(640, 275, 510, 350, css_class="foam", rx=18)
    svg.badge(666, 294, "WORLDFOAM", fill="#059669")
    svg.text(895, 361, ("Retain ray depth z",), css_class="lane", anchor="middle")
    svg.rect(678, 390, 434, 82, css_class="input")
    svg.text(
        895,
        420,
        ("Ordered cell-path word", "physical intervals  (tau_r, c_r)"),
        css_class="body",
        anchor="middle",
        line_height=27,
    )
    svg.rect(678, 493, 434, 82, css_class="measure")
    svg.text(
        895,
        523,
        ("Translated optical-depth measure  (kappa, nu)", "Exact affine transfer  (beta, m)"),
        css_class="body",
        anchor="middle",
        line_height=27,
    )
    svg.text(
        895,
        588,
        ("Constant-state ordered reverse;", "preserve owner words for world gradients."),
        css_class="body",
        anchor="middle",
        line_height=25,
    )

    svg.rect(50, 647, 1100, 50, css_class="boundary", rx=12)
    svg.text(
        600,
        678,
        ("Target contract: shared compiled world-side work + linear sample / output slice",),
        css_class="node",
        anchor="middle",
    )
    svg.text(
        600,
        731,
        ("No measured memory or speed claim in this figure",),
        css_class="subtitle",
        anchor="middle",
    )
    return svg.render()


def ray_fiber_atlas_svg() -> str:
    svg = Svg(
        1200,
        850,
        "WorldFoam ray-fiber atlas and shared ordered adjoint",
        (
            "A camera gauge maps sensor-time samples to ordered ray-depth fibers. "
            "Stable owner words compile into optical-depth measures and exact affine "
            "transfers; events split charts or trigger explicit fallback. Sample "
            "cotangents reduce to compiler nodes before one ordered-word reverse."
        ),
    )
    svg.rect(0, 0, 1200, 850, css_class="background", rx=0)
    svg.text(42, 48, ("WorldFoam: compile ordered transfer on stable ray-fiber strata",), css_class="title")
    svg.text(
        42,
        80,
        ("Gauge covariance preserves physical ray measure; event boundaries preserve order semantics.",),
        css_class="subtitle",
    )

    svg.rect(42, 110, 470, 470, css_class="input", rx=18)
    svg.text(277, 145, ("Sensor-time base b = (u, v, t)",), css_class="lane", anchor="middle")
    svg.text(277, 177, ("each sample lifts to one Ray-depth fiber z",), css_class="body", anchor="middle")

    ray_ys = (250, 345, 440)
    labels = ("t < t*", "t = t*", "t > t*")
    for y, label in zip(ray_ys, labels, strict=True):
        svg.text(73, y + 6, (label,), css_class="body")
        svg.line(158, y, 475, y, css_class="ray")
    svg.text(459, 226, ("z",), css_class="mono")

    # Differently colored intervals exchange order at the middle event row.
    for x, y, width, fill in (
        (205, 232, 92, "#fecaca"),
        (330, 232, 92, "#bfdbfe"),
        (264, 327, 78, "#fed7aa"),
        (205, 422, 92, "#bfdbfe"),
        (330, 422, 92, "#fecaca"),
    ):
        svg.elements.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="36" rx="8" fill="{fill}" stroke="#475569" stroke-width="1.5"/>'
        )
    svg.line(303, 205, 303, 478, css_class="dashed")
    svg.text(303, 510, ("Order event", "split chart at t*"), css_class="node", anchor="middle", line_height=26)
    svg.text(
        277,
        558,
        ("lambda = rho · physical ray-length Jacobian",),
        css_class="mono",
        anchor="middle",
    )

    svg.rect(548, 110, 610, 470, css_class="foam", rx=18)
    svg.badge(572, 128, "COMPILED ATLAS", fill="#059669")
    nodes = (
        (574, 190, 250, 76, "Stable owner word", "front-to-back cell intervals"),
        (882, 190, 250, 76, "Translated measure (kappa, nu)", "order-explicit proof object"),
        (574, 302, 250, 76, "Exact quotient (beta, m)", "rear-radiance action"),
        (882, 302, 250, 76, "Decode sensor sample", "I = m + beta I_bg"),
    )
    for x, y, width, height, title, body in nodes:
        svg.rect(x, y, width, height, css_class="measure" if "measure" in title or "quotient" in title else "input", rx=12)
        svg.text(x + width / 2, y + 29, (title,), css_class="node", anchor="middle")
        svg.text(x + width / 2, y + 56, (body,), css_class="body", anchor="middle")
    svg.line(824, 228, 872, 228, css_class="arrow-green")
    svg.polyline(((1007, 266), (1007, 282), (699, 282), (699, 292)), css_class="arrow-purple")
    svg.line(824, 340, 872, 340, css_class="arrow-green")

    svg.rect(574, 420, 250, 104, css_class="event", rx=12)
    svg.text(699, 450, ("Certificate outcome",), css_class="node", anchor="middle")
    svg.text(699, 480, ("stable → compile", "event → split chart"), css_class="body", anchor="middle", line_height=27)
    svg.rect(882, 420, 250, 104, css_class="fallback", rx=12)
    svg.text(1007, 450, ("Unresolved domain",), css_class="node", anchor="middle")
    svg.text(1007, 480, ("explicit fallback", "never extrapolate order"), css_class="body", anchor="middle", line_height=27)

    svg.rect(42, 610, 1116, 142, css_class="measure", rx=18)
    svg.badge(66, 628, "SHARED ADJOINT", fill="#7c3aed")
    svg.text(
        600,
        686,
        ("Stream sample bars into node bars",),
        css_class="lane",
        anchor="middle",
    )
    svg.line(345, 714, 470, 714, css_class="arrow-purple")
    svg.text(260, 720, ("sample cotangents",), css_class="body", anchor="middle")
    svg.text(600, 720, ("compiler-node cotangents",), css_class="body", anchor="middle")
    svg.line(730, 714, 850, 714, css_class="arrow-purple")
    svg.text(
        972,
        714,
        ("One ordered-word VJP per compiler node",),
        css_class="body",
        anchor="middle",
    )
    svg.text(600, 744, ("No F x word reverse tape",), css_class="mono", anchor="middle")

    svg.rect(42, 782, 1116, 45, css_class="boundary", rx=11)
    svg.text(
        600,
        811,
        ("Claim boundary: Event, chart, and rank choices are not differentiated",),
        css_class="node",
        anchor="middle",
    )
    return svg.render()


def expected_figures() -> dict[str, str]:
    return {
        FIGURE_FILENAMES[0]: representation_split_svg(),
        FIGURE_FILENAMES[1]: ray_fiber_atlas_svg(),
    }


def write_figures(out_dir: Path = DEFAULT_OUT_DIR) -> tuple[Path, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, source in expected_figures().items():
        path = out_dir / filename
        path.write_text(source, encoding="utf-8", newline="\n")
        paths.append(path)
    return tuple(paths)


def verify_figure_dir(out_dir: Path = DEFAULT_OUT_DIR) -> list[str]:
    errors: list[str] = []
    expected = expected_figures()
    required = {
        FIGURE_FILENAMES[0]: REPRESENTATION_REQUIRED_LABELS,
        FIGURE_FILENAMES[1]: ATLAS_REQUIRED_LABELS,
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
        for token in FORBIDDEN_RESULT_TOKENS:
            if token in actual:
                errors.append(f"{filename} contains measured-result token: {token}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic, result-free WorldFoam concept SVGs."
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
