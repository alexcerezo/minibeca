"""
manim_cover.py — Portada «prisma» en estilo Manim (5:2, 3000×1200), sin título.

Un haz blanco con lo que activa de verdad a la neurona 2787 de GPT-2 entra en un
prisma (el SAE) y sale separado en conceptos: las latentes reales del diccionario.
Conceptos, colores y fragmentos se leen de src/viz/cover.py (misma fuente que las otras portadas).

Render (→ assets/cover/cover_B_prisma_manim.png):  make covers
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    BLACK,
    DOWN,
    GRAY,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    Dot,
    Line,
    MarkupText,
    MathTex,
    ManimColor,
    Polygon,
    Scene,
    Tex,
    VGroup,
    config,
    interpolate_color,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cover import CONCEPTS, neuron_fragments  # noqa: E402

config.pixel_width, config.pixel_height = 3000, 1200
config.frame_height = 8.0
config.frame_width = 20.0
config.background_color = BLACK

MONO = "DejaVu Sans Mono"


def glow_line(a, b, color, core: float, layers=((14, 0.05), (7, 0.10), (3, 0.25))) -> VGroup:
    """Línea con halo: varias copias más gruesas y translúcidas debajo del trazo principal."""
    g = VGroup(*(Line(a, b, stroke_color=color, stroke_width=core * k, stroke_opacity=o) for k, o in layers))
    g.add(Line(a, b, stroke_color=color, stroke_width=core))
    return g


def on_segment(p, q, t: float) -> np.ndarray:
    return p + (q - p) * t


class CoverPrismScene(Scene):
    def construct(self) -> None:
        # --- prisma equilátero -------------------------------------------------
        side, c = 3.9, np.array([0.2, -0.35, 0.0])
        h = side * np.sqrt(3) / 2
        top = c + UP * (2 * h / 3)
        bl = c + DOWN * (h / 3) + LEFT * side / 2
        br = c + DOWN * (h / 3) + RIGHT * side / 2
        prism = Polygon(top, bl, br, stroke_color=WHITE, stroke_width=3)
        prism.set_fill(WHITE, opacity=0.06)
        prism_glow = prism.copy().set_fill(opacity=0).set_stroke(WHITE, width=14, opacity=0.08)

        entry = on_segment(bl, top, 0.46)   # cara izquierda
        exit_ = on_segment(br, top, 0.56)   # cara derecha

        # --- haz de entrada ----------------------------------------------------
        start = np.array([-10.6, -3.35, 0.0])
        beam = glow_line(start, entry, WHITE, 4.5)
        inner = Line(entry, exit_, stroke_color=WHITE, stroke_width=4, stroke_opacity=0.6)

        # lo que activa a la neurona 2787: una preposición, la coma de los miles y la sílaba de un apellido
        d = entry - start
        ang = np.arctan2(d[1], d[0])
        normal = np.array([-np.sin(ang), np.cos(ang), 0.0])
        frags = neuron_fragments()
        chips = VGroup()
        for i, (fi, t, side_off) in enumerate(((0, 0.22, 0.34), (2, 0.45, -0.40), (4, 0.70, 0.34))):
            a, b, rest = frags[fi]
            m = MarkupText(
                f"<span foreground='#8A8F98'>{a}</span><b>{b}</b><span foreground='#8A8F98'>{rest}</span>",
                font=MONO, font_size=20, color=WHITE,
            )
            m.rotate(ang).move_to(on_segment(start, entry, t) + normal * side_off)
            chips.add(m)
        neuron_lbl = Tex(r"neurona 2787", color=GRAY).scale(0.55)
        neuron_lbl.rotate(ang).move_to(on_segment(start, entry, 0.20) - normal * 0.42)

        # --- rayos de salida: cada uno una latente real -------------------------
        rays = CONCEPTS[:7]
        x_end = 6.9
        ys = np.linspace(3.2, -3.2, len(rays))
        fans, labels, ends = VGroup(), VGroup(), VGroup()
        for (lid, lab, col), y in zip(rays, ys):
            tip = np.array([x_end, y, 0.0])
            wedge = Polygon(
                exit_ + UP * 0.02, tip + UP * 0.07, tip + DOWN * 0.07, exit_ + DOWN * 0.02,
                stroke_width=0,
            )
            wedge.set_fill([col, interpolate_color(ManimColor(col), BLACK, 0.55)], opacity=0.95)
            wedge.set_sheen_direction(tip - exit_)
            halo = Line(exit_, tip, stroke_color=col, stroke_width=18, stroke_opacity=0.07)
            fans.add(halo, wedge)
            ends.add(Dot(tip, radius=0.05, color=col))
            name = Tex(lab, color=col).scale(0.72)
            idx = Tex(rf"\#{lid}", color=GRAY).scale(0.42)
            row = VGroup(name, idx).arrange(RIGHT, buff=0.18)
            idx.align_to(name[0][0], DOWN)  # misma línea base que la primera letra (sin descendente)
            row.next_to(tip, RIGHT, buff=0.25)
            labels.add(row)

        # --- la fórmula del SAE, discreta bajo el prisma -----------------------
        formula = MathTex(r"\mathbf{x} \;\approx\; \sum_{i=1}^{24\,576} z_i\, \mathbf{d}_i", color=GRAY)
        formula.scale(0.62).next_to(prism, DOWN, buff=0.45)
        sparsity = MathTex(r"\|z\|_0 = 32", color=GRAY).scale(0.5).next_to(formula, DOWN, buff=0.18)

        self.add(beam, chips, neuron_lbl, fans, prism_glow, prism, inner, ends, labels, formula, sparsity)
