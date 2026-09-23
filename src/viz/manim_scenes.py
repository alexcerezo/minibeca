"""
manim_scenes.py — Escenas del artículo (Minibecas XMihura).

Todas las cifras se leen en tiempo de render de ficheros generados por el
pipeline; ninguna está escrita a mano:
  * data/manim_data.json        resumen compacto (métricas held-out, curvas de
                                entrenamiento con y sin AuxK, evaluación ciega,
                                steering)
  * data/analysis/*.json        salidas de src/analyze_sae.py

Render (1080p60 → assets/videos/):  make videos
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from manim import (
    BLUE,
    BLUE_A,
    DARK_GRAY,
    DOWN,
    GRAY,
    GREEN,
    LEFT,
    LIGHT_GRAY,
    ORIGIN,
    PI,
    PURPLE,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Axes,
    Create,
    DashedLine,
    DashedVMobject,
    Dot,
    FadeIn,
    FadeOut,
    FadeTransform,
    GrowArrow,
    GrowFromEdge,
    Integer,
    Line,
    MarkupText,
    MathTex,
    Matrix,
    NumberPlane,
    Rectangle,
    RoundedRectangle,
    Scene,
    Tex,
    Transform,
    ValueTracker,
    VGroup,
    VMobject,
    Write,
    linear,
    smooth,
)

# ---------------------------------------------------------------------------
# Paleta y datos
# ---------------------------------------------------------------------------

C_ACCENT = YELLOW
C_GOOD = "#63D471"
C_BAD = "#F26B4E"
C_MLP = "#6C8EBF"
C_RAND = "#8A8F98"
C_BLOOM = "#B57EDC"
C_STEER = ["#F2C14E", "#F26B4E", "#4EA3F2"]
BG_PANEL = "#0E1420"
MONO = "DejaVu Sans Mono"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"


def _data() -> dict:
    return json.loads((DATA_DIR / "manim_data.json").read_text())


def _es(x: float, nd: int) -> str:
    """Número LaTeX con coma decimal y espacio fino de miles (convención española)."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "@").replace(".", "{,}").replace("@", r"\,")


def _line(text: str, scale: float, color=WHITE, max_w: float = 13.2) -> Tex:
    """Tex en una sola línea (Tex envuelve en `center` con ancho fijo)."""
    t = Tex(rf"\mbox{{{text}}}", color=color).scale(scale)
    return t.scale_to_fit_width(max_w) if t.width > max_w else t


def _title(text: str, subtitle: str | None = None) -> VGroup:
    t = Tex(text).scale(0.75)
    grp = VGroup(t)
    if subtitle:
        grp.add(_line(subtitle, 0.45, LIGHT_GRAY).next_to(t, DOWN, buff=0.12))
    return grp.to_edge(UP, buff=0.25)


def _make_axes(x_range, y_range, length_x=8.5, length_y=4.0, y_dec=2, x_dec=0):
    return Axes(
        x_range=list(x_range),
        y_range=list(y_range),
        x_length=length_x,
        y_length=length_y,
        tips=False,
        axis_config={"stroke_opacity": 0.55, "stroke_width": 2, "include_ticks": True},
        x_axis_config={
            "include_numbers": True, "font_size": 18,
            "decimal_number_config": {"num_decimal_places": x_dec, "group_with_commas": False},
        },
        y_axis_config={
            "include_numbers": True, "font_size": 18,
            "decimal_number_config": {"num_decimal_places": y_dec, "group_with_commas": False},
        },
    )


def _polyline(ax: Axes, xs, ys, color, width: float = 4.0) -> VMobject:
    return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(
        [ax.c2p(float(x), float(y)) for x, y in zip(xs, ys)]
    )


def _legend(items: list[tuple[str, str, bool]], scale: float = 0.45) -> VGroup:
    rows = VGroup()
    for label, color, dashed in items:
        seg = Line(ORIGIN, RIGHT * 0.45, color=color, stroke_width=5)
        if dashed:
            seg = DashedLine(ORIGIN, RIGHT * 0.45, color=color, stroke_width=4, dash_length=0.08)
        rows.add(VGroup(seg, Tex(label).scale(scale)).arrange(RIGHT, buff=0.15))
    return rows.arrange(DOWN, aligned_edge=LEFT, buff=0.14)


def _panel(mob, pad: float = 0.25) -> RoundedRectangle:
    return RoundedRectangle(
        width=mob.width + 2 * pad, height=mob.height + 2 * pad, corner_radius=0.12,
        stroke_width=1, stroke_color=DARK_GRAY, fill_color=BG_PANEL, fill_opacity=0.85,
    ).move_to(mob)


# ---------------------------------------------------------------------------
# Escena 1 — Superposición
# ---------------------------------------------------------------------------


class SuperpositionScene(Scene):
    def construct(self) -> None:
        title = _title(r"Superposición", r"$n$ features en $d$ dimensiones, con $n \gg d$")
        self.play(Write(title))

        plane = NumberPlane(
            x_range=(-1.6, 1.6, 0.5),
            y_range=(-1.6, 1.6, 0.5),
            background_line_style={"stroke_opacity": 0.25, "stroke_width": 1},
            axis_config={"stroke_opacity": 0.55},
        ).scale(1.25).shift(LEFT * 3.2 + DOWN * 0.3)
        self.play(Create(plane), run_time=0.6)

        e1 = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color=BLUE, stroke_width=6)
        e2 = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color=GREEN, stroke_width=6)
        e1_lbl = MathTex(r"f_1", color=BLUE).scale(0.75).next_to(e1, DOWN, buff=0.12)
        e2_lbl = MathTex(r"f_2", color=GREEN).scale(0.75).next_to(e2, LEFT, buff=0.12)
        legend_a = VGroup(
            MathTex(r"d = 2 \text{ dims}"),
            MathTex(r"n = 2 \text{ features}"),
            MathTex(r"\langle f_1,\, f_2\rangle = 0"),
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.7).to_edge(RIGHT, buff=0.6).shift(DOWN * 0.5)
        self.play(GrowArrow(e1), GrowArrow(e2), FadeIn(e1_lbl, e2_lbl))
        self.play(Write(legend_a))
        self.wait(0.9)

        n = 5
        colors = [BLUE, GREEN, YELLOW, RED, PURPLE]
        vecs = [np.array([math.cos(2 * PI * i / n), math.sin(2 * PI * i / n)]) for i in range(n)]
        arrows = [
            Arrow(plane.c2p(0, 0), plane.c2p(*v), buff=0, color=c, stroke_width=6)
            for v, c in zip(vecs, colors)
        ]
        labels = [
            MathTex(rf"f_{{{i+1}}}", color=colors[i]).scale(0.55).next_to(
                arrows[i].get_end(), np.array([vecs[i][0], vecs[i][1], 0.0]) * 0.4, buff=0.03,
            )
            for i in range(n)
        ]
        legend_b = VGroup(
            MathTex(r"d = 2,\ n = 5"),
            MathTex(r"|\langle f_i, f_j\rangle| \le |\cos 72^\circ| \approx 0{,}31"),
            MathTex(r"\text{Interferencia tolerable si pocas se activan a la vez}"),
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.55).to_edge(RIGHT, buff=0.45).shift(DOWN * 0.5)
        self.play(
            Transform(e1, arrows[0]),
            Transform(e2, arrows[1]),
            *(GrowArrow(a) for a in arrows[2:]),
            FadeOut(e1_lbl, e2_lbl),
            *(FadeIn(lbl) for lbl in labels),
            FadeTransform(legend_a, legend_b),
        )
        self.remove(e1, e2)
        self.add(arrows[0], arrows[1])
        self.wait(0.8)

        gram = np.round(np.array([[v @ w for w in vecs] for v in vecs]), 2)
        gram_data = [[f"{gram[i, j]:+.2f}".replace(".", ",") for j in range(n)] for i in range(n)]
        gram_matrix = Matrix(gram_data, h_buff=1.8, v_buff=0.7).scale(0.30)
        gram_matrix.to_edge(RIGHT, buff=0.4).shift(UP * 1.2)
        gram_lbl = MathTex(r"G_{ij} = \langle f_i, f_j\rangle", color=LIGHT_GRAY)\
            .scale(0.5).next_to(gram_matrix, UP, buff=0.15)
        self.play(FadeIn(gram_matrix), Write(gram_lbl))
        self.wait(1.4)

        active_idx = 2
        highlight = arrows[active_idx].copy().set_color(C_ACCENT).set_stroke(width=10)
        note = Tex(
            r"Si sólo $f_3$ está activa, las demás apenas interfieren con ella.",
        ).scale(0.55).to_edge(DOWN, buff=0.4)
        self.play(
            *(a.animate.set_opacity(0.15) for i, a in enumerate(arrows) if i != active_idx),
            *(lbl.animate.set_opacity(0.4) for i, lbl in enumerate(labels) if i != active_idx),
            FadeIn(highlight),
            Write(note),
        )
        self.wait(1.4)

        dense_n = 12
        dense_arrows = VGroup(*[
            Arrow(plane.c2p(0, 0), plane.c2p(math.cos(2 * PI * i / dense_n), math.sin(2 * PI * i / dense_n)),
                  buff=0, color=BLUE_A, stroke_width=3)
            for i in range(dense_n)
        ])
        conclusion = Tex(
            r"GPT-2 small: $768$ dimensiones. El SAE busca $24\,576$ direcciones.",
        ).scale(0.55).to_edge(DOWN, buff=0.4)
        self.play(
            *(FadeOut(a) for a in arrows), FadeOut(highlight),
            *(FadeOut(lbl) for lbl in labels),
            FadeOut(gram_matrix, gram_lbl),
            Create(dense_arrows, lag_ratio=0.08),
            FadeOut(note, shift=DOWN * 0.2),
            FadeIn(conclusion, shift=UP * 0.2),
        )
        self.wait(2.5)
        self.play(*(FadeOut(o) for o in self.mobjects))


# ---------------------------------------------------------------------------
# Escena 2 — Anatomía del SAE
# ---------------------------------------------------------------------------


class SAEAnatomyScene(Scene):
    def construct(self) -> None:
        title = Tex(r"Anatomía del Sparse Autoencoder").scale(0.7).to_edge(UP, buff=0.25)
        self.play(Write(title))

        d, m, k = 8, 24, 3

        def make_column(n: int, x: float, cell: float, color, opacity=0.55) -> VGroup:
            g = VGroup()
            for i in range(n):
                y = ((n - 1) / 2 - i) * cell * 1.05
                sq = Rectangle(width=cell, height=cell, fill_color=color, fill_opacity=opacity,
                               stroke_color=WHITE, stroke_width=1.2)
                g.add(sq.move_to(np.array([x, y, 0])))
            return g

        col_shift = UP * 0.2
        x_col = make_column(d, x=-4.7, cell=0.45, color=BLUE).shift(col_shift)
        z_col = make_column(m, x=0.0, cell=0.23, color=DARK_GRAY, opacity=0.3).shift(col_shift)
        r_col = make_column(d, x=4.7, cell=0.45, color=BLUE, opacity=0.0).shift(col_shift)

        x_lbl = MathTex(r"x \in \mathbb{R}^{768}", color=BLUE).scale(0.6).next_to(x_col, DOWN, buff=0.25)
        z_lbl = MathTex(r"z \in \mathbb{R}^{24576},\ \|z\|_0 = 32", color=WHITE)\
            .scale(0.5).next_to(z_col, DOWN, buff=0.15)
        r_lbl = MathTex(r"\hat x \in \mathbb{R}^{768}", color=BLUE).scale(0.6).next_to(r_col, DOWN, buff=0.25)

        self.play(FadeIn(x_col), Write(x_lbl))
        self.wait(0.3)

        enc_arrow = Arrow(x_col.get_right() + RIGHT * 0.15, z_col.get_left() + LEFT * 0.15,
                          buff=0, color=C_ACCENT, stroke_width=4)
        enc_lbl = MathTex(r"\mathrm{TopK}_{k=32}(W_{\text{enc}}\,x + b_{\text{enc}})").scale(0.5)\
            .next_to(enc_arrow, UP, buff=0.12)
        self.play(GrowArrow(enc_arrow), Write(enc_lbl))

        rng = np.random.default_rng(11)
        pre = rng.random(m)
        self.play(
            *[sq.animate.set_fill(BLUE_A, opacity=0.15 + 0.65 * pre[i]) for i, sq in enumerate(z_col)],
            Write(z_lbl),
        )
        self.wait(0.6)

        topk_idx = set(np.argsort(-pre)[:k].tolist())
        self.play(*[
            sq.animate.set_fill(C_ACCENT, opacity=0.95).set_stroke(width=2)
            if i in topk_idx else
            sq.animate.set_fill(DARK_GRAY, opacity=0.15).set_stroke(width=0.5)
            for i, sq in enumerate(z_col)
        ])

        dec_arrow = Arrow(z_col.get_right() + RIGHT * 0.15, r_col.get_left() + LEFT * 0.15,
                          buff=0, color=C_ACCENT, stroke_width=4)
        dec_lbl = MathTex(r"W_{\text{dec}}\,z + b_{\text{dec}}").scale(0.55).next_to(dec_arrow, UP, buff=0.12)
        self.play(GrowArrow(dec_arrow), Write(dec_lbl))
        self.play(*[sq.animate.set_fill(BLUE, opacity=0.55) for sq in r_col], Write(r_lbl))
        self.wait(0.4)

        loss = MathTex(
            r"\mathcal{L}\ =\ \|x - \hat x\|_2^2\ +\ \tfrac{1}{32}\,\mathcal{L}_{\text{AuxK}}"
            r"\quad\text{{\scriptsize (reactiva latentes muertas)}}"
        ).scale(0.6).to_edge(DOWN, buff=0.4)
        self.play(Write(loss))
        self.wait(2.4)
        self.play(*(FadeOut(o) for o in self.mobjects))


# ---------------------------------------------------------------------------
# Escena 3 — Ablación de AuxK (dos entrenamientos completos)
# ---------------------------------------------------------------------------


class AuxKAblationScene(Scene):
    def construct(self) -> None:
        D = _data()
        ev = D["dead_events"]
        dead_a = np.array(D["dead_curve"]["auxk"])
        dead_n = np.array(D["dead_curve"]["noauxk"])
        ec = D["eval_curve"]
        live = D["liveness_heldout"]
        h128 = D["heldout_ctx128"]

        title = _title(
            r"Hallazgo 1: sin AuxK, el diccionario muere",
            r"Dos entrenamientos idénticos de $100$ M tokens. Sólo cambia el término AuxK.",
        )
        self.play(Write(title))

        ax_l = _make_axes((0, 12500, 2500), (0, 25, 5), length_x=5.4, length_y=3.6, y_dec=0)\
            .shift(LEFT * 3.45 + DOWN * 0.35)
        ax_r = _make_axes((0, 12500, 2500), (0.75, 1.0, 0.05), length_x=5.4, length_y=3.6, y_dec=2)\
            .shift(RIGHT * 3.45 + DOWN * 0.35)
        lbl_l = Tex(r"latentes muertas ($\times 10^3$, de $24\,576$)").scale(0.48).next_to(ax_l, UP, buff=0.18)
        lbl_r = Tex(r"CE loss score (1 = modelo intacto)").scale(0.48).next_to(ax_r, UP, buff=0.18)
        xl_l = Tex(r"paso", color=LIGHT_GRAY).scale(0.45).next_to(ax_l.x_axis, DOWN, buff=0.35)
        xl_r = Tex(r"paso", color=LIGHT_GRAY).scale(0.45).next_to(ax_r.x_axis, DOWN, buff=0.35)
        legend = _legend([(r"con AuxK", C_GOOD, False), (r"sin AuxK", C_BAD, False)], scale=0.45)
        legend.move_to(ax_l.c2p(8200, 13.5))
        legend = VGroup(_panel(legend, pad=0.12), legend)
        self.play(Create(ax_l), Create(ax_r), Write(lbl_l), Write(lbl_r), FadeIn(xl_l, xl_r), FadeIn(legend))

        curve_a = _polyline(ax_l, dead_a[:, 0], dead_a[:, 1] / 1000, C_GOOD)
        curve_n = _polyline(ax_l, dead_n[:, 0], dead_n[:, 1] / 1000, C_BAD)
        ce_a = _polyline(ax_r, ec["auxk"]["step"], ec["auxk"]["ce"], C_GOOD)
        ce_n = _polyline(ax_r, ec["noauxk"]["step"], ec["noauxk"]["ce"], C_BAD)

        # contadores en vivo sincronizados con el trazado (puntos equiespaciados en paso)
        prog = ValueTracker(0.0)
        n_pts = len(dead_a)

        def _idx() -> int:
            return min(n_pts - 1, int(prog.get_value() * (n_pts - 1)))

        # Cada número se maqueta con su ancho máximo (5 cifras) y luego se pone a 0: Integer crece
        # desde su borde izquierdo, así el valor nunca invade la etiqueta siguiente.
        cnt_a, cnt_n, stepc = (
            Integer(88888, group_with_commas=False, color=c).scale(0.55) for c in (C_GOOD, C_BAD, WHITE)
        )
        pairs = []
        for text, num in ((r"paso", stepc), (r"muertas con AuxK", cnt_a), (r"sin AuxK", cnt_n)):
            lbl = Tex(text).scale(0.5)
            num.next_to(lbl, RIGHT, buff=0.3)
            num.align_to(lbl[0][1], DOWN)  # 2.º glifo sin descendente: misma línea base
            pairs.append(VGroup(lbl, num))
        counter = VGroup(*pairs).arrange(RIGHT, buff=0.55).to_edge(DOWN, buff=0.3)
        counter_bg = _panel(counter, pad=0.15)
        for c in (cnt_a, cnt_n, stepc):
            c.set_value(0)
        cnt_a.add_updater(lambda m: m.set_value(int(dead_a[_idx(), 1])))
        cnt_n.add_updater(lambda m: m.set_value(int(dead_n[_idx(), 1])))
        stepc.add_updater(lambda m: m.set_value(int(dead_a[_idx(), 0])))
        self.add(counter_bg, counter)

        self.play(
            Create(curve_a, rate_func=linear), Create(curve_n, rate_func=linear),
            Create(ce_a, rate_func=linear), Create(ce_n, rate_func=linear),
            prog.animate(rate_func=linear).set_value(1.0),
            run_time=7.0,
        )
        for c in (cnt_a, cnt_n, stepc):
            c.clear_updaters()
        self.wait(0.5)

        pk_step, pk_val = ev["auxk_peak"]
        pk = ax_l.c2p(pk_step, pk_val / 1000)
        ann_peak = VGroup(
            Dot(pk, color=YELLOW, radius=0.07),
            Tex(rf"paso {pk_step}: ${_es(pk_val, 0)}$ muertas ($89\,\%$)", color=YELLOW)
            .scale(0.4).next_to(ax_l.c2p(1600, 18.5), RIGHT, buff=0.0),
        )
        end_a = ax_l.c2p(dead_a[-1, 0], dead_a[-1, 1] / 1000)
        end_n = ax_l.c2p(dead_n[-1, 0], dead_n[-1, 1] / 1000)
        ann_end = VGroup(
            Tex(rf"final: {ev['auxk_final']}", color=C_GOOD).scale(0.42)
            .next_to(end_a, UP, buff=0.18).align_to(end_a, RIGHT),
            Tex(rf"final: ${_es(ev['noauxk_final'], 0)}$", color=C_BAD).scale(0.42)
            .next_to(end_n, UP, buff=0.12).align_to(end_n, RIGHT),
        )
        fa = ax_r.c2p(ec["auxk"]["step"][-1], ec["auxk"]["ce"][-1])
        fn = ax_r.c2p(ec["noauxk"]["step"][-1], ec["noauxk"]["ce"][-1])
        ann_ce = VGroup(
            Tex(f"${_es(ec['auxk']['ce'][-1], 3)}$", color=C_GOOD).scale(0.42).next_to(fa, UP, buff=0.1),
            Tex(f"${_es(ec['noauxk']['ce'][-1], 3)}$", color=C_BAD).scale(0.42).next_to(fn, DOWN, buff=0.1),
        )
        self.play(FadeIn(ann_peak), FadeIn(ann_end), FadeIn(ann_ce))
        self.wait(2.0)

        alive_a = 100 * live["ours"]["alive_frac"]
        alive_n = 100 * live["noauxk"]["alive_frac"]
        summary = _line(
            rf"En texto nunca visto: latentes que llegan a usarse ${_es(alive_a, 1)}\,\%$ frente a "
            rf"${_es(alive_n, 1)}\,\%$. Pérdida extra ${_es(h128['ours_topk_24k']['ce_delta'], 2)}$ "
            rf"frente a ${_es(h128['ours_noauxk']['ce_delta'], 2)}$ nats.",
            0.46,
        )
        summary.move_to(counter)
        self.play(FadeOut(counter), FadeOut(counter_bg))
        sbg = _panel(summary, pad=0.15)
        self.play(FadeIn(sbg), Write(summary))
        self.wait(3.2)
        self.play(*(FadeOut(o) for o in self.mobjects))


# ---------------------------------------------------------------------------
# Escena 4 — Neurona vs latente (evaluación ciega)
# ---------------------------------------------------------------------------


def _context_lines(kind: str, unit_id: int, n: int, left: int = 22, right: int = 10) -> list[str]:
    units = json.loads((ANALYSIS_DIR / "interp_sample.json").read_text())
    u = next(u for u in units if u["kind"] == kind and u["id"] == unit_id)
    out = []
    for c in u["contexts"]:
        t = c["text"]
        if "\ufffd" in t:
            continue
        pre, rest = t.split("«", 1)
        cur, post = rest.split("»", 1)
        pre = pre.replace("⏎", "↵")[-left:].rjust(left)
        post = post.replace("⏎", "↵")[:right]
        out.append(
            f"{escape(pre)}<span foreground='{C_ACCENT}' weight='bold'>{escape(cur.replace('⏎', '↵'))}</span>{escape(post)}"
        )
        if len(out) == n:
            break
    return out


class NeuronVsLatentScene(Scene):
    NEURON = 2787
    LATENT = 10274

    def construct(self) -> None:
        D = _data()
        ratings = json.loads((ANALYSIS_DIR / "interp_ratings.json").read_text())
        rmap = {(r["kind"], r["id"]): r for r in ratings}

        title = _title(
            r"Hallazgo 2: las latentes se leen, las neuronas no",
            r"Textos que más activan cada unidad, en datos que el modelo nunca vio en entrenamiento",
        )
        self.play(Write(title))

        def column(kind: str, uid: int, header: str, color: str) -> VGroup:
            lines = VGroup(*[
                MarkupText(s, font=MONO, font_size=17) for s in _context_lines(kind, uid, 7)
            ]).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
            r = rmap[(kind, uid)]
            head = Tex(header, color=color).scale(0.55)
            score = Tex(
                rf"puntuación ciega: {r['rater_default']['score']}/5 y {r['rater_slow']['score']}/5",
                color=LIGHT_GRAY,
            ).scale(0.42)
            col = VGroup(head, lines, score).arrange(DOWN, buff=0.28)
            return col

        left = column("mlp_neuron_l7", self.NEURON, rf"Neurona MLP {self.NEURON} (capa 7)", C_MLP)
        right = column("sae_latent", self.LATENT, rf"Latente {self.LATENT} del SAE", C_ACCENT)
        for col in (left, right):
            if col.width > 6.3:
                col.scale_to_fit_width(6.3)
        left.move_to(LEFT * 3.45 + DOWN * 0.35)
        right.move_to(RIGHT * 3.45 + DOWN * 0.35)
        bl, br = _panel(left), _panel(right)
        self.play(FadeIn(bl), FadeIn(left[0]), FadeIn(br), FadeIn(right[0]))
        self.play(
            *(FadeIn(ln, shift=RIGHT * 0.15) for ln in left[1]),
            *(FadeIn(ln, shift=RIGHT * 0.15) for ln in right[1]),
            lag_ratio=0.08, run_time=2.4,
        )
        self.wait(0.8)
        verdict_l = Tex(r"?`\,``Up to'', ``1,000'', ``Moroccan''\,?", color=C_MLP).scale(0.5)
        verdict_r = Tex(r"robar: \emph{steal, stole, stolen, theft}", color=C_ACCENT).scale(0.5)
        verdict_l.next_to(bl, DOWN, buff=0.18)
        verdict_r.next_to(br, DOWN, buff=0.18)
        self.play(FadeIn(left[2]), FadeIn(right[2]), Write(verdict_l), Write(verdict_r))
        self.wait(3.0)
        self.play(*(FadeOut(o) for o in (bl, br, left, right, verdict_l, verdict_r)))

        # --- histograma de la evaluación ciega completa
        sae = D["interp"]["sae"]
        mlp = D["interp"]["mlp"]
        n_s, n_m = sum(sae), sum(mlp)
        mean_s = sum((i + 1) * c for i, c in enumerate(sae)) / n_s
        mean_m = sum((i + 1) * c for i, c in enumerate(mlp)) / n_m
        ge4_s = 100 * (sae[3] + sae[4]) / n_s
        ge4_m = 100 * (mlp[3] + mlp[4]) / n_m

        sub = Tex(
            rf"{n_s} latentes y {n_m} neuronas al azar, mezcladas y puntuadas a ciegas de 1 a 5",
            color=LIGHT_GRAY,
        ).scale(0.45).next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(sub))

        base_y, max_h, top = -2.45, 2.9, max(max(sae), max(mlp))
        bar_w, gap = 0.55, 1.55
        x0 = -5.3
        baseline = Line([x0 - 1.0, base_y, 0], [x0 + 4 * gap + 1.0, base_y, 0], stroke_opacity=0.5)
        bars, cats = VGroup(), VGroup()
        for i in range(5):
            cx = x0 + i * gap
            for j, (cnt, color) in enumerate(((mlp[i], C_MLP), (sae[i], C_ACCENT))):
                h = max(0.02, max_h * cnt / top)
                b = Rectangle(width=bar_w, height=h, fill_color=color, fill_opacity=0.9, stroke_width=0)
                b.move_to([cx + (j - 0.5) * (bar_w + 0.06), base_y + h / 2, 0])
                v = Tex(str(cnt)).scale(0.4).next_to(b, UP, buff=0.06)
                bars.add(VGroup(b, v))
            cats.add(Tex(str(i + 1)).scale(0.5).move_to([cx, base_y - 0.3, 0]))
        cat_lbl = Tex(r"1 = sin patrón \qquad 5 = un solo concepto nítido", color=LIGHT_GRAY)\
            .scale(0.42).next_to(cats, DOWN, buff=0.18)
        leg = _legend([(r"neuronas MLP", C_MLP, False), (r"latentes SAE", C_ACCENT, False)])\
            .move_to([x0 + 0.6, 0.9, 0])
        self.play(Create(baseline), FadeIn(cats), FadeIn(cat_lbl), FadeIn(leg))
        self.play(*(GrowFromEdge(b[0], DOWN) for b in bars), run_time=1.6)
        self.play(*(FadeIn(b[1]) for b in bars))

        stats = VGroup(
            Tex(r"media").scale(0.5),
            Tex(rf"${_es(mean_m, 2)}$ frente a ${_es(mean_s, 2)}$").scale(0.6),
            Tex(r"puntuación $\ge 4$").scale(0.5),
            Tex(rf"${_es(ge4_m, 0)}\,\%$ frente a ${_es(ge4_s, 0)}\,\%$").scale(0.6),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        stats[2].shift(DOWN * 0.15)
        stats[3].shift(DOWN * 0.15)
        stats.move_to([4.95, -0.9, 0])
        stats = VGroup(_panel(stats, pad=0.22), stats)
        self.play(Write(stats))
        foot = _line(
            r"Dos evaluadores LLM independientes (correlación de Spearman $0{,}95$). "
            r"Mann--Whitney: $p < 10^{-19}$.",
            0.38, LIGHT_GRAY,
        ).to_edge(DOWN, buff=0.12)
        self.play(FadeIn(foot))
        self.wait(3.5)
        self.play(*(FadeOut(o) for o in self.mobjects))


# ---------------------------------------------------------------------------
# Escena 5 — Steering causal
# ---------------------------------------------------------------------------


STEER_SHOW = [(10274, "robo"), (19814, "peligro"), (19443, "color")]
THEFT_RX = re.compile(r"\b(steal\w*|stole\w*|theft\w*|thie(?:f|ves))\b", re.IGNORECASE)


def _wrap(s: str, width: int) -> list[str]:
    words, lines, cur = s.replace("\n", " ").split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


class SteeringScene(Scene):
    PROMPT_IDX = 1
    NEEDLE = "stole nearly"

    def construct(self) -> None:
        D = _data()
        st = json.loads((ANALYSIS_DIR / "steering.json").read_text())
        prompt = st["prompts"][self.PROMPT_IDX]
        feat = st["latents"]["10274"]["feature"]
        i = next(k for k, t in enumerate(feat["1.0"]["texts"]) if self.NEEDLE in t)
        base_txt = feat["0.0"]["texts"][i]
        steer_txt = feat["1.0"]["texts"][i]

        title = _title(
            r"Hallazgo 3: las latentes son palancas causales",
            r"Sumamos la dirección de una latente al flujo residual y dejamos escribir a GPT-2",
        )
        self.play(Write(title))

        def gen_block(label: str, color: str, text: str, highlight: bool) -> VGroup:
            lines = _wrap(text, 58)[:3]
            body = []
            for ln in lines:
                e = escape(ln)
                if highlight:
                    e = THEFT_RX.sub(lambda m: f"<span foreground='{C_ACCENT}' weight='bold'>{m.group(0)}</span>", e)
                body.append(MarkupText(e, font=MONO, font_size=18))
            body_g = VGroup(*body).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            head = Tex(label, color=color).scale(0.5)
            return VGroup(head, body_g).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        ptxt = MarkupText(f"<b>{escape(prompt)}</b> …", font=MONO, font_size=22)
        plabel = Tex(r"prompt", color=LIGHT_GRAY).scale(0.45)
        pgrp = VGroup(plabel, ptxt).arrange(RIGHT, buff=0.3).move_to(UP * 1.75)
        base = gen_block(r"sin intervención", C_RAND, base_txt, False)
        steer = gen_block(r"$+$ latente 10274 («robar»), $c = 1$", C_ACCENT, steer_txt, True)
        VGroup(base, steer).arrange(DOWN, aligned_edge=LEFT, buff=0.45).next_to(pgrp, DOWN, buff=0.5)
        VGroup(base, steer).set_x(0)
        self.play(FadeIn(pgrp))
        self.play(FadeIn(base, shift=UP * 0.1))
        self.wait(1.0)
        self.play(FadeIn(steer, shift=UP * 0.1))
        note = Tex(r"Misma semilla de muestreo en ambos casos.", color=LIGHT_GRAY).scale(0.4)\
            .to_edge(DOWN, buff=0.35)
        self.play(FadeIn(note))
        self.wait(3.5)
        self.play(*(FadeOut(o) for o in (pgrp, base, steer, note)))

        # --- curva dosis-respuesta con control aleatorio
        coeffs = st["coeffs"]
        ax = _make_axes((0, 3, 0.5), (0, 0.6, 0.1), length_x=7.2, length_y=3.9, y_dec=1, x_dec=1)\
            .shift(LEFT * 1.3 + DOWN * 0.45)
        xl = Tex(r"intensidad $c$ (múltiplos de la activación máxima observada)", color=LIGHT_GRAY)\
            .scale(0.42).next_to(ax.x_axis, DOWN, buff=0.4)
        yl = Tex(r"textos que mencionan el concepto", color=LIGHT_GRAY).scale(0.42)\
            .rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.45)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))

        curves, dots, items = [], [], []
        rand_all = []
        for (f, label), color in zip(STEER_SHOW, C_STEER):
            e = D["steering"][str(f)]
            curves.append(_polyline(ax, coeffs, e["feature"], color, width=4.5))
            dots.append(VGroup(*[Dot(ax.c2p(c, y), color=color, radius=0.05) for c, y in zip(coeffs, e["feature"])]))
            items.append((rf"latente {f} («{label}»)", color, False))
            rand_all.append(e["random"])
        rand_mean = np.mean(np.array(rand_all), axis=0)
        rand_curve = DashedVMobject(_polyline(ax, coeffs, rand_mean, C_RAND, width=3.5), num_dashes=40)
        items.append((r"dirección aleatoria, misma norma", C_RAND, True))
        leg = _legend(items, scale=0.42).next_to(ax, RIGHT, buff=0.35).shift(UP * 0.9)

        self.play(FadeIn(leg))
        self.play(*(Create(c) for c in curves), Create(rand_curve), run_time=2.2)
        self.play(*(FadeIn(d) for d in dots))

        best = max(
            ((f, lab, max(D["steering"][str(f)]["feature"])) for f, lab in STEER_SHOW), key=lambda t: t[2]
        )
        ann = VGroup(
            Tex(rf"pico: {_es(100 * best[2], 0)}\,\% («{best[1]}»)", color=C_STEER[[f for f, _ in STEER_SHOW].index(best[0])]).scale(0.45),
            Tex(rf"control: $\le {_es(100 * rand_mean.max(), 0)}\,\%$", color=C_RAND).scale(0.45),
            Tex(r"sin intervención: $\le 4\,\%$", color=LIGHT_GRAY).scale(0.45),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(leg, DOWN, buff=0.45, aligned_edge=LEFT)
        self.play(Write(ann))
        foot = _line(
            r"$5$ prompts $\times$ $16$ muestras por punto. Si empujas demasiado ($c \ge 2$), el texto se degrada.",
            0.38, LIGHT_GRAY,
        ).to_edge(DOWN, buff=0.12)
        self.play(FadeIn(foot))
        self.wait(3.5)
        self.play(*(FadeOut(o) for o in self.mobjects))


# ---------------------------------------------------------------------------
# Escena 6 — Generalización a contexto largo
# ---------------------------------------------------------------------------


class ContextLengthScene(Scene):
    ROWS = [
        ("ours_topk_24k", r"Este trabajo\\TopK 24k, entrenado a 1024", C_ACCENT),
        ("openai_topk_32k", r"OpenAI\\TopK 32k, entrenado a 64", C_MLP),
        ("jbloom_res_jb", r"J. Bloom\\ReLU 24k, entrenado a 128", C_BLOOM),
    ]

    def construct(self) -> None:
        D = _data()
        h128, h1024 = D["heldout_ctx128"], D["heldout_ctx1024"]

        title = _title(
            r"Hallazgo 4: entrenar con contexto largo importa",
            r"Pérdida extra (nats) al sustituir la capa 8 de GPT-2 por la reconstrucción del SAE. Menos es mejor.",
        )
        self.play(Write(title))

        base_y, max_h, vmax = -2.3, 3.2, 1.0
        xs = [-4.0, 0.0, 4.0]
        bw = 1.6
        baseline = Line([-6, base_y, 0], [6, base_y, 0], stroke_opacity=0.5)
        names = VGroup(*[
            Tex(lbl, color=col).scale(0.45).move_to([x, base_y - 0.55, 0])
            for (key, lbl, col), x in zip(self.ROWS, xs)
        ])

        def bars_for(h: dict) -> VGroup:
            g = VGroup()
            for (key, _lbl, col), x in zip(self.ROWS, xs):
                v = h[key]["ce_delta"]
                hh = max(0.03, max_h * v / vmax)
                b = Rectangle(width=bw, height=hh, fill_color=col, fill_opacity=0.9, stroke_width=0)
                b.move_to([x, base_y + hh / 2, 0])
                t = Tex(_es(v, 2)).scale(0.55).next_to(b, UP, buff=0.1)
                g.add(VGroup(b, t))
            return g

        tag128 = Tex(r"Evaluado con secuencias de 128 tokens").scale(0.55).move_to(UP * 2.1)
        tag1024 = Tex(r"Evaluado con secuencias de 1024 tokens").scale(0.55).move_to(UP * 2.1)
        b128 = bars_for(h128)
        self.play(Create(baseline), FadeIn(names), Write(tag128))
        self.play(*(GrowFromEdge(b[0], DOWN) for b in b128), run_time=1.2)
        self.play(*(FadeIn(b[1]) for b in b128))
        tie = Tex(r"Empate técnico: los tres pierden entre $0{,}12$ y $0{,}15$ nats.", color=LIGHT_GRAY)\
            .scale(0.45).next_to(tag128, DOWN, buff=0.15)
        self.play(FadeIn(tie))
        self.wait(2.5)

        b1024 = bars_for(h1024)
        self.play(
            FadeOut(tie),
            FadeTransform(tag128, tag1024),
            *(Transform(a[0], b[0]) for a, b in zip(b128, b1024)),
            *(FadeOut(a[1]) for a in b128),
            run_time=1.8, rate_func=smooth,
        )
        self.play(*(FadeIn(b[1]) for b in b1024))
        l0_j = h1024["jbloom_res_jb"]["l0"]
        l0_j128 = h128["jbloom_res_jb"]["l0"]
        notes = VGroup(
            Tex(
                rf"CE loss score: {_es(h1024['ours_topk_24k']['ce_score'], 3)} (este trabajo), "
                rf"{_es(h1024['openai_topk_32k']['ce_score'], 3)} (OpenAI), {_es(h1024['jbloom_res_jb']['ce_score'], 3)} (Bloom)",
            ).scale(0.42),
            Tex(
                rf"El SAE de Bloom pasa de {_es(l0_j128, 0)} a {_es(l0_j, 0)} latentes activas por token.",
                color=LIGHT_GRAY,
            ).scale(0.42),
        ).arrange(DOWN, buff=0.1).next_to(tag1024, DOWN, buff=0.15)
        self.play(FadeIn(notes))
        foot = _line(r"Mismos tokens de OpenWebText nunca vistos para los tres SAEs; BOS excluido.", 0.38, LIGHT_GRAY)\
            .to_edge(DOWN, buff=0.12)
        self.play(FadeIn(foot))
        self.wait(3.5)
        self.play(*(FadeOut(o) for o in self.mobjects))
