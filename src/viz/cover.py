"""
cover.py — Portadas minimalistas del artículo (formato 5:2 de X Articles, 3000×1200), sin título.

Todo lo que aparece sale de datos reales del proyecto:
  * activaciones del SAE (capa 8 de GPT-2) sobre frases de ejemplo
  * ejemplos máximos de la neurona 2787          data/analysis/interp_sample.json
  * mapa 2D de las 24 576 direcciones del decoder  UMAP (coseno) de W_dec

Los datos derivados se cachean en data/cover/; las imágenes van a assets/cover/.

Uso:
    make covers      # = uv run --with umap-learn python src/viz/cover.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cairo
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "cover"
CACHE = ROOT / "data" / "cover"
W, H = 3000, 1200
SAE_REPO = "alexcerezo/sae-gpt2-small-l8-topk32"
HOOK = "blocks.8.hook_resid_pre"

SANS, SERIF, MONO = "Inter", "EB Garamond", "DejaVu Sans Mono"

# (latente, etiqueta en español, color)
CONCEPTS = [
    (10274, "robar", "#F26B4E"),
    (19814, "peligro", "#FF9F43"),
    (19443, "color", "#F2C14E"),
    (17961, "meses", "#63D471"),
    (11659, "Reuters", "#4EC9F2"),
    (14748, "Star Wars", "#7C8CFF"),
    (5183, "role of", "#B57EDC"),
    (21543, "olvidar", "#FF6FB5"),
    (3433, "negación", "#E0E0E0"),
    (5078, "días", "#3DDC97"),
    (7840, "actuar", "#C0A0FF"),
]
COLOR = {lid: c for lid, _, c in CONCEPTS}
LABEL = {lid: lab for lid, lab, _ in CONCEPTS}

SENTENCE = "On 17 March, Reuters reported that he stole a dangerous weapon."
GRID_TOKEN = " stole"  # palabra cuyo patrón de 32 latentes muestra la portada E


# ---------------------------------------------------------------------------
# Utilidades de dibujo
# ---------------------------------------------------------------------------


def rgb(h: str, a: float = 1.0) -> tuple[float, float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


def canvas() -> tuple[cairo.ImageSurface, cairo.Context]:
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    return s, cairo.Context(s)


def font(cr: cairo.Context, family: str, size: float, bold: bool = False, italic: bool = False) -> None:
    cr.select_font_face(
        family,
        cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
    )
    cr.set_font_size(size)


def text(cr, x, y, s, color="#FFFFFF", alpha=1.0, anchor="l") -> float:
    ext = cr.text_extents(s)
    if anchor == "c":
        x -= ext.x_advance / 2
    elif anchor == "r":
        x -= ext.x_advance
    cr.set_source_rgba(*rgb(color, alpha))
    cr.move_to(x, y)
    cr.show_text(s)
    return ext.x_advance


def radial_bg(cr, c0: str, c1: str, cx: float, cy: float, r: float) -> None:
    g = cairo.RadialGradient(cx, cy, 0, cx, cy, r)
    g.add_color_stop_rgba(0, *rgb(c0))
    g.add_color_stop_rgba(1, *rgb(c1))
    cr.set_source(g)
    cr.paint()


def glow_dot(cr, x, y, r, color, alpha=1.0, halo=4.0) -> None:
    g = cairo.RadialGradient(x, y, 0, x, y, r * halo)
    g.add_color_stop_rgba(0, *rgb(color, alpha))
    g.add_color_stop_rgba(0.25, *rgb(color, alpha * 0.35))
    g.add_color_stop_rgba(1, *rgb(color, 0))
    cr.set_source(g)
    cr.arc(x, y, r * halo, 0, 2 * math.pi)
    cr.fill()


def rounded(cr, x, y, w, h, r) -> None:
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


def vignette(cr, strength=0.65) -> None:
    g = cairo.RadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, W * 0.62)
    g.add_color_stop_rgba(0, 0, 0, 0, 0)
    g.add_color_stop_rgba(1, 0, 0, 0, strength)
    cr.set_source(g)
    cr.paint()


def grain(surface: cairo.ImageSurface, amount: float = 6.0, seed: int = 0) -> None:
    """Grano fino para que los degradados no hagan bandas."""
    surface.flush()
    buf = np.ndarray((H, W, 4), dtype=np.uint8, buffer=surface.get_data())
    noise = np.random.default_rng(seed).normal(0, amount, (H, W, 1))
    buf[..., :3] = np.clip(buf[..., :3].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    surface.mark_dirty()


def save(surface, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    surface.write_to_png(str(p))
    return p


# ---------------------------------------------------------------------------
# Datos (cacheados en data/cover/)
# ---------------------------------------------------------------------------


def _model_and_sae():
    import torch
    from huggingface_hub import snapshot_download
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    sae = SAE.load_from_disk(snapshot_download(SAE_REPO), device="cpu")
    return model, sae


def sentence_data() -> dict:
    """Activaciones reales del SAE para SENTENCE: latentes de CONCEPTS por token y las 32 de GRID_TOKEN."""
    cache = CACHE / "sentence_acts.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        if d.get("sentence") == SENTENCE and d.get("grid_token") == GRID_TOKEN:
            return d
    model, sae = _model_and_sae()
    toks = model.to_tokens(SENTENCE)
    _, c = model.run_with_cache(toks, names_filter=HOOK)
    z = sae.encode(c[HOOK])[0]
    strs = model.to_str_tokens(toks)
    ids = [lid for lid, _, _ in CONCEPTS]
    t_grid = strs.index(GRID_TOKEN)
    active = (z[t_grid] > 0).nonzero().flatten().tolist()
    d = {
        "sentence": SENTENCE,
        "grid_token": GRID_TOKEN,
        "tokens": [[strs[t], {str(i): float(z[t, i]) for i in ids if z[t, i] > 0}] for t in range(1, len(strs))],
        "grid_active": {str(i): float(z[t_grid, i]) for i in active},
        "d_sae": int(z.shape[-1]),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(d, ensure_ascii=False))
    return d


def umap_xy() -> np.ndarray:
    cache = CACHE / "decoder_umap.npy"
    if not cache.exists():
        import umap  # uv run --with umap-learn
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        wd = load_file(Path(snapshot_download(SAE_REPO)) / "sae_weights.safetensors")["W_dec"]
        wd = (wd / wd.norm(dim=1, keepdim=True)).numpy()
        emb = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=0, low_memory=True).fit_transform(wd)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb.astype(np.float32))
    return np.load(cache)


def neuron_fragments(neuron: int = 2787) -> list[tuple[str, str, str]]:
    units = json.loads((ROOT / "data" / "analysis" / "interp_sample.json").read_text())
    u = next(u for u in units if u["kind"] == "mlp_neuron_l7" and u["id"] == neuron)
    out = []
    for c in u["contexts"]:
        if "\ufffd" in c["text"]:
            continue
        pre, rest = c["text"].split("«", 1)
        cur, post = rest.split("»", 1)
        out.append((pre[-10:].lstrip(), cur, post[:8].rstrip()))
    return out


# ---------------------------------------------------------------------------
# A — Constelación: las 24 576 direcciones del diccionario
# ---------------------------------------------------------------------------


def cover_constellation() -> Path:
    xy = umap_xy()
    s, cr = canvas()
    radial_bg(cr, "#0E1730", "#02040A", W * 0.5, H * 0.5, W * 0.6)

    lo, hi = np.percentile(xy, 0.3, 0), np.percentile(xy, 99.7, 0)
    x0, x1, y0, y1 = W * 0.18, W * 0.82, H * 0.07, H * 0.93
    sc = min((x1 - x0) / (hi[0] - lo[0]), (y1 - y0) / (hi[1] - lo[1]))
    ox = (x0 + x1) / 2 - sc * (lo[0] + hi[0]) / 2
    oy = (y0 + y1) / 2 - sc * (lo[1] + hi[1]) / 2
    P = np.column_stack([xy[:, 0] * sc + ox, xy[:, 1] * sc + oy])

    t = (P[:, 0] - P[:, 0].min()) / np.ptp(P[:, 0])
    palette = [np.array(rgb(c)[:3]) for c in ("#3A6FF2", "#7C5CFF", "#C04EF2", "#4EC9F2")]
    cr.set_operator(cairo.OPERATOR_ADD)
    for i in np.random.default_rng(1).permutation(len(P)):
        x, y = P[i]
        k = t[i] * (len(palette) - 1)
        j = min(int(k), len(palette) - 2)
        c = palette[j] * (1 - (k - j)) + palette[j + 1] * (k - j)
        cr.set_source_rgba(*c, 0.08)
        cr.arc(x, y, 6, 0, 2 * math.pi)
        cr.fill()
        cr.set_source_rgba(*c, 0.5)
        cr.arc(x, y, 1.6, 0, 2 * math.pi)
        cr.fill()
    cr.set_operator(cairo.OPERATOR_OVER)
    vignette(cr, 0.6)

    # unas pocas estrellas con nombre: la posición real de su dirección
    cx, cy = P[:, 0].mean(), P[:, 1].mean()
    for lid, lab, col in CONCEPTS[:6]:
        x, y = P[lid]
        glow_dot(cr, x, y, 8, col, 1.0, halo=7)
        cr.set_source_rgba(1, 1, 1, 1)
        cr.arc(x, y, 4, 0, 2 * math.pi)
        cr.fill()
        font(cr, SANS, 30)
        # etiquetas hacia fuera del mapa; «robar» y «peligro» casi coinciden, así que se separan en vertical
        right = x >= cx
        dy = {10274: -26, 19814: 52}.get(lid, -18 if y < cy else 38)
        text(cr, x + (22 if right else -22), y + dy, lab, "#E8ECF2", 0.9, anchor="l" if right else "r")

    grain(s, 5)
    return save(s, "cover_A_constelacion")


# ---------------------------------------------------------------------------
# B — Prisma: una neurona caótica se descompone en conceptos limpios
# ---------------------------------------------------------------------------


def cover_prism() -> Path:
    s, cr = canvas()
    radial_bg(cr, "#120F22", "#030206", W * 0.5, H * 0.5, W * 0.6)

    px, py, ph = W * 0.47, H * 0.55, 520
    tri = [(px, py - ph / 2), (px - ph * 0.55, py + ph / 2), (px + ph * 0.55, py + ph / 2)]
    ex, ey = px - ph * 0.27, py            # entrada del haz en la cara izquierda
    sx, sy = px + ph * 0.25, py - 30        # salida en la cara derecha
    by0 = H * 0.86

    # haz de entrada
    g = cairo.LinearGradient(0, by0, ex, ey)
    g.add_color_stop_rgba(0, 1, 1, 1, 0.0)
    g.add_color_stop_rgba(1, 1, 1, 1, 0.95)
    for wdt, a in ((50, 0.05), (20, 0.12)):
        cr.set_source_rgba(1, 1, 1, a)
        cr.set_line_width(wdt)
        cr.move_to(0, by0)
        cr.line_to(ex, ey)
        cr.stroke()
    cr.set_source(g)
    cr.set_line_width(6)
    cr.move_to(0, by0)
    cr.line_to(ex, ey)
    cr.stroke()

    # rayos de salida
    rays = CONCEPTS[:7]
    targets = np.linspace(H * 0.14, H * 0.86, len(rays))
    tx = W * 0.86
    cr.set_operator(cairo.OPERATOR_ADD)
    for (_, _, col), ty in zip(rays, targets):
        cr.move_to(sx, sy - 5)
        cr.line_to(tx, ty - 14)
        cr.line_to(tx, ty + 14)
        cr.line_to(sx, sy + 5)
        cr.close_path()
        lg = cairo.LinearGradient(sx, sy, tx, ty)
        lg.add_color_stop_rgba(0, *rgb(col, 0.95))
        lg.add_color_stop_rgba(1, *rgb(col, 0.25))
        cr.set_source(lg)
        cr.fill()
        cr.set_source_rgba(*rgb(col, 0.06))
        cr.set_line_width(40)
        cr.move_to(sx, sy)
        cr.line_to(tx, ty)
        cr.stroke()
    cr.set_operator(cairo.OPERATOR_OVER)
    for (lid, lab, col), ty in zip(rays, targets):
        font(cr, SANS, 32)
        text(cr, tx + 28, ty + 11, lab, col, 0.95)

    # prisma
    cr.move_to(*tri[0])
    for p in tri[1:]:
        cr.line_to(*p)
    cr.close_path()
    lg = cairo.LinearGradient(tri[1][0], tri[1][1], tri[0][0], tri[0][1])
    lg.add_color_stop_rgba(0, 1, 1, 1, 0.03)
    lg.add_color_stop_rgba(1, 1, 1, 1, 0.14)
    cr.set_source(lg)
    cr.fill_preserve()
    cr.set_source_rgba(1, 1, 1, 0.8)
    cr.set_line_width(3)
    cr.stroke()
    cr.move_to(ex, ey)
    cr.line_to(sx, sy)
    cr.set_source_rgba(1, 1, 1, 0.55)
    cr.set_line_width(4)
    cr.stroke()

    # entrada: lo que activa de verdad a la neurona 2787, en gris y pequeño
    frags = neuron_fragments()
    ang = math.atan2(ey - by0, ex)
    for i, fi in enumerate((0, 2, 4)):
        a, b, c = frags[fi]
        f = 0.14 + 0.25 * i
        bx, by = ex * f, by0 + (ey - by0) * f
        off = -46 if i % 2 == 0 else 72
        cr.save()
        cr.translate(bx + off * math.sin(ang), by - off * math.cos(ang))
        cr.rotate(ang)
        font(cr, MONO, 22)
        x = text(cr, 0, 0, a, "#8A8F98", 0.55)
        font(cr, MONO, 22, bold=True)
        x += text(cr, x, 0, b, "#FFFFFF", 0.85)
        font(cr, MONO, 22)
        text(cr, x, 0, c, "#8A8F98", 0.55)
        cr.restore()

    vignette(cr, 0.45)
    grain(s, 5, 2)
    return save(s, "cover_B_prisma")


# ---------------------------------------------------------------------------
# C — Una sola entrada de diccionario, en papel
# ---------------------------------------------------------------------------


def cover_dictionary() -> Path:
    s, cr = canvas()
    radial_bg(cr, "#F7F1E3", "#E4D8BD", W * 0.5, H * 0.45, W * 0.65)
    grain(s, 6, 3)
    ink, red = "#1E1A16", "#A3261B"

    cx = W * 0.5
    font(cr, SANS, 30, bold=True)
    text(cr, cx, 400, "10 274", red, 0.9, anchor="c")
    font(cr, SERIF, 190, bold=True)
    wword = cr.text_extents("robar").x_advance
    font(cr, SERIF, 64, italic=True)
    wcat = cr.text_extents("v.").x_advance
    x0 = cx - (wword + 30 + wcat) / 2
    font(cr, SERIF, 190, bold=True)
    text(cr, x0, 600, "robar", ink)
    font(cr, SERIF, 64, italic=True)
    text(cr, x0 + wword + 30, 600, "v.", red)
    font(cr, SERIF, 52)
    text(cr, cx, 700, "steal · stole · stolen · theft", ink, 0.85, anchor="c")

    cr.set_source_rgba(*rgb(ink, 0.35))
    cr.set_line_width(2)
    cr.move_to(cx - 60, 780)
    cr.line_to(cx + 60, 780)
    cr.stroke()
    font(cr, SERIF, 34, italic=True)
    text(cr, cx, 860, "1 de 24 576", ink, 0.55, anchor="c")

    vignette(cr, 0.16)
    return save(s, "cover_C_diccionario")


# ---------------------------------------------------------------------------
# D — Una frase real, iluminada por las latentes que se encienden
# ---------------------------------------------------------------------------


def cover_lit_text() -> Path:
    d = sentence_data()
    toks = [(tok, {int(k): v for k, v in a.items() if int(k) in COLOR}) for tok, a in d["tokens"]]
    s, cr = canvas()
    radial_bg(cr, "#0B1222", "#020308", W * 0.5, H * 0.5, W * 0.6)
    vignette(cr, 0.45)

    size, vmax = 104, 40.0
    font(cr, SERIF, size)
    total = sum(cr.text_extents(t).x_advance for t, _ in toks)
    x, y = (W - total) / 2, H * 0.52
    space = cr.text_extents(" ").x_advance
    labels = []
    for tok, acts in toks:
        font(cr, SERIF, size)
        wtok = cr.text_extents(tok).x_advance
        # solo marcamos activaciones fuertes: el resto queda en gris, como texto sin concepto
        acts = {k: v for k, v in acts.items() if v > 10}
        if acts:
            lid, v = max(acts.items(), key=lambda kv: kv[1])
            col = COLOR[lid]
            sx = x + (space if tok.startswith(" ") else 0)
            ww = x + wtok - sx
            cxg, cyg, rg = sx + ww / 2, y - size * 0.3, ww * 0.75 + 80
            glow = cairo.RadialGradient(cxg, cyg, 0, cxg, cyg, rg)
            glow.add_color_stop_rgba(0, *rgb(col, 0.45 * min(1, v / vmax)))
            glow.add_color_stop_rgba(1, *rgb(col, 0))
            cr.set_source(glow)
            cr.arc(cxg, cyg, rg, 0, 2 * math.pi)
            cr.fill()
            cr.rectangle(sx, y + size * 0.22, ww, 5)
            cr.set_source_rgba(*rgb(col, 0.95))
            cr.fill()
            labels.append((sx + ww / 2, LABEL[lid], col))
        font(cr, SERIF, size)
        text(cr, x, y, tok, "#FFFFFF" if acts else "#5E6676")
        x += wtok
    for lx, lab, col in labels:
        font(cr, SANS, 28)
        text(cr, lx, y + 92, lab, col, 0.95, anchor="c")

    grain(s, 4, 4)
    return save(s, "cover_D_frase")


# ---------------------------------------------------------------------------
# E — 24 576 casillas; se encienden las 32 que usa GPT-2 para « stole»
# ---------------------------------------------------------------------------


def cover_grid() -> Path:
    d = sentence_data()
    active = {int(k): v for k, v in d["grid_active"].items()}
    n = d["d_sae"]
    cols, rows = 256, n // 256  # 256 × 96
    s, cr = canvas()
    radial_bg(cr, "#0A0F1C", "#020308", W * 0.5, H * 0.5, W * 0.6)

    pitch = min((W * 0.86) / cols, (H * 0.80) / rows)
    gx0 = (W - pitch * (cols - 1)) / 2
    gy0 = (H - pitch * (rows - 1)) / 2
    cr.set_source_rgba(1, 1, 1, 0.16)
    for i in range(n):
        if i in active:
            continue
        cr.rectangle(gx0 + (i % cols) * pitch - 1.4, gy0 + (i // cols) * pitch - 1.4, 2.8, 2.8)
    cr.fill()

    vmax = max(active.values())
    hero = 10274
    for i, v in sorted(active.items(), key=lambda kv: kv[1]):
        x, y = gx0 + (i % cols) * pitch, gy0 + (i // cols) * pitch
        a = 0.35 + 0.65 * v / vmax
        col = "#F26B4E" if i == hero else "#F2E6C8"
        glow_dot(cr, x, y, (10 if i == hero else 4 + 6 * v / vmax), col, (1.0 if i == hero else a), halo=5)
        cr.set_source_rgba(*rgb(col if i == hero else "#FFFFFF", 1))
        cr.rectangle(x - 3, y - 3, 6, 6)
        cr.fill()

    hx, hy = gx0 + (hero % cols) * pitch, gy0 + (hero // cols) * pitch
    font(cr, SANS, 28)
    text(cr, hx + 30, hy - 22, "robar", "#F26B4E")

    font(cr, MONO, 24)
    text(cr, gx0, H - 60, f"«{GRID_TOKEN.strip()}»  ·  {len(active)} de {n:,}".replace(",", " "), "#8A8F98", 0.8)

    vignette(cr, 0.4)
    grain(s, 4, 5)
    return save(s, "cover_E_rejilla")


if __name__ == "__main__":
    for fn in (cover_constellation, cover_prism, cover_dictionary, cover_lit_text, cover_grid):
        print(fn())
