"""
analyze_sae.py — Verificación independiente y análisis del SAE publicado.

Todo se mide sobre datos NUNCA vistos en entrenamiento: el último shard
(`train-00072-of-00073`) del OpenWebText pretokenizado. El entrenamiento
consumió ~98 k filas del inicio del split, dentro del primer shard.

Etapas (`--stage`):

  metrics   Evalúa el SAE publicado en HF y dos SAEs públicos de referencia en
            el mismo punto del residual stream (blocks.8.hook_resid_pre ≡
            blocks.7.hook_resid_post), con el MISMO protocolo y los MISMOS
            tokens: varianza explicada, cossim, L0, ratio L2, CE/KL con la
            reconstrucción parcheada, ablación a cero, fracción de latentes vivas.

  features  Pasada larga sólo con nuestro SAE: densidad de disparo por latente,
            top-k contextos por latente y por neurona MLP de la capa 7 (para la
            comparación neurona vs feature), geometría del diccionario
            (similitud coseno entre direcciones del decoder vs vectores
            aleatorios en R^768) y muestra aleatoria para auto-interpretación.

  steer     Intervención causal: añade  c · max_act · W_dec[f]  al residual stream
            durante la generación (5 prompts × n_gen muestras) y mide, releyendo
            el texto con el modelo limpio, la tasa de disparo de f y la NLL; con
            control de dirección aleatoria de la misma norma. Logit lens de f.

Uso:
  python src/analyze_sae.py --stage metrics  --n-seqs 512 --ctx 128 [--local-sae name=dir]
  python src/analyze_sae.py --stage features --n-seqs 4096
  python src/analyze_sae.py --stage steer --latents 123,456
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
from sae_lens import SAE  # type: ignore[import-untyped]
from transformer_lens import HookedTransformer  # type: ignore[import-untyped]

REPO_ID = "alexcerezo/sae-gpt2-small-l8-topk32"
HELDOUT_REPO = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
HELDOUT_FILE = "data/train-00072-of-00073.parquet"
BOS = 50256
MLP_HOOK = "blocks.7.mlp.hook_post"

REFERENCES: dict[str, tuple[str, str]] = {
    "jbloom_res_jb": ("gpt2-small-res-jb", "blocks.8.hook_resid_pre"),
    "openai_topk_32k": ("gpt2-small-resid-post-v5-32k", "blocks.7.hook_resid_post"),
}

OUT = Path(__file__).resolve().parent.parent / "data" / "analysis"


# ---------------------------------------------------------------------------
# Datos y modelos
# ---------------------------------------------------------------------------


def load_heldout_tokens(n_seqs: int, ctx: int) -> torch.Tensor:
    """Ventanas de `ctx` tokens (BOS + ctx-1 tokens de texto) del shard held-out."""
    ds = load_dataset(HELDOUT_REPO, data_files=HELDOUT_FILE, split="train", streaming=True)
    body = ctx - 1
    rows: list[list[int]] = []
    for row in ds:
        ids = row["input_ids"]
        for s in range(0, len(ids) - body + 1, body):
            rows.append([BOS, *ids[s : s + body]])
            if len(rows) == n_seqs:
                return torch.tensor(rows, dtype=torch.long)
    raise RuntimeError(f"shard held-out agotado con {len(rows)} secuencias")


def load_our_sae(device: str) -> SAE:
    path = snapshot_download(REPO_ID, repo_type="model")
    return SAE.load_from_disk(path, device=device)


def load_model(kwargs: dict[str, Any], device: str, cache: dict[str, HookedTransformer]) -> HookedTransformer:
    key = json.dumps(kwargs, sort_keys=True)
    if key not in cache:
        model = HookedTransformer.from_pretrained("gpt2", device=device, **kwargs)
        model.eval()
        cache[key] = model
    return cache[key]


def model_kwargs_of(sae: SAE) -> dict[str, Any]:
    return dict(sae.cfg.metadata.model_from_pretrained_kwargs or {})


def layer_of(hook: str) -> int:
    return int(hook.split(".")[1])


# ---------------------------------------------------------------------------
# Etapa 1 — métricas comparables
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_sae(sae: SAE, model: HookedTransformer, tokens: torch.Tensor, batch: int) -> dict[str, float]:
    hook = sae.cfg.metadata.hook_name
    d_in, d_sae = sae.cfg.d_in, sae.cfg.d_sae
    dev = next(model.parameters()).device

    sum_x = torch.zeros(d_in, dtype=torch.float64)
    sum_x2 = 0.0
    sum_res = 0.0
    sum_cos = 0.0
    sum_l0 = 0.0
    sum_l2ratio = 0.0
    n_tok = 0
    fired = torch.zeros(d_sae, dtype=torch.bool)
    ce = defaultdict(float)
    kl = defaultdict(float)
    n_pred = 0

    for i in range(0, len(tokens), batch):
        tok = tokens[i : i + batch].to(dev)
        B, T = tok.shape
        clean_logits, cache = model.run_with_cache(tok, names_filter=hook)
        x = cache[hook][:, 1:].reshape(-1, d_in).float()  # sin BOS
        feats = sae.encode(x)
        xhat = sae.decode(feats).float()

        res = (x - xhat).pow(2).sum(-1)
        sum_res += res.sum().item()
        sum_x += x.sum(0).double().cpu()
        sum_x2 += x.pow(2).sum().item()
        sum_cos += torch.nn.functional.cosine_similarity(x, xhat, dim=-1).sum().item()
        sum_l0 += (feats > 0).sum().item()
        sum_l2ratio += (xhat.norm(dim=-1) / x.norm(dim=-1)).sum().item()
        fired |= (feats > 0).any(0).cpu()
        n_tok += x.shape[0]

        xhat_btd = xhat.reshape(B, T - 1, d_in).to(cache[hook].dtype)

        def patch_sae(act: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ARG001
            act[:, 1:] = xhat_btd
            return act

        def patch_zero(act: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ARG001
            act[:, 1:] = 0.0
            return act

        sae_logits = model.run_with_hooks(tok, fwd_hooks=[(hook, patch_sae)])
        abl_logits = model.run_with_hooks(tok, fwd_hooks=[(hook, patch_zero)])

        # predicciones desde las posiciones 1..T-2 (la posición 0 no se parchea)
        tgt = tok[:, 2:]
        lp_clean = clean_logits[:, 1:-1].log_softmax(-1).float()
        p_clean = lp_clean.exp()
        for name, logits in (("clean", clean_logits), ("sae", sae_logits), ("abl", abl_logits)):
            lp = logits[:, 1:-1].log_softmax(-1).float()
            ce[name] += -lp.gather(-1, tgt[..., None]).sum().item()
            if name != "clean":
                kl[name] += (p_clean * (lp_clean - lp)).sum().item()
        n_pred += tgt.numel()
        del clean_logits, sae_logits, abl_logits, lp_clean, p_clean, cache

    mean_x = sum_x / n_tok
    var = sum_x2 / n_tok - mean_x.pow(2).sum().item()
    ce_clean, ce_sae, ce_abl = (ce[k] / n_pred for k in ("clean", "sae", "abl"))
    kl_sae, kl_abl = kl["sae"] / n_pred, kl["abl"] / n_pred
    return {
        "d_sae": d_sae,
        "tokens": n_tok,
        "explained_variance": 1.0 - (sum_res / n_tok) / var,
        "cossim": sum_cos / n_tok,
        "l0": sum_l0 / n_tok,
        "l2_ratio": sum_l2ratio / n_tok,
        "frac_latents_fired": fired.float().mean().item(),
        "ce_clean": ce_clean,
        "ce_sae": ce_sae,
        "ce_ablation": ce_abl,
        "ce_delta": ce_sae - ce_clean,
        "ce_score": (ce_abl - ce_sae) / (ce_abl - ce_clean),
        "kl_sae": kl_sae,
        "kl_ablation": kl_abl,
        "kl_score": (kl_abl - kl_sae) / kl_abl,
    }


def stage_metrics(args: argparse.Namespace) -> None:
    tokens = load_heldout_tokens(args.n_seqs, args.ctx)
    models: dict[str, HookedTransformer] = {}
    saes: dict[str, SAE] = {"ours_topk_24k": load_our_sae(args.device)}
    for name, (release, sae_id) in REFERENCES.items():
        saes[name] = SAE.from_pretrained(release, sae_id, device=args.device)
    for spec in args.local_sae:
        name, path = spec.split("=", 1)
        saes[name] = SAE.load_from_disk(path, device=args.device)

    results: dict[str, Any] = {"protocol": {
        "heldout": f"{HELDOUT_REPO}/{HELDOUT_FILE}",
        "n_seqs": args.n_seqs, "ctx": args.ctx,
        "bos_excluded": True,
        "patched_positions": "1..T-1 (BOS intacto)",
        "prediction_positions": "1..T-2",
    }}
    for name, sae in saes.items():
        kw = model_kwargs_of(sae)
        model = load_model(kw, args.device, models)
        t0 = time.time()
        m = eval_sae(sae, model, tokens, args.batch)
        m.update({
            "hook": sae.cfg.metadata.hook_name,
            "architecture": sae.cfg.architecture(),
            "model_kwargs": kw,
            "normalize_activations": sae.cfg.normalize_activations,
            "seconds": round(time.time() - t0, 1),
        })
        results[name] = m
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"metrics_heldout_ctx{args.ctx}.json"
    path.write_text(json.dumps(results, indent=2))
    print("→", path)


# ---------------------------------------------------------------------------
# Etapa 2 — features
# ---------------------------------------------------------------------------


def _merge_topk(run_v: torch.Tensor, run_i: torch.Tensor, acts: torch.Tensor,
                gidx: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mantiene los k mayores valores por columna a lo largo de todas las pasadas."""
    kk = min(k, acts.shape[0])
    bv, bi = acts.topk(kk, dim=0)
    cat_v = torch.cat([run_v, bv], 0)
    cat_i = torch.cat([run_i, gidx[bi]], 0)
    v, sel = cat_v.topk(k, dim=0)
    return v, cat_i.gather(0, sel)


def render_context(tokens: torch.Tensor, g: int, tokenizer: Any, left: int = 14, right: int = 4) -> str:
    T = tokens.shape[1]
    s, p = divmod(int(g), T)
    seq = tokens[s].tolist()
    lo = max(1, p - left)
    pre = tokenizer.decode(seq[lo:p])
    cur = tokenizer.decode([seq[p]])
    post = tokenizer.decode(seq[p + 1 : p + 1 + right])
    return f"{pre}«{cur}»{post}".replace("\n", "⏎")


@torch.no_grad()
def decoder_geometry(W_dec: torch.Tensor, n_pairs: int, seed: int) -> dict[str, Any]:
    W = W_dec.float()
    W = W / W.norm(dim=1, keepdim=True)
    n, d = W.shape
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(n, d, generator=g)
    R = R / R.norm(dim=1, keepdim=True)

    def max_cos(M: torch.Tensor) -> torch.Tensor:
        out = torch.empty(n)
        for s in range(0, n, 2048):
            sims = M[s : s + 2048] @ M.T
            idx = torch.arange(s, min(s + 2048, n))
            sims[idx - s, idx] = -2.0
            out[s : s + 2048] = sims.max(1).values
        return out

    mc_dec, mc_rand = max_cos(W), max_cos(R)
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n, (n_pairs,), generator=g)
    keep = i != j
    i, j = i[keep], j[keep]

    def pair_cos(M: torch.Tensor) -> torch.Tensor:
        # por bloques: indexar 2M filas de golpe materializa ~6 GB por tensor
        out = torch.empty(len(i))
        for s in range(0, len(i), 65536):
            out[s : s + 65536] = (M[i[s : s + 65536]] * M[j[s : s + 65536]]).sum(-1)
        return out

    pair_dec, pair_rand = pair_cos(W), pair_cos(R)

    def summ(t: torch.Tensor) -> dict[str, float]:
        q = torch.quantile(t, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
        return {"mean": t.mean().item(), "std": t.std().item(),
                "q05": q[0].item(), "q25": q[1].item(), "median": q[2].item(),
                "q75": q[3].item(), "q95": q[4].item(), "max": t.max().item()}

    bins = np.linspace(-0.3, 1.0, 131)
    return {
        "n_directions": n, "d": d,
        "max_cos_decoder": summ(mc_dec), "max_cos_random": summ(mc_rand),
        "pair_cos_decoder": summ(pair_dec), "pair_cos_random": summ(pair_rand),
        "abs_pair_cos_decoder_mean": pair_dec.abs().mean().item(),
        "abs_pair_cos_random_mean": pair_rand.abs().mean().item(),
        "hist_bins": bins.tolist(),
        "hist_max_cos_decoder": np.histogram(mc_dec.numpy(), bins)[0].tolist(),
        "hist_max_cos_random": np.histogram(mc_rand.numpy(), bins)[0].tolist(),
    }


@torch.no_grad()
def stage_features(args: argparse.Namespace) -> None:
    tokens = load_heldout_tokens(args.n_seqs, args.ctx)
    sae = load_our_sae(args.device)
    model = load_model(model_kwargs_of(sae), args.device, {})
    hook = sae.cfg.metadata.hook_name
    d_in, d_sae, d_mlp = sae.cfg.d_in, sae.cfg.d_sae, model.cfg.d_mlp
    k = args.topk
    dev = next(model.parameters()).device
    T = tokens.shape[1]

    counts = torch.zeros(d_sae, dtype=torch.long)
    sum_act = torch.zeros(d_sae, dtype=torch.float64)
    f_v = torch.full((k, d_sae), -math.inf)
    f_i = torch.full((k, d_sae), -1, dtype=torch.long)
    n_v = torch.full((k, d_mlp), -math.inf)
    n_i = torch.full((k, d_mlp), -1, dtype=torch.long)
    n_tok = 0
    t0 = time.time()
    for s in range(0, len(tokens), args.batch):
        tok = tokens[s : s + args.batch].to(dev)
        B = tok.shape[0]
        _, cache = model.run_with_cache(tok, names_filter=[hook, MLP_HOOK], stop_at_layer=layer_of(hook) + 1)
        gidx = ((torch.arange(B)[:, None] + s) * T + torch.arange(1, T)[None, :]).reshape(-1)
        x = cache[hook][:, 1:].reshape(-1, d_in).float()
        feats = sae.encode(x).float().cpu()
        counts += (feats > 0).sum(0)
        sum_act += feats.sum(0).double()
        f_v, f_i = _merge_topk(f_v, f_i, feats, gidx, k)
        neur = cache[MLP_HOOK][:, 1:].reshape(-1, d_mlp).float().cpu()
        n_v, n_i = _merge_topk(n_v, n_i, neur, gidx, k)
        n_tok += x.shape[0]
        if (s // args.batch) % 50 == 0:
            print(f"  {s + B}/{len(tokens)} seqs  {time.time() - t0:.0f}s", flush=True)

    density = (counts.double() / n_tok).numpy()
    log_d = np.log10(np.clip(density, 1e-9, None))
    bins = np.linspace(-9, 0, 91)
    live = counts > 0

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "features_raw.npz",
        tokens=tokens.numpy().astype(np.int32),
        counts=counts.numpy(), density=density,
        mean_act_when_active=(sum_act / counts.clamp(min=1)).numpy(),
        feat_top_v=f_v.numpy(), feat_top_i=f_i.numpy(),
        neur_top_v=n_v.numpy(), neur_top_i=n_i.numpy(),
    )

    geom = decoder_geometry(sae.W_dec.detach().cpu(), n_pairs=2_000_000, seed=args.seed)
    stats = {
        "tokens": n_tok,
        "n_seqs": len(tokens), "ctx": T,
        "latents_never_fired": int((~live).sum()),
        "latents_density_gt_1e-2": int((density > 1e-2).sum()),
        "latents_density_lt_1e-5": int(((density < 1e-5) & live.numpy()).sum()),
        "density_median_live": float(np.median(density[live.numpy()])),
        "log10_density_hist_bins": bins.tolist(),
        "log10_density_hist": np.histogram(log_d[live.numpy()], bins)[0].tolist(),
        "w_dec_row_norm_mean": sae.W_dec.norm(dim=1).mean().item(),
        "geometry": geom,
    }
    (OUT / "feature_stats.json").write_text(json.dumps(stats, indent=2))

    # muestra aleatoria (ciega para el evaluador) de latentes vivas y neuronas MLP
    rng = random.Random(args.seed)
    eligible = [i for i in range(d_sae) if counts[i] >= k]
    sample_f = rng.sample(eligible, args.n_sample)
    sample_n = rng.sample(range(d_mlp), args.n_sample)
    tk = model.tokenizer

    def contexts(top_v: torch.Tensor, top_i: torch.Tensor, unit: int, n: int) -> list[dict[str, Any]]:
        out = []
        for r in range(n):
            v = float(top_v[r, unit])
            if not math.isfinite(v) or v <= 0:
                break
            out.append({"act": round(v, 3), "text": render_context(tokens, int(top_i[r, unit]), tk)})
        return out

    units = [{"kind": "sae_latent", "id": i, "density": float(density[i]),
              "contexts": contexts(f_v, f_i, i, args.n_ctx)} for i in sample_f]
    units += [{"kind": "mlp_neuron_l7", "id": i,
               "contexts": contexts(n_v, n_i, i, args.n_ctx)} for i in sample_n]
    rng.shuffle(units)
    (OUT / "interp_sample.json").write_text(json.dumps(units, indent=1, ensure_ascii=False))
    print(json.dumps({k2: v for k2, v in stats.items() if not k2.startswith(("log10", "geometry"))}, indent=2))
    print("geometry:", json.dumps({k2: v for k2, v in geom.items() if not k2.startswith("hist")}, indent=1))


# ---------------------------------------------------------------------------
# Etapa 3 — intervención causal
# ---------------------------------------------------------------------------


STEER_PROMPTS = [
    "I think that",
    "The main thing to know is",
    "Yesterday, we went to",
    "In this article, I will explain",
    "My favorite part was",
]


# Palabras que delatan el concepto de cada latente de la demo (etiquetado ciego previo).
CONCEPT_KEYWORDS: dict[int, re.Pattern[str]] = {
    f: re.compile(rf"\b({p})\b", re.IGNORECASE) for f, p in {
        10274: r"steal\w*|stole\w*|theft\w*|thie(f|ves)|rob\w*|burglar\w*|loot\w*|shoplift\w*",
        19814: r"danger\w*|hazard\w*|risk\w*|threat\w*|unsafe|perilous",
        14036: r"drink\w*|drank|drunk\w*|alcohol\w*|beer\w*|wine\w*|booze|liquor|vodka|whiskey",
        14527: r"jok\w+|joking|jest\w*|prank\w*|kidding",
        19443: r"colou?r\w*|red|blue|green|yellow|purple|orange|pink",
        17961: r"january|february|march|april|may|june|july|august|september|october|november|december",
    }.items()
}


@torch.no_grad()
def stage_steer(args: argparse.Namespace) -> None:
    """Suma  c · max_act_f · Ŵ_dec[f]  al residual en todas las posiciones (salvo BOS) y mide,
    por continuación muestreada:

    * `concept_rate`: fracción de continuaciones que contienen alguna palabra del concepto
      (`CONCEPT_KEYWORDS`, fijadas a partir de la etiqueta ciega, antes de ver resultados).
    * `fires_in_text`: fracción de continuaciones donde f se activa en algún token al releer
      el texto con el modelo LIMPIO (sin hook).
    * `ce_clean`: NLL media de la continuación bajo el modelo limpio (coste de fluidez).
    * Control: misma norma, dirección aleatoria uniforme en la esfera de R^768.
    """
    sae = load_our_sae(args.device)
    model = load_model(model_kwargs_of(sae), args.device, {})
    hook = sae.cfg.metadata.hook_name
    raw = np.load(OUT / "features_raw.npz")
    latents = [int(x) for x in args.latents.split(",")]
    W_U = model.W_U
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(args.seed)

    def make_hook(vec: torch.Tensor) -> Any:
        def add(act: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ARG001
            # con KV-cache cada paso posterior ve sólo el token nuevo: (B, 1, d)
            if act.shape[1] == 1:
                act += vec.to(act.dtype)
            else:
                act[:, 1:] += vec.to(act.dtype)  # BOS intacto
            return act
        return add

    def run(vec: torch.Tensor | None, f: int) -> dict[str, Any]:
        fires, ces, hits, texts = [], [], [], []
        kw = CONCEPT_KEYWORDS.get(f)
        for pi, prompt in enumerate(STEER_PROMPTS):
            ptok = model.to_tokens(prompt).to(dev)
            P = ptok.shape[1]
            torch.manual_seed(args.seed * 1000 + pi)
            hooks = [] if vec is None else [(hook, make_hook(vec))]
            with model.hooks(fwd_hooks=hooks):
                out = model.generate(ptok.repeat(args.n_gen, 1), max_new_tokens=args.max_new,
                                     temperature=0.8, top_p=0.95, verbose=False,
                                     stop_at_eos=False, return_type="tokens")
            logits, cache = model.run_with_cache(out, names_filter=hook)
            lp = logits[:, P - 1 : -1].log_softmax(-1).float()
            nll = -lp.gather(-1, out[:, P:, None]).squeeze(-1)
            z = sae.encode(cache[hook][:, P:].float())[..., f]
            fires.append((z > 0).any(1).float().cpu())
            ces.append(nll.mean(1).cpu())
            for t in out[:, P:]:
                s = model.tokenizer.decode(t)
                texts.append(s)
                hits.append(float(bool(kw and kw.search(s))))
        fr, ce, ht = torch.cat(fires), torch.cat(ces), torch.tensor(hits)
        n = len(fr)
        sem = lambda v: (v.std() / n**0.5).item()  # noqa: E731
        return {"concept_rate": ht.mean().item(), "concept_rate_sem": sem(ht),
                "fires_in_text": fr.mean().item(), "fires_in_text_sem": sem(fr),
                "ce_clean": ce.mean().item(), "ce_clean_sem": sem(ce), "texts": texts}

    results: dict[str, Any] = {"prompts": STEER_PROMPTS, "coeffs": args.coeffs,
                               "n_gen_per_prompt": args.n_gen, "max_new": args.max_new,
                               "latents": {}}
    for f in latents:
        d = sae.W_dec[f].detach()
        d = d / d.norm()
        r = torch.randn(d.shape[0], generator=g).to(d.device)
        r = r / r.norm()
        max_act = float(raw["feat_top_v"][0, f])
        logit_dir = d @ W_U
        entry: dict[str, Any] = {
            "max_act": max_act,
            "density": float(raw["density"][f]),
            "logit_lens_up": [model.tokenizer.decode([t]) for t in logit_dir.topk(12).indices.tolist()],
            "logit_lens_down": [model.tokenizer.decode([t]) for t in (-logit_dir).topk(12).indices.tolist()],
            "feature": {}, "random": {},
        }
        for c in args.coeffs:
            scale = c * max_act
            entry["feature"][str(c)] = run(scale * d if c else None, f)
            if c:
                entry["random"][str(c)] = run(scale * r, f)
            fe = entry["feature"][str(c)]
            print(f"latent {f} c={c}: concept={fe['concept_rate']:.2f} fires={fe['fires_in_text']:.2f} "
                  f"ce={fe['ce_clean']:.2f}", flush=True)
        results["latents"][str(f)] = entry

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "steering.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("→", OUT / "steering.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--stage", required=True, choices=["metrics", "features", "steer"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-seqs", type=int, default=1024)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--n-sample", type=int, default=120)
    p.add_argument("--n-ctx", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latents", type=str, default="")
    p.add_argument("--coeffs", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    p.add_argument("--max-new", type=int, default=40)
    p.add_argument("--n-gen", type=int, default=8)
    p.add_argument("--local-sae", action="append", default=[],
                   help="name=path de un SAE local extra para --stage metrics (repetible)")
    args = p.parse_args()
    torch.set_grad_enabled(False)
    {"metrics": stage_metrics, "features": stage_features, "steer": stage_steer}[args.stage](args)


if __name__ == "__main__":
    main()
