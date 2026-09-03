"""
train_sae.py — TopK Sparse Autoencoder sobre gpt2-small con checkpointing atómico
y reanudación bit-exacta.

Diseño:
  * `TransformerLens.HookedTransformer` congelado en bf16 provee activaciones
    de `blocks.8.hook_resid_pre` (equivalente a `blocks.7.hook_resid_post`).
  * `sae_lens.ActivationsStore` alimenta un buffer mezclador de activaciones.
  * `sae_lens.TopKTrainingSAE` (k=32, expansión 32x → 24 576 latentes, AuxK
    para revivir dead latents) actualizado con Adam + LR warmup+cosine.
  * Escalado de activaciones estilo OpenAI: factor constante
    sqrt(d_in)/mean_norm estimado una vez y plegado en los pesos al final
    vía `fold_activation_norm_scaling_factor`. Compatible con la aux-loss
    de TopK (que rechaza `constant_norm_rescale`/`layer_norm`).
  * Checkpoint atómico cada N pasos o T minutos: escritura a directorio .tmp/,
    fsync recursivo, `os.replace` para publicar, `manifest.json` con SHA-256
    de cada fichero para detectar corrupción en el resume, y symlink `latest`.
    Reanudación por defecto: `--resume auto` busca el checkpoint íntegro más
    reciente, restaura pesos + optimizador + scheduler + RNG (torch/cuda/numpy/
    random) + contadores de pasos y dead-latents + estado del ActivationsStore
    + run-id de W&B (`resume="must"`).
  * SIGTERM/SIGINT provoca un checkpoint final atómico y salida limpia.
  * Al terminar: `save_inference_model` (pliega norma del decoder), evaluación
    completa (`run_evals` con CE-loss, L0, varianza explicada, KL) y push
    opcional a Hugging Face Hub y a W&B como artifact.

Uso mínimo en Runpod (dentro de tmux, con el pod ya provisionado):

  export WANDB_API_KEY=...
  export HF_TOKEN=...
  export HF_HOME=/workspace/hf-cache
  python -u src/train_sae.py \
      --checkpoint-dir /workspace/checkpoints/sae-gpt2 \
      --resume auto \
      --hf-repo-id <user>/sae-gpt2-small-l8-topk32

Smoke test local (CPU, sin red requerido tras primera descarga):

  python -u src/train_sae.py --smoke \
      --checkpoint-dir /tmp/sae_smoke --resume auto
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import logging
import math
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

# --- SAELens / TransformerLens (imports diferidos donde causen coste alto) ---
from sae_lens import (  # type: ignore[import-untyped]
    ActivationsStore,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
)
from sae_lens.constants import SAE_WEIGHTS_FILENAME  # type: ignore[import-untyped]
from sae_lens.saes.sae import (  # type: ignore[import-untyped]
    SAEMetadata,
    TrainStepInput,
)
from transformer_lens import HookedTransformer  # type: ignore[import-untyped]

logger = logging.getLogger("train_sae")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Modelo objetivo
    model_name: str = "gpt2"
    hook_name: str = "blocks.8.hook_resid_pre"
    d_in: int = 768
    context_size: int = 1024
    prepend_bos: bool = True

    # SAE
    expansion_factor: int = 32
    k: int = 32
    aux_loss_coefficient: float = 1.0 / 32.0
    decoder_init_norm: float = 0.1
    apply_b_dec_to_input: bool = True

    # Datos
    dataset_path: str = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
    dataset_config: str | None = None
    dataset_split: str = "train"
    streaming: bool = True
    store_batch_size_prompts: int = 32
    n_batches_in_buffer: int = 32
    train_batch_size_tokens: int = 8192
    total_training_tokens: int = 100_000_000
    n_batches_for_norm_estimate: int = 100
    exclude_bos: bool = True

    # Optimización
    lr: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 6.25e-10  # OpenAI: 1 / (batch * d_in) heurístico
    grad_clip: float = 1.0
    warm_up_steps: int = 1000
    decay_frac: float = 0.20  # último 20 % con cosine hasta lr/10
    lr_end_frac: float = 0.10  # lr_end = lr * 0.1
    dead_feature_window: int = 1000  # pasos sin dispararse ⇒ dead

    # Precisión / dispositivo
    device: str = "auto"
    precision: str = "bf16"  # {bf16, fp16, fp32}
    autocast_lm: bool = True

    # Checkpoints
    checkpoint_dir: str = "/workspace/checkpoints/sae-gpt2"
    save_every_steps: int = 1000
    save_every_seconds: int = 600
    keep_last_k: int = 3

    # Eval
    eval_every_steps: int = 500
    n_eval_reconstruction_batches: int = 8
    n_eval_sparsity_variance_batches: int = 4
    eval_batch_size_prompts: int = 16

    # Logging / upload
    wandb_project: str = "minibeca-xmihura"
    wandb_entity: str | None = None
    wandb_mode: str = "online"  # online | offline | disabled
    wandb_log_every: int = 10
    hf_repo_id: str | None = None
    hf_private: bool = False

    # Reanudación
    resume: str = "auto"  # "auto" | "none" | ruta explícita
    seed: int = 42
    deadline_hours: float | None = None  # límite blando de wall time

    # Smoke test
    smoke: bool = False

    @property
    def d_sae(self) -> int:
        return self.d_in * self.expansion_factor

    @property
    def training_steps(self) -> int:
        return max(1, self.total_training_tokens // self.train_batch_size_tokens)

    @property
    def decay_steps(self) -> int:
        return int(self.training_steps * self.decay_frac)


def apply_smoke_overrides(cfg: TrainConfig) -> TrainConfig:
    """Reduce todos los ejes para verificar checkpoint+resume en CPU."""
    cfg.expansion_factor = 2
    cfg.k = 8
    cfg.context_size = 64
    cfg.store_batch_size_prompts = 2
    cfg.n_batches_in_buffer = 2
    cfg.train_batch_size_tokens = 64
    cfg.total_training_tokens = 64 * 20
    cfg.n_batches_for_norm_estimate = 2
    cfg.warm_up_steps = 2
    cfg.save_every_steps = 5
    cfg.save_every_seconds = 3600
    cfg.eval_every_steps = 10_000  # desactivado en smoke
    cfg.n_eval_reconstruction_batches = 1
    cfg.n_eval_sparsity_variance_batches = 1
    cfg.dataset_path = "Salesforce/wikitext"
    cfg.dataset_config = "wikitext-103-raw-v1"
    cfg.streaming = False
    cfg.precision = "fp32"
    cfg.autocast_lm = False
    cfg.wandb_mode = "disabled"
    cfg.hf_repo_id = None
    cfg.deadline_hours = None
    return cfg


# ---------------------------------------------------------------------------
# Utilidades: dispositivo, precisión, semillas, RNG
# ---------------------------------------------------------------------------


def resolve_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Escalado constante estilo OpenAI  (compatible con TopK aux-loss)
# ---------------------------------------------------------------------------


class ConstantScaler:
    """Multiplica cada batch por un escalar α = sqrt(d_in) / mean_norm.

    El factor se estima UNA vez sobre `n_batches` batches del store y se
    persiste en el checkpoint. Al final se pliega en los pesos del SAE con
    `fold_activation_norm_scaling_factor` para que la inferencia opere en el
    espacio original de activaciones (idéntico contrato que Anthropic 2024).
    """

    def __init__(self) -> None:
        self.factor: float | None = None

    @torch.no_grad()
    def estimate(self, store: ActivationsStore, d_in: int, n_batches: int) -> None:
        norms: list[float] = []
        for _ in range(n_batches):
            batch = next(store)
            norms.append(batch.float().norm(dim=-1).mean().item())
        mean_norm = float(np.mean(norms))
        self.factor = math.sqrt(d_in) / max(mean_norm, 1e-8)
        logger.info(
            "constant scaler: mean_norm=%.4f  factor=%.6f (n_batches=%d)",
            mean_norm,
            self.factor,
            n_batches,
        )

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.factor is None else x * self.factor

    def state_dict(self) -> dict[str, Any]:
        return {"factor": self.factor}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.factor = state.get("factor")


# ---------------------------------------------------------------------------
# LR scheduler: warmup lineal → constante → cosine hasta lr_end
# ---------------------------------------------------------------------------


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_steps: int,
    warm_up_steps: int,
    decay_steps: int,
    lr_end_frac: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warm_up_steps = max(0, min(warm_up_steps, training_steps))
    decay_steps = max(0, min(decay_steps, training_steps - warm_up_steps))
    plateau_end = training_steps - decay_steps

    def lr_lambda(step: int) -> float:
        if step < warm_up_steps:
            return (step + 1) / max(1, warm_up_steps)
        if step < plateau_end:
            return 1.0
        if decay_steps == 0:
            return lr_end_frac
        progress = (step - plateau_end) / decay_steps
        cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return lr_end_frac + (1.0 - lr_end_frac) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint atómico
# ---------------------------------------------------------------------------

TRAINER_STATE_FILE = "trainer_state.pt"
MANIFEST_FILE = "manifest.json"
SCALER_FILE = "constant_scaler.json"
WANDB_ID_FILE = "wandb_run_id.txt"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_save_checkpoint(
    root: Path,
    step: int,
    *,
    sae: TopKTrainingSAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    store: ActivationsStore,
    scaler: ConstantScaler,
    trainer_state: dict[str, Any],
    wandb_run_id: str | None,
    keep_last_k: int,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step_{step:010d}"
    tmp = root / f"step_{step:010d}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # 1) Pesos + config del SAE en formato SAELens (safetensors + cfg.json).
    sae.save_model(str(tmp))

    # 2) Estado del ActivationsStore (n_dataset_processed) — via SAELens.
    store.save_to_checkpoint(str(tmp))

    # 3) Escalador constante.
    with (tmp / SCALER_FILE).open("w") as fh:
        json.dump(scaler.state_dict(), fh)

    # 4) Estado del trainer (optimizer, scheduler, RNG, contadores).
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer_state": trainer_state,
            "rng": snapshot_rng(),
            "step": step,
        },
        tmp / TRAINER_STATE_FILE,
    )

    # 5) wandb run id (texto plano, opcional).
    if wandb_run_id:
        (tmp / WANDB_ID_FILE).write_text(wandb_run_id)

    # 6) Manifest con SHA-256 (se escribe último → señal de completitud).
    manifest = {
        "step": step,
        "created_at": time.time(),
        "files": {
            p.name: _sha256(p)
            for p in sorted(tmp.iterdir())
            if p.is_file() and p.name != MANIFEST_FILE
        },
    }
    (tmp / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))

    # 7) Fsync de cada fichero + del propio directorio antes de publicar.
    for p in tmp.iterdir():
        if p.is_file():
            fd = os.open(p, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    _fsync_dir(tmp)

    # 8) Publicación atómica: rename tmp → final.
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    _fsync_dir(root)

    # 9) Symlink `latest` (best-effort; sistemas sin enlaces caen a fichero).
    latest = root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(final.name)
    except OSError:
        (root / "latest.txt").write_text(final.name)

    # 10) Retención de últimos K.
    steps = sorted(
        (p for p in root.glob("step_*") if p.is_dir() and not p.name.endswith(".tmp")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for old in steps[:-keep_last_k]:
        shutil.rmtree(old, ignore_errors=True)

    logger.info("checkpoint written atomically: %s", final)
    return final


def _verify_manifest(ckpt: Path) -> bool:
    mf = ckpt / MANIFEST_FILE
    if not mf.exists():
        return False
    try:
        manifest = json.loads(mf.read_text())
    except json.JSONDecodeError:
        return False
    for name, sha in manifest.get("files", {}).items():
        p = ckpt / name
        if not p.exists() or _sha256(p) != sha:
            logger.warning("checkpoint %s: %s failed hash check", ckpt.name, name)
            return False
    return True


def find_latest_valid_checkpoint(root: Path) -> Path | None:
    if not root.exists():
        return None
    for p in root.glob("*.tmp"):  # limpieza de escrituras interrumpidas
        shutil.rmtree(p, ignore_errors=True)
    candidates = sorted(
        (p for p in root.glob("step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
        reverse=True,
    )
    for c in candidates:
        if _verify_manifest(c):
            return c
    return None


def load_checkpoint(
    ckpt: Path,
    *,
    sae: TopKTrainingSAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    store: ActivationsStore,
    scaler: ConstantScaler,
) -> tuple[int, dict[str, Any], str | None]:
    # SAE weights
    state = load_file(str(ckpt / SAE_WEIGHTS_FILENAME))
    sae.load_state_dict(state)

    # Constant scaler
    scaler.load_state_dict(json.loads((ckpt / SCALER_FILE).read_text()))

    # ActivationsStore (fast-forward del stream por n_dataset_processed)
    store.load_from_checkpoint(str(ckpt))

    # Trainer state
    bundle = torch.load(ckpt / TRAINER_STATE_FILE, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(bundle["optimizer"])
    scheduler.load_state_dict(bundle["scheduler"])
    restore_rng(bundle["rng"])
    step: int = int(bundle["step"])
    trainer_state: dict[str, Any] = bundle["trainer_state"]

    wandb_id_path = ckpt / WANDB_ID_FILE
    wandb_run_id = wandb_id_path.read_text().strip() if wandb_id_path.exists() else None

    logger.info("resumed from %s (step=%d, wandb_run_id=%s)", ckpt, step, wandb_run_id)
    return step, trainer_state, wandb_run_id


# ---------------------------------------------------------------------------
# Manejo de señales (SIGTERM / SIGINT) para checkpoint final limpio
# ---------------------------------------------------------------------------


class GracefulKiller:
    def __init__(self) -> None:
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, _frame: Any) -> None:
        logger.warning("signal %d recibida — checkpoint pendiente antes de salir", signum)
        self.requested = True


# ---------------------------------------------------------------------------
# Construcción de piezas (modelo, store, SAE)
# ---------------------------------------------------------------------------


def build_model(cfg: TrainConfig, device: torch.device) -> HookedTransformer:
    dtype = resolve_dtype(cfg.precision)
    logger.info("loading %s in %s on %s", cfg.model_name, dtype, device)
    model = HookedTransformer.from_pretrained(cfg.model_name)
    model = model.to(dtype).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if cfg.d_in != model.cfg.d_model:
        raise ValueError(
            f"d_in={cfg.d_in} ≠ model.d_model={model.cfg.d_model}. Ajusta --d-in."
        )
    return model


def build_store(
    cfg: TrainConfig, model: HookedTransformer, device: torch.device
) -> ActivationsStore:
    dataset: Any = cfg.dataset_path
    if cfg.dataset_config is not None:
        from datasets import load_dataset  # type: ignore[import-untyped]

        dataset = load_dataset(
            cfg.dataset_path,
            cfg.dataset_config,
            split=cfg.dataset_split,
            streaming=cfg.streaming,
        )

    exclude_special: torch.Tensor | None = None
    if cfg.exclude_bos:
        tok = model.tokenizer
        bos = getattr(tok, "bos_token_id", None) if tok is not None else None
        if bos is not None:
            exclude_special = torch.tensor([bos], dtype=torch.long, device=device)

    return ActivationsStore(
        model=model,
        dataset=dataset,
        streaming=cfg.streaming,
        hook_name=cfg.hook_name,
        hook_head_index=None,
        context_size=cfg.context_size,
        d_in=cfg.d_in,
        n_batches_in_buffer=cfg.n_batches_in_buffer,
        total_training_tokens=cfg.total_training_tokens,
        store_batch_size_prompts=cfg.store_batch_size_prompts,
        train_batch_size_tokens=cfg.train_batch_size_tokens,
        prepend_bos=cfg.prepend_bos,
        normalize_activations="none",  # escalado externo constante
        device=device,
        dtype="float32",
        autocast_lm=cfg.autocast_lm,
        exclude_special_tokens=exclude_special,
    )


def build_sae(cfg: TrainConfig, device: torch.device) -> TopKTrainingSAE:
    metadata = SAEMetadata(
        model_name=cfg.model_name,
        hook_name=cfg.hook_name,
        hook_head_index=None,
        context_size=cfg.context_size,
        prepend_bos=cfg.prepend_bos,
        dataset_path=cfg.dataset_path,
        seqpos_slice=(None,),
    )
    sae_cfg = TopKTrainingSAEConfig(
        d_in=cfg.d_in,
        d_sae=cfg.d_sae,
        k=cfg.k,
        aux_loss_coefficient=cfg.aux_loss_coefficient,
        rescale_acts_by_decoder_norm=True,
        decoder_init_norm=cfg.decoder_init_norm,
        apply_b_dec_to_input=cfg.apply_b_dec_to_input,
        normalize_activations="none",  # ver ConstantScaler
        dtype="float32",  # pesos del SAE en fp32 (estables y baratos)
        device=str(device),
        metadata=metadata,
    )
    return TopKTrainingSAE(sae_cfg)


# ---------------------------------------------------------------------------
# W&B (opcional, con resume por run-id)
# ---------------------------------------------------------------------------


def init_wandb(cfg: TrainConfig, existing_run_id: str | None) -> Any | None:
    if cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("wandb no instalado; logging deshabilitado")
        return None
    kwargs: dict[str, Any] = {
        "project": cfg.wandb_project,
        "entity": cfg.wandb_entity,
        "config": dataclasses.asdict(cfg),
        "mode": cfg.wandb_mode,
    }
    if existing_run_id is not None:
        kwargs["id"] = existing_run_id
        kwargs["resume"] = "must"
    run = wandb.init(**kwargs)
    return run


# ---------------------------------------------------------------------------
# Evaluación (usa SAELens.run_evals con un wrapper adaptador de ConstantScaler)
# ---------------------------------------------------------------------------


class _ScalerAdapter:
    """SAELens.run_evals espera un objeto con `.scale(x)` (interfaz de
    `ActivationScaler`); nuestro ConstantScaler ya lo cumple, este wrapper solo
    fija un `scaling_factor` inspeccionable por los evals que lo usen."""

    def __init__(self, scaler: ConstantScaler) -> None:
        self._s = scaler
        self.scaling_factor = scaler.factor

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return self._s.scale(x)

    def unscale(self, x: torch.Tensor) -> torch.Tensor:
        return x if self._s.factor is None else x / self._s.factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale(x)


def maybe_run_evals(
    cfg: TrainConfig,
    step: int,
    sae: TopKTrainingSAE,
    store: ActivationsStore,
    model: HookedTransformer,
    scaler: ConstantScaler,
) -> dict[str, Any] | None:
    if cfg.eval_every_steps <= 0 or step == 0 or step % cfg.eval_every_steps != 0:
        return None
    from sae_lens import run_evals  # type: ignore[import-untyped]
    from sae_lens.evals import EvalConfig  # type: ignore[import-untyped]

    eval_cfg = EvalConfig(
        batch_size_prompts=cfg.eval_batch_size_prompts,
        n_eval_reconstruction_batches=cfg.n_eval_reconstruction_batches,
        n_eval_sparsity_variance_batches=cfg.n_eval_sparsity_variance_batches,
        compute_kl=True,
        compute_ce_loss=True,
        compute_l2_norms=True,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
    )
    sae.eval()
    with torch.no_grad():
        metrics, _feat_metrics = run_evals(
            sae=sae,
            activation_store=store,
            model=model,
            activation_scaler=_ScalerAdapter(scaler),  # type: ignore[arg-type]
            eval_config=eval_cfg,
            exclude_special_tokens=cfg.exclude_bos,
        )
    sae.train()
    return metrics


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    seed_everything(cfg.seed)

    device = resolve_device(cfg.device)
    logger.info("device=%s  precision=%s  d_sae=%d  steps=%d",
                device, cfg.precision, cfg.d_sae, cfg.training_steps)

    model = build_model(cfg, device)
    store = build_store(cfg, model, device)
    sae = build_sae(cfg, device)
    sae.train()

    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )
    scheduler = build_scheduler(
        optimizer,
        training_steps=cfg.training_steps,
        warm_up_steps=cfg.warm_up_steps,
        decay_steps=cfg.decay_steps,
        lr_end_frac=cfg.lr_end_frac,
    )
    scaler = ConstantScaler()

    # ---- Reanudación ----------------------------------------------------
    ckpt_root = Path(cfg.checkpoint_dir)
    resume_path: Path | None = None
    if cfg.resume == "auto":
        resume_path = find_latest_valid_checkpoint(ckpt_root)
    elif cfg.resume not in ("none", ""):
        p = Path(cfg.resume)
        resume_path = p if _verify_manifest(p) else None

    step = 0
    trainer_state: dict[str, Any] = {
        "n_forward_passes_since_fired": torch.zeros(cfg.d_sae, device=device),
        "act_freq_scores": torch.zeros(cfg.d_sae, device=device),
        "n_frac_active_tokens": 0,
        "tokens_seen": 0,
    }
    wandb_run_id: str | None = None
    if resume_path is not None:
        step, trainer_state, wandb_run_id = load_checkpoint(
            resume_path,
            sae=sae,
            optimizer=optimizer,
            scheduler=scheduler,
            store=store,
            scaler=scaler,
        )
        # Mover los tensores de estado al device tras load
        for k in ("n_forward_passes_since_fired", "act_freq_scores"):
            trainer_state[k] = trainer_state[k].to(device)
    else:
        logger.info("no hay checkpoint reanudable; entrenamiento desde cero")

    # ---- Estimación del escalador constante (solo si no venía en ckpt) ---
    if scaler.factor is None:
        scaler.estimate(store, cfg.d_in, cfg.n_batches_for_norm_estimate)

    # ---- W&B ------------------------------------------------------------
    wandb_run = init_wandb(cfg, wandb_run_id)
    if wandb_run is not None:
        wandb_run_id = wandb_run.id

    killer = GracefulKiller()
    autocast_ctx: Any = (
        torch.autocast(device_type=device.type, dtype=resolve_dtype(cfg.precision))
        if cfg.precision in ("bf16", "fp16") and device.type in ("cuda", "cpu")
        else torch.autocast(device_type="cpu", enabled=False)
    )

    t_start = time.time()
    last_save = time.time()
    last_log_loss = float("nan")

    def do_save(final: bool = False) -> Path:
        return atomic_save_checkpoint(
            ckpt_root,
            step,
            sae=sae,
            optimizer=optimizer,
            scheduler=scheduler,
            store=store,
            scaler=scaler,
            trainer_state={
                "n_forward_passes_since_fired": trainer_state[
                    "n_forward_passes_since_fired"
                ].cpu(),
                "act_freq_scores": trainer_state["act_freq_scores"].cpu(),
                "n_frac_active_tokens": trainer_state["n_frac_active_tokens"],
                "tokens_seen": trainer_state["tokens_seen"],
                "final": final,
            },
            wandb_run_id=wandb_run_id,
            keep_last_k=cfg.keep_last_k,
        )

    # ---- Loop -----------------------------------------------------------
    while step < cfg.training_steps:
        batch_raw = next(store)  # (train_batch_size_tokens, d_in)
        batch = scaler.scale(batch_raw).to(device=device, dtype=torch.float32)

        dead_mask = (
            trainer_state["n_forward_passes_since_fired"] > cfg.dead_feature_window
        )

        with autocast_ctx:
            out = sae.training_forward_pass(
                TrainStepInput(
                    sae_in=batch,
                    coefficients=sae.get_coefficients(),
                    dead_neuron_mask=dead_mask,
                    n_training_steps=step,
                    is_logging_step=(step % cfg.wandb_log_every == 0),
                )
            )

        loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # --- estadística de disparo / dead-latents ---
        with torch.no_grad():
            feats = out.feature_acts
            if feats.is_sparse:
                feats = feats.to_dense()
            did_fire = (feats > 0).any(dim=0)
            trainer_state["n_forward_passes_since_fired"] += 1
            trainer_state["n_forward_passes_since_fired"][did_fire] = 0
            trainer_state["act_freq_scores"] += (feats > 0).float().sum(0)
            trainer_state["n_frac_active_tokens"] += batch.shape[0]
            trainer_state["tokens_seen"] += batch.shape[0]

        last_log_loss = float(loss.detach().item())
        step += 1

        # --- logging ---
        if wandb_run is not None and step % cfg.wandb_log_every == 0:
            n_active = int(trainer_state["n_frac_active_tokens"])
            density = trainer_state["act_freq_scores"] / max(1, n_active)
            wandb_run.log(
                {
                    "loss/total": last_log_loss,
                    "loss/mse": float(out.losses["mse_loss"].detach().item()),
                    "loss/aux": float(
                        out.losses.get(
                            "auxiliary_reconstruction_loss",
                            torch.tensor(0.0),
                        ).detach().item()
                    ),
                    "sparsity/dead_features": int(dead_mask.sum().item()),
                    "sparsity/log10_density_mean": float(
                        torch.log10(density.clamp(min=1e-10)).mean().item()
                    ),
                    "opt/lr": float(scheduler.get_last_lr()[0]),
                    "throughput/tokens_seen": trainer_state["tokens_seen"],
                    "throughput/tokens_per_sec": trainer_state["tokens_seen"]
                    / max(1e-6, time.time() - t_start),
                },
                step=step,
            )

        # --- evals periódicos ---
        eval_metrics = maybe_run_evals(cfg, step, sae, store, model, scaler)
        if eval_metrics is not None and wandb_run is not None:
            wandb_run.log(
                {f"eval/{k}": v for k, v in eval_metrics.items()},
                step=step,
            )

        # --- checkpoint: por pasos, por tiempo o por señal ---
        need_save = (
            (cfg.save_every_steps > 0 and step % cfg.save_every_steps == 0)
            or (
                cfg.save_every_seconds > 0
                and time.time() - last_save >= cfg.save_every_seconds
            )
            or killer.requested
            or (
                cfg.deadline_hours is not None
                and (time.time() - t_start) / 3600 >= cfg.deadline_hours
            )
        )
        if need_save:
            do_save()
            last_save = time.time()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if killer.requested or (
            cfg.deadline_hours is not None
            and (time.time() - t_start) / 3600 >= cfg.deadline_hours
        ):
            logger.warning("saliendo del bucle por señal/deadline en step=%d", step)
            break

    logger.info("loop finalizado en step=%d loss=%.4f", step, last_log_loss)

    # ---- Guardado final: pliega escalador y norma del decoder -----------
    final_ckpt = do_save(final=True)
    if scaler.factor is not None:
        sae.fold_activation_norm_scaling_factor(scaler.factor)
        sae.cfg.normalize_activations = "none"
    inference_dir = ckpt_root / "final_inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    sae.save_inference_model(str(inference_dir))
    logger.info("modelo de inferencia exportado a %s", inference_dir)

    # ---- Evaluación final ----------------------------------------------
    # El factor de escala ya está plegado en los pesos: evaluar con escalador
    # identidad. Pasar `scaler` aquí escalaría las activaciones dos veces.
    saved_eval_every = cfg.eval_every_steps
    cfg.eval_every_steps = 1
    final_metrics = maybe_run_evals(cfg, 1, sae, store, model, ConstantScaler()) or {}
    cfg.eval_every_steps = saved_eval_every
    (inference_dir / "eval_metrics.json").write_text(
        json.dumps({k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in final_metrics.items()}, indent=2)
    )
    if wandb_run is not None:
        wandb_run.log({f"final/{k}": v for k, v in final_metrics.items()})

    # ---- Upload opcional a HF Hub y a W&B como artifact -----------------
    if cfg.hf_repo_id:
        try:
            from huggingface_hub import HfApi  # type: ignore[import-untyped]

            api = HfApi()
            # Crea el repo si no existe (idempotente); respeta --hf-private.
            api.create_repo(
                repo_id=cfg.hf_repo_id,
                repo_type="model",
                private=cfg.hf_private,
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=str(inference_dir),
                repo_id=cfg.hf_repo_id,
                repo_type="model",
                commit_message=f"Upload SAE (step={step})",
            )
            logger.info(
                "subido a Hugging Face Hub: %s (private=%s)",
                cfg.hf_repo_id, cfg.hf_private,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("upload a HF Hub falló: %s", e)

    if wandb_run is not None:
        try:
            import wandb  # type: ignore[import-untyped]

            art = wandb.Artifact("sae_gpt2_topk", type="model")
            art.add_dir(str(inference_dir))
            wandb_run.log_artifact(art)
        except Exception as e:  # noqa: BLE001
            logger.exception("upload a W&B artifact falló: %s", e)
        wandb_run.finish()

    logger.info("training completo; artefacto final en %s", inference_dir)
    return final_ckpt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    # Solo exponemos los flags que se tocan en la práctica; el resto se edita
    # arriba en TrainConfig (una única fuente de verdad legible).
    p.add_argument("--checkpoint-dir", type=str, default=cfg.checkpoint_dir)
    p.add_argument("--resume", type=str, default=cfg.resume)
    p.add_argument("--total-tokens", type=int, default=cfg.total_training_tokens)
    p.add_argument("--expansion-factor", type=int, default=cfg.expansion_factor)
    p.add_argument("--k", type=int, default=cfg.k)
    p.add_argument("--aux-coef", type=float, default=cfg.aux_loss_coefficient,
                   help="coeficiente AuxK (0 desactiva el rescate de latentes muertas)")
    p.add_argument("--context-size", type=int, default=cfg.context_size)
    p.add_argument("--batch-tokens", type=int, default=cfg.train_batch_size_tokens)
    p.add_argument("--store-batch-prompts", type=int, default=cfg.store_batch_size_prompts)
    p.add_argument("--n-buffer", type=int, default=cfg.n_batches_in_buffer)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--precision", type=str, default=cfg.precision, choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", type=str, default=cfg.device)
    p.add_argument("--dataset", type=str, default=cfg.dataset_path)
    p.add_argument("--dataset-config", type=str, default=cfg.dataset_config)
    p.add_argument("--save-every-steps", type=int, default=cfg.save_every_steps)
    p.add_argument("--save-every-seconds", type=int, default=cfg.save_every_seconds)
    p.add_argument("--eval-every-steps", type=int, default=cfg.eval_every_steps)
    p.add_argument("--wandb-project", type=str, default=cfg.wandb_project)
    p.add_argument("--wandb-entity", type=str, default=cfg.wandb_entity)
    p.add_argument("--wandb-mode", type=str, default=cfg.wandb_mode,
                   choices=["online", "offline", "disabled"])
    p.add_argument("--hf-repo-id", type=str, default=cfg.hf_repo_id)
    p.add_argument("--hf-private", action="store_true")
    p.add_argument("--deadline-hours", type=float, default=cfg.deadline_hours)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)

    cfg.checkpoint_dir = a.checkpoint_dir
    cfg.resume = a.resume
    cfg.total_training_tokens = a.total_tokens
    cfg.expansion_factor = a.expansion_factor
    cfg.k = a.k
    cfg.aux_loss_coefficient = a.aux_coef
    cfg.context_size = a.context_size
    cfg.train_batch_size_tokens = a.batch_tokens
    cfg.store_batch_size_prompts = a.store_batch_prompts
    cfg.n_batches_in_buffer = a.n_buffer
    cfg.lr = a.lr
    cfg.precision = a.precision
    cfg.device = a.device
    cfg.dataset_path = a.dataset
    cfg.dataset_config = a.dataset_config
    cfg.save_every_steps = a.save_every_steps
    cfg.save_every_seconds = a.save_every_seconds
    cfg.eval_every_steps = a.eval_every_steps
    cfg.wandb_project = a.wandb_project
    cfg.wandb_entity = a.wandb_entity
    cfg.wandb_mode = a.wandb_mode
    cfg.hf_repo_id = a.hf_repo_id
    cfg.hf_private = a.hf_private
    cfg.deadline_hours = a.deadline_hours
    cfg.seed = a.seed
    cfg.smoke = a.smoke
    if cfg.smoke:
        cfg = apply_smoke_overrides(cfg)
    return cfg


def main() -> int:
    cfg = parse_args()
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
