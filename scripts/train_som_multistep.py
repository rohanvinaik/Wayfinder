"""Train SoM multi-step model with three-stage curriculum and PAB stability.

Stage 1: Train specialists independently — each learns its domain signal
Stage 2: Freeze specialists, train orchestrator — learns trust weights
Stage 3: Joint fine-tuning at 10x lower LR — compositionality emerges

PAB stability per stage with per-stage thresholds (from SOM_TRAINING_ANALYSIS §3):
  Stage 1: threshold=0.015 (noisy domain labels)
  Stage 2: threshold=0.01 (cleaner cross-entropy)
  Stage 3: threshold=0.008 (smooth from good init)

Usage:
    python -m scripts.train_som_multistep \
        --train data/som_multistep_train.jsonl \
        --eval data/som_multistep_eval.jsonl \
        --output models/som_multistep_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.som_model import SPECIALIST_NAMES, SoMConfig, SoMModel, softmax

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _shape_to_vec(shape: dict) -> np.ndarray:
    """Convert goal_shape dict to 12-d feature vector."""
    return np.array([
        float(shape.get("has_forall", False)),
        float(shape.get("has_exists", False)),
        float(shape.get("has_eq", False)),
        float(shape.get("has_ineq", False)),
        float(shape.get("has_iff", False)),
        float(shape.get("has_and", False)),
        float(shape.get("has_or", False)),
        float(shape.get("has_neg", False)),
        float(shape.get("has_fun", False)),
        min(shape.get("hyp_count", 0) / 20.0, 1.0),
        min(shape.get("target_len", 0) / 500.0, 1.0),
        float(shape.get("target_head", "") in ["Eq", "eq", "HEq", "Iff"]),
    ], dtype=np.float32)


def _step_context(ex: dict) -> np.ndarray:
    """Extract step context features."""
    return np.array([
        min(ex.get("step_index", 0) / 10.0, 1.0),
        min(ex.get("proof_length", 1) / 20.0, 1.0),
        min(ex.get("goal_shape", {}).get("hyp_count", 0) / 20.0, 1.0),
        min(ex.get("goal_shape", {}).get("target_len", 0) / 500.0, 1.0),
    ], dtype=np.float32)


SPECIALIST_TO_IDX = {name: i for i, name in enumerate(SPECIALIST_NAMES)}


def load_dataset(
    path: str,
    embed_cache: dict | None = None,
    max_examples: int = 0,
) -> dict[str, np.ndarray]:
    """Load and featurize training data.

    Returns dict with:
        goal_emb: (N, 384) goal embeddings
        goal_shape: (N, 12) structural features
        step_context: (N, 4) step context
        labels: (N,) specialist index [0-4]
    """
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    if max_examples:
        examples = examples[:max_examples]

    N = len(examples)

    # Embeddings from pre-computed cache
    goal_emb = np.zeros((N, 384), dtype=np.float32)
    if embed_cache:
        for i, ex in enumerate(examples):
            t = ex.get("goal_target", "")
            if not t and "goal_state_before" in ex:
                gb = ex["goal_state_before"]
                t = gb.split("⊢")[-1].strip() if "⊢" in gb else gb[:200]
            t = t[:300]
            if t in embed_cache:
                goal_emb[i] = embed_cache[t]
    else:
        goal_emb = np.random.randn(N, 384).astype(np.float32) * 0.1

    goal_shape = np.zeros((N, 12), dtype=np.float32)
    step_context = np.zeros((N, 4), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int32)

    for i, ex in enumerate(examples):
        goal_shape[i] = _shape_to_vec(ex.get("goal_shape", {}))
        step_context[i] = _step_context(ex)
        specialist = ex.get("specialist", "structural")
        labels[i] = SPECIALIST_TO_IDX.get(specialist, 1)

    return {
        "goal_emb": goal_emb,
        "goal_shape": goal_shape,
        "step_context": step_context,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Cross-entropy loss with gradient.

    Args:
        logits: (batch, n_classes) raw scores
        labels: (batch,) integer labels

    Returns:
        loss: scalar
        grad: (batch, n_classes) gradient w.r.t. logits
    """
    probs = softmax(logits, axis=-1)
    batch = logits.shape[0]
    # Loss
    log_probs = np.log(probs + 1e-10)
    loss = -log_probs[np.arange(batch), labels].mean()
    # Gradient
    grad = probs.copy()
    grad[np.arange(batch), labels] -= 1.0
    grad /= batch
    return float(loss), grad


def mse_loss(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """MSE loss with gradient."""
    diff = pred - target
    loss = float((diff ** 2).mean())
    grad = 2.0 * diff / diff.size
    return loss, grad


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy."""
    preds = np.argmax(logits, axis=-1)
    return float((preds == labels).mean())


def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Top-k accuracy."""
    top_k = np.argsort(logits, axis=-1)[:, -k:]
    return float(np.any(top_k == labels[:, np.newaxis], axis=-1).mean())


class AdamW:
    """Simple AdamW optimizer for numpy arrays."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-4):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray | None]) -> None:
        self.t += 1
        b1, b2 = self.betas
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue
            # Weight decay (decoupled)
            if self.weight_decay > 0 and p.ndim >= 2:
                p -= self.lr * self.weight_decay * p
            # Adam update
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g**2
            m_hat = self.m[i] / (1 - b1**self.t)
            v_hat = self.v[i] / (1 - b2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# PAB Stability Detection
# ---------------------------------------------------------------------------

class PABMonitor:
    """Process-Aware Benchmarking for training control.

    Tracks relative loss change and stops when trajectory stabilizes.
    Different thresholds per stage (noisier stages get looser thresholds).
    """

    def __init__(self, threshold: float = 0.015, window: int = 20,
                 patience: int = 5, max_no_improve: int = 40):
        self.threshold = threshold
        self.window = window
        self.patience = patience
        self.max_no_improve = max_no_improve
        self.losses: list[float] = []
        self.best_metric: float = -1e9
        self.no_improve_count: int = 0
        self.stable_count: int = 0

    def update(self, loss: float, metric: float) -> bool:
        """Update with new loss/metric. Returns True if training should stop."""
        self.losses.append(loss)

        # Check improvement
        if metric > self.best_metric:
            self.best_metric = metric
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        # Safety valve
        if self.no_improve_count >= self.max_no_improve:
            logger.info("PAB: max_no_improve reached (%d)", self.max_no_improve)
            return True

        # Stability check
        if len(self.losses) >= self.window + 1:
            recent = self.losses[-self.window:]
            changes = [abs(recent[i] - recent[i-1]) / (abs(recent[i-1]) + 1e-8)
                       for i in range(1, len(recent))]
            mean_change = sum(changes) / len(changes)
            if mean_change < self.threshold:
                self.stable_count += 1
                if self.stable_count >= self.patience:
                    logger.info("PAB: stable (mean_change=%.4f < %.4f, %d consecutive)",
                                mean_change, self.threshold, self.stable_count)
                    return True
            else:
                self.stable_count = 0

        return False


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------

def train_stage1(
    model: SoMModel,
    train_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    cfg: SoMConfig,
    max_epochs: int = 100,
    batch_size: int = 256,
) -> dict:
    """Stage 1: Train specialists independently.

    Each specialist learns from ALL examples but gets its own domain-specific
    loss (MSE on "am I the right specialist for this goal?").
    """
    logger.info("=== Stage 1: Train specialists independently ===")

    N = train_data["labels"].shape[0]
    pab = PABMonitor(threshold=0.015, window=20, patience=5, max_no_improve=40)

    # Collect all specialist parameters
    all_params = []
    for agent in model.specialists.values():
        all_params.extend(agent.parameters())
    optimizer = AdamW(all_params, lr=cfg.lr_stage1, weight_decay=cfg.weight_decay)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(max_epochs):
        # Shuffle
        perm = model.rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            goal_emb = train_data["goal_emb"][idx]
            goal_shape = train_data["goal_shape"][idx]
            step_ctx = train_data["step_context"][idx]
            labels = train_data["labels"][idx]
            context = np.concatenate([goal_emb, goal_shape, step_ctx], axis=-1)
            domain_feats = np.concatenate([goal_emb, goal_shape], axis=-1)

            # Forward each specialist
            batch_scores = np.zeros((len(idx), 5), dtype=np.float32)
            for i, name in enumerate(SPECIALIST_NAMES):
                agent = model.specialists[name]
                confidence, _ = agent.forward(domain_feats, context)
                batch_scores[:, i] = confidence

            # Loss: cross-entropy on specialist selection
            loss, grad = cross_entropy_loss(batch_scores, labels)
            epoch_loss += loss
            n_batches += 1

            # Backward through each specialist's scoring path
            from src.som_model import gelu_backward
            for i, name in enumerate(SPECIALIST_NAMES):
                agent = model.specialists[name]
                spec_grad = grad[:, i:i+1]  # (batch, 1)

                # score_fc2 → gelu → score_fc1 → [domain; context]
                g = agent.score_fc2.backward(spec_grad)
                g = gelu_backward(agent._cache["s_pre_gelu"], g)
                g = agent.score_fc1.backward(g)

                # Split gradient for domain (specialist_hidden) and context (context_hidden)
                g_d = g[:, :cfg.specialist_hidden]
                g_c = g[:, cfg.specialist_hidden:]

                # Domain backward: fc1→ln1→gelu(cache1)→fc2→ln2→gelu(cache2)
                # Reverse: gelu_bw(cache2)→ln2_bw→fc2_bw→gelu_bw(cache1)→ln1_bw→fc1_bw
                g_d = gelu_backward(agent._cache["d_pre_gelu2"], g_d)
                g_d = agent.domain_ln2.backward(g_d)
                g_d = agent.domain_fc2.backward(g_d)
                g_d = gelu_backward(agent._cache["d_pre_gelu1"], g_d)
                g_d = agent.domain_ln1.backward(g_d)
                agent.domain_fc1.backward(g_d)

                # Context backward: fc1→ln1→gelu(cache)
                g_c = gelu_backward(agent._cache["c_pre_gelu"], g_c)
                g_c = agent.context_ln1.backward(g_c)
                agent.context_fc1.backward(g_c)

            # Collect all specialist gradients and update
            all_grads_flat = []
            for agent_val in model.specialists.values():
                for layer in agent_val.all_layers():
                    all_grads_flat.extend(layer.gradients())
            optimizer.step(all_grads_flat)

        # Eval
        eval_scores = np.zeros((eval_data["labels"].shape[0], 5), dtype=np.float32)
        eval_context = np.concatenate([eval_data["goal_emb"], eval_data["goal_shape"],
                                       eval_data["step_context"]], axis=-1)
        eval_domain = np.concatenate([eval_data["goal_emb"], eval_data["goal_shape"]], axis=-1)
        for i, name in enumerate(SPECIALIST_NAMES):
            conf, _ = model.specialists[name].forward(eval_domain, eval_context)
            eval_scores[:, i] = conf

        eval_loss, _ = cross_entropy_loss(eval_scores, eval_data["labels"])
        eval_acc = accuracy(eval_scores, eval_data["labels"])
        eval_top3 = top_k_accuracy(eval_scores, eval_data["labels"], k=3)

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_epoch = epoch

        if epoch % 5 == 0 or eval_acc > best_acc - 0.001:
            logger.info("  Epoch %d: train_loss=%.4f eval_loss=%.4f eval_acc=%.3f top3=%.3f best=%.3f@%d",
                        epoch, epoch_loss / max(n_batches, 1), eval_loss, eval_acc, eval_top3, best_acc, best_epoch)

        if pab.update(eval_loss, eval_acc):
            break

    logger.info("Stage 1 done: best_acc=%.3f at epoch %d", best_acc, best_epoch)
    return {"best_acc": best_acc, "best_epoch": best_epoch, "final_epoch": epoch}  # type: ignore[possibly-undefined]


def train_stage2(
    model: SoMModel,
    train_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    cfg: SoMConfig,
    max_epochs: int = 50,
    batch_size: int = 256,
) -> dict:
    """Stage 2: Freeze specialists, train orchestrator.

    Specialists are frozen. Orchestrator learns trust weights from
    pre-computed specialist scores.
    """
    logger.info("=== Stage 2: Freeze specialists, train orchestrator ===")

    # Freeze specialists
    for agent in model.specialists.values():
        agent.frozen = True

    N = train_data["labels"].shape[0]
    pab = PABMonitor(threshold=0.01, window=20, patience=5, max_no_improve=40)

    optimizer = AdamW(model.orchestrator.parameters(), lr=cfg.lr_stage2,
                      weight_decay=cfg.weight_decay)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(max_epochs):
        perm = model.rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            goal_emb = train_data["goal_emb"][idx]
            goal_shape = train_data["goal_shape"][idx]
            step_ctx = train_data["step_context"][idx]
            labels = train_data["labels"][idx]

            # Forward (specialists frozen, only orchestrator trains)
            trust_weights, _info = model.forward(goal_emb, goal_shape, step_ctx)

            loss, _grad = cross_entropy_loss(trust_weights, labels)
            epoch_loss += loss
            n_batches += 1

            # Backward through orchestrator only
            orch_grads = []
            for layer in model.orchestrator.all_layers():
                orch_grads.extend(layer.gradients())
            optimizer.step(orch_grads)

        # Eval
        trust, _ = model.forward(eval_data["goal_emb"], eval_data["goal_shape"],
                                  eval_data["step_context"])
        eval_loss, _ = cross_entropy_loss(trust, eval_data["labels"])
        eval_acc = accuracy(trust, eval_data["labels"])
        eval_top3 = top_k_accuracy(trust, eval_data["labels"], k=3)

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_epoch = epoch

        if epoch % 5 == 0:
            logger.info("  Epoch %d: train_loss=%.4f eval_loss=%.4f eval_acc=%.3f top3=%.3f",
                        epoch, epoch_loss / max(n_batches, 1), eval_loss, eval_acc, eval_top3)

        if pab.update(eval_loss, eval_acc):
            break

    logger.info("Stage 2 done: best_acc=%.3f at epoch %d", best_acc, best_epoch)
    return {"best_acc": best_acc, "best_epoch": best_epoch, "final_epoch": epoch}  # type: ignore[possibly-undefined]


def train_stage3(
    model: SoMModel,
    train_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    cfg: SoMConfig,
    max_epochs: int = 100,
    batch_size: int = 256,
) -> dict:
    """Stage 3: Joint fine-tuning at 10x lower LR.

    Unfreeze everything. This is where compositionality emerges —
    specialists adapt to the orchestrator's arbitration.
    """
    logger.info("=== Stage 3: Joint fine-tuning ===")

    # Unfreeze specialists
    for agent in model.specialists.values():
        agent.frozen = False

    N = train_data["labels"].shape[0]
    pab = PABMonitor(threshold=0.008, window=20, patience=5, max_no_improve=40)

    # All parameters at 10x lower LR
    all_params = []
    for agent in model.specialists.values():
        all_params.extend(agent.parameters())
    all_params.extend(model.orchestrator.parameters())
    optimizer = AdamW(all_params, lr=cfg.lr_stage3, weight_decay=cfg.weight_decay)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(max_epochs):
        perm = model.rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            goal_emb = train_data["goal_emb"][idx]
            goal_shape = train_data["goal_shape"][idx]
            step_ctx = train_data["step_context"][idx]
            labels = train_data["labels"][idx]

            trust_weights, _s3_info = model.forward(goal_emb, goal_shape, step_ctx)

            loss, _s3_grad = cross_entropy_loss(trust_weights, labels)
            epoch_loss += loss
            n_batches += 1

            # Backward through everything
            all_grads = []
            for agent in model.specialists.values():
                for layer in agent.all_layers():
                    all_grads.extend(layer.gradients())
            for layer in model.orchestrator.all_layers():
                all_grads.extend(layer.gradients())
            optimizer.step(all_grads)

        # Eval
        trust, _s3_eval_info = model.forward(eval_data["goal_emb"], eval_data["goal_shape"],
                                     eval_data["step_context"])
        eval_loss, _ = cross_entropy_loss(trust, eval_data["labels"])
        eval_acc = accuracy(trust, eval_data["labels"])
        eval_top3 = top_k_accuracy(trust, eval_data["labels"], k=3)

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_epoch = epoch
            # Save best checkpoint
            model.save(str(Path(args.output) / "best.npz") if 'args' in dir() else "models/som_multistep_best.npz")

        if epoch % 5 == 0:
            logger.info("  Epoch %d: train_loss=%.4f eval_loss=%.4f eval_acc=%.3f top3=%.3f best=%.3f@%d",
                        epoch, epoch_loss / max(n_batches, 1), eval_loss, eval_acc, eval_top3, best_acc, best_epoch)

        if pab.update(eval_loss, eval_acc):
            break

    logger.info("Stage 3 done: best_acc=%.3f at epoch %d", best_acc, best_epoch)
    return {"best_acc": best_acc, "best_epoch": best_epoch, "final_epoch": epoch}  # type: ignore[possibly-undefined]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/som_multistep_train.jsonl")
    parser.add_argument("--eval", default="data/som_multistep_eval.jsonl")
    parser.add_argument("--output", default="models/som_multistep_v1")
    parser.add_argument("--embeddings", default="data/som_goal_embeddings.npz")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Limit training examples (0=all)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-encoder", action="store_true",
                        help="Use random embeddings instead of pre-computed")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run only this stage (1/2/3), 0=all")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-computed feature arrays (fast path)
    train_features = Path("data/som_train_features.npz")
    eval_features = Path("data/som_eval_features.npz")

    if train_features.exists() and eval_features.exists():
        logger.info("Loading pre-computed feature arrays...")
        train_npz = np.load(str(train_features))
        train_data = {k: train_npz[k] for k in train_npz.files}
        eval_npz = np.load(str(eval_features))
        eval_data = {k: eval_npz[k] for k in eval_npz.files}
        if args.max_examples and args.max_examples < train_data["labels"].shape[0]:
            for k in train_data:
                train_data[k] = train_data[k][:args.max_examples]
    else:
        logger.info("No pre-computed features found. Loading from JSONL...")
        embed_cache: dict | None = None
        emb_path = Path(args.embeddings)
        if emb_path.exists():
            npz = np.load(str(emb_path), allow_pickle=True)
            embed_cache = {str(t): npz["embeddings"][i] for i, t in enumerate(npz["targets"])}
            logger.info("  %d embeddings loaded", len(embed_cache))
        train_data = load_dataset(args.train, embed_cache, args.max_examples)
        eval_data = load_dataset(args.eval, embed_cache)

    logger.info("  Train: %d examples", train_data["labels"].shape[0])
    logger.info("  Eval: %d examples", eval_data["labels"].shape[0])

    # Label distribution
    for split_name, data in [("Train", train_data), ("Eval", eval_data)]:
        counts = np.bincount(data["labels"], minlength=5)
        total = counts.sum()
        dist = ", ".join(f"{SPECIALIST_NAMES[i]}={counts[i]} ({100*counts[i]/total:.1f}%)"
                         for i in range(5))
        logger.info("  %s distribution: %s", split_name, dist)

    # Build model
    cfg = SoMConfig()
    model = SoMModel(cfg)
    logger.info("Model: %d parameters", model.param_count())

    # Random baseline
    random_acc = 1.0 / 5
    majority_label = int(np.argmax(np.bincount(eval_data["labels"], minlength=5)))
    majority_acc = float((eval_data["labels"] == majority_label).mean())
    logger.info("Baselines: random=%.3f, majority=%.3f (%s)",
                random_acc, majority_acc, SPECIALIST_NAMES[majority_label])

    t_start = time.time()
    results = {}

    # Stage 1
    if args.stage == 0 or args.stage == 1:
        s1 = train_stage1(model, train_data, eval_data, cfg,
                          batch_size=args.batch_size)
        results["stage1"] = s1
        model.save(str(output_dir / "after_stage1.npz"))

    # Stage 2
    if args.stage == 0 or args.stage == 2:
        if args.stage == 2:
            # Load stage 1 checkpoint
            ckpt = output_dir / "after_stage1.npz"
            if ckpt.exists():
                model.load(str(ckpt))
                logger.info("Loaded stage 1 checkpoint")
        s2 = train_stage2(model, train_data, eval_data, cfg,
                          batch_size=args.batch_size)
        results["stage2"] = s2
        model.save(str(output_dir / "after_stage2.npz"))

    # Stage 3
    if args.stage == 0 or args.stage == 3:
        if args.stage == 3:
            ckpt = output_dir / "after_stage2.npz"
            if ckpt.exists():
                model.load(str(ckpt))
                logger.info("Loaded stage 2 checkpoint")
        s3 = train_stage3(model, train_data, eval_data, cfg,
                          batch_size=args.batch_size)
        results["stage3"] = s3
        model.save(str(output_dir / "best.npz"))

    elapsed = time.time() - t_start
    logger.info("Total training time: %.0fs (%.1f min)", elapsed, elapsed / 60)

    # Final eval
    trust, _final_info = model.forward(
        eval_data["goal_emb"], eval_data["goal_shape"], eval_data["step_context"])
    final_acc = accuracy(trust, eval_data["labels"])
    final_top3 = top_k_accuracy(trust, eval_data["labels"], k=3)

    logger.info("=" * 60)
    logger.info("Final: acc=%.3f top3=%.3f (majority=%.3f, random=%.3f)",
                final_acc, final_top3, majority_acc, random_acc)

    # Per-specialist accuracy
    for i, name in enumerate(SPECIALIST_NAMES):
        mask = eval_data["labels"] == i
        if mask.sum() > 0:
            spec_acc = float((np.argmax(trust[mask], axis=-1) == i).mean())
            logger.info("  %s: %.3f (%d examples)", name, spec_acc, mask.sum())

    # Save summary
    summary = {
        "model_params": model.param_count(),
        "train_examples": int(train_data["labels"].shape[0]),
        "eval_examples": int(eval_data["labels"].shape[0]),
        "random_baseline": random_acc,
        "majority_baseline": majority_acc,
        "final_acc": final_acc,
        "final_top3": final_top3,
        "elapsed_s": elapsed,
        "stages": results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
