"""Society of Mind model for multi-step proof tactic selection.

Architecture adapted from Yami chess SoM (SOM_TRAINING_ANALYSIS):
- 5 specialist agents (rewrite, structural, solver, apply, closer)
- 1 orchestrator with trust-weight softmax
- Three-stage training: specialists → orchestrator → joint fine-tuning

Each specialist sees:
  - Domain signals at HIGH capacity (specialist-specific goal features)
  - Full goal context at LOW capacity (shared embedding)

The orchestrator sees:
  - Goal context + all specialist confidence scores
  - Outputs softmax trust weights over specialists

Total parameters: ~1.2M (5 × ~210K + ~72K orchestrator)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Use numpy for all numerical computation (project rule)


@dataclass
class SoMConfig:
    """Configuration for the SoM model."""
    # Embedding
    goal_emb_dim: int = 384        # MiniLM output dimension
    goal_shape_dim: int = 12       # Structural features (from extract_goal_shape)
    step_context_dim: int = 4      # step_index, proof_length, hyp_count, target_len

    # Specialist architecture
    n_specialists: int = 5
    domain_hidden: int = 256       # Domain encoder hidden (high capacity)
    context_hidden: int = 128      # Context encoder hidden (low capacity)
    specialist_hidden: int = 128   # Specialist scoring hidden

    # Orchestrator architecture
    orch_hidden: int = 128
    override_threshold: float = 0.8
    override_boost: float = 5.0

    # Training
    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-3
    lr_stage3: float = 1e-4        # 10x lower for joint fine-tuning
    weight_decay: float = 1e-4
    aux_weight: float = 0.05       # Auxiliary loss weight


# ---------------------------------------------------------------------------
# NumPy-based layers
# ---------------------------------------------------------------------------

def _glorot_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Glorot uniform initialization."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)


def _zeros(shape: int | tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


class Linear:
    """Dense layer with optional bias."""
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator, bias: bool = True):
        self.W = _glorot_init(in_dim, out_dim, rng)
        self.b = _zeros(out_dim) if bias else None
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Cached for backprop
        self._input: np.ndarray | None = None
        self._grad_W: np.ndarray | None = None
        self._grad_b: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self._input is not None
        self._grad_W = self._input.T @ grad_out
        if self.b is not None:
            self._grad_b = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def parameters(self) -> list[np.ndarray]:
        if self.b is not None:
            return [self.W, self.b]
        return [self.W]

    def gradients(self) -> list[np.ndarray | None]:
        if self.b is not None:
            return [self._grad_W, self._grad_b]
        return [self._grad_W]


class LayerNorm:
    """Layer normalization."""
    def __init__(self, dim: int):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = _zeros(dim)
        self.dim = dim
        self._x_norm: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._grad_gamma: np.ndarray | None = None
        self._grad_beta: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        self._std = x.std(axis=-1, keepdims=True) + 1e-5
        self._x_norm = (x - mean) / self._std
        return self.gamma * self._x_norm + self.beta  # type: ignore[operator]

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self._x_norm is not None and self._std is not None
        self._grad_gamma = (grad_out * self._x_norm).sum(axis=0)
        self._grad_beta = grad_out.sum(axis=0)
        dx_norm = grad_out * self.gamma
        dx = (1.0 / self._std) * (dx_norm - dx_norm.mean(axis=-1, keepdims=True)
              - self._x_norm * (dx_norm * self._x_norm).mean(axis=-1, keepdims=True))
        return dx

    def parameters(self) -> list[np.ndarray]:
        return [self.gamma, self.beta]

    def gradients(self) -> list[np.ndarray | None]:
        return [self._grad_gamma, self._grad_beta]


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def gelu_backward(x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
    """GELU backward pass."""
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner**2
    d_inner = c * (1.0 + 3 * 0.044715 * x**2)
    grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
    return grad_out * grad


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Specialist Agent
# ---------------------------------------------------------------------------

class SpecialistAgent:
    """One specialist in the SoM ensemble.

    Dual-path architecture (Decision 2 from SOM_TRAINING_ANALYSIS):
    - Domain encoder: specialist-specific features at HIGH capacity
    - Context encoder: full goal embedding at LOW capacity
    - Scoring head: produces confidence score for "should I handle this?"

    Each specialist is forced to learn its domain because it sees domain-relevant
    inputs at high capacity. The full context path prevents catastrophic blindness
    but doesn't dominate.
    """

    def __init__(
        self,
        name: str,
        domain_dim: int,
        context_dim: int,
        cfg: SoMConfig,
        rng: np.random.Generator,
    ):
        self.name = name

        # Domain encoder (HIGH capacity) — this is what makes it a specialist
        self.domain_fc1 = Linear(domain_dim, cfg.domain_hidden, rng)
        self.domain_ln1 = LayerNorm(cfg.domain_hidden)
        self.domain_fc2 = Linear(cfg.domain_hidden, cfg.specialist_hidden, rng)
        self.domain_ln2 = LayerNorm(cfg.specialist_hidden)

        # Context encoder (LOW capacity) — prevents blindness
        self.context_fc1 = Linear(context_dim, cfg.context_hidden, rng)
        self.context_ln1 = LayerNorm(cfg.context_hidden)

        # Scoring head: domain + context → confidence
        score_in = cfg.specialist_hidden + cfg.context_hidden
        self.score_fc1 = Linear(score_in, cfg.specialist_hidden, rng)
        self.score_fc2 = Linear(cfg.specialist_hidden, 1, rng)

        # Auxiliary head: predict proof progress (weak supervisory signal)
        self.aux_fc = Linear(cfg.specialist_hidden, 1, rng)

        self.frozen = False

        # Cache for backprop
        self._cache: dict = {}

    def forward(self, domain_feats: np.ndarray, context_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass.

        Returns:
            confidence: (batch,) specialist confidence scores
            aux_pred: (batch,) auxiliary prediction (proof progress)
        """
        # Domain path (high capacity)
        d = self.domain_fc1.forward(domain_feats)
        d = self.domain_ln1.forward(d)
        self._cache["d_pre_gelu1"] = d.copy()
        d = gelu(d)
        d = self.domain_fc2.forward(d)
        d = self.domain_ln2.forward(d)
        self._cache["d_pre_gelu2"] = d.copy()
        d = gelu(d)

        # Context path (low capacity)
        c = self.context_fc1.forward(context_feats)
        c = self.context_ln1.forward(c)
        self._cache["c_pre_gelu"] = c.copy()
        c = gelu(c)

        # Combine and score
        combined = np.concatenate([d, c], axis=-1)
        s = self.score_fc1.forward(combined)
        self._cache["s_pre_gelu"] = s.copy()
        s = gelu(s)
        confidence = self.score_fc2.forward(s).squeeze(-1)

        # Auxiliary
        aux_pred = self.aux_fc.forward(d).squeeze(-1)
        aux_pred = np.tanh(aux_pred)

        return confidence, aux_pred

    def all_layers(self) -> list:
        return [
            self.domain_fc1, self.domain_ln1, self.domain_fc2, self.domain_ln2,
            self.context_fc1, self.context_ln1,
            self.score_fc1, self.score_fc2, self.aux_fc,
        ]

    def parameters(self) -> list[np.ndarray]:
        params = []
        for layer in self.all_layers():
            params.extend(layer.parameters())
        return params

    def param_count(self) -> int:
        return sum(p.size for p in self.parameters())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Orchestrator that learns trust weights over specialists.

    Input: goal context + all specialist confidence scores
    Output: softmax trust weights (which specialist to invoke)

    Decision 3 (Override): When override > threshold for any specialist,
    that specialist's trust logit gets boosted. Handles cases where one
    specialist should clearly dominate.

    Decision 4 (Trust as softmax): Trust weights sum to 1.0 and naturally
    compete. Override as logit boost preserves gradient flow.
    """

    def __init__(self, cfg: SoMConfig, rng: np.random.Generator):
        self.cfg = cfg

        # Context encoder
        ctx_in = cfg.goal_emb_dim + cfg.goal_shape_dim + cfg.step_context_dim
        self.ctx_fc1 = Linear(ctx_in, cfg.orch_hidden, rng)
        self.ctx_ln1 = LayerNorm(cfg.orch_hidden)
        self.ctx_fc2 = Linear(cfg.orch_hidden, cfg.orch_hidden // 2, rng)

        # Trust head: context + specialist scores → logits
        trust_in = cfg.orch_hidden // 2 + cfg.n_specialists
        self.trust_fc1 = Linear(trust_in, cfg.orch_hidden // 2, rng)
        self.trust_fc2 = Linear(cfg.orch_hidden // 2, cfg.n_specialists, rng)

        # Override head: sigmoid per specialist
        self.override_fc = Linear(cfg.orch_hidden // 2, cfg.n_specialists, rng)

        # Auxiliary: predict overall proof outcome
        self.outcome_fc1 = Linear(cfg.orch_hidden // 2, 32, rng)
        self.outcome_fc2 = Linear(32, 1, rng)

        self._cache: dict = {}

    def forward(
        self,
        goal_context: np.ndarray,
        specialist_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass.

        Args:
            goal_context: (batch, ctx_dim) concatenated goal features
            specialist_scores: (batch, n_specialists) confidence from each specialist

        Returns:
            trust_weights: (batch, n_specialists) softmax trust weights
            override_flags: (batch, n_specialists) sigmoid override scores
            outcome_pred: (batch,) auxiliary outcome prediction
        """
        # Encode context
        c = self.ctx_fc1.forward(goal_context)
        c = self.ctx_ln1.forward(c)
        self._cache["c_pre_gelu1"] = c.copy()
        c = gelu(c)
        c = self.ctx_fc2.forward(c)
        self._cache["c_pre_gelu2"] = c.copy()
        c = gelu(c)

        # Trust weights
        trust_in = np.concatenate([c, specialist_scores], axis=-1)
        t = self.trust_fc1.forward(trust_in)
        self._cache["t_pre_gelu"] = t.copy()
        t = gelu(t)
        trust_logits = self.trust_fc2.forward(t)

        # Override mechanism (Decision 3)
        override_logits = self.override_fc.forward(c)
        override_flags = 1.0 / (1.0 + np.exp(-override_logits))  # sigmoid

        # Apply override boost to trust logits
        boost_mask = (override_flags > self.cfg.override_threshold).astype(np.float32)
        trust_logits = trust_logits + boost_mask * self.cfg.override_boost

        trust_weights = softmax(trust_logits, axis=-1)

        # Auxiliary outcome prediction
        o = self.outcome_fc1.forward(c)
        self._cache["o_pre_gelu"] = o.copy()
        o = gelu(o)
        outcome_pred = self.outcome_fc2.forward(o).squeeze(-1)
        outcome_pred = np.tanh(outcome_pred)

        return trust_weights, override_flags, outcome_pred

    def all_layers(self) -> list:
        return [
            self.ctx_fc1, self.ctx_ln1, self.ctx_fc2,
            self.trust_fc1, self.trust_fc2,
            self.override_fc,
            self.outcome_fc1, self.outcome_fc2,
        ]

    def parameters(self) -> list[np.ndarray]:
        params = []
        for layer in self.all_layers():
            params.extend(layer.parameters())
        return params

    def param_count(self) -> int:
        return sum(p.size for p in self.parameters())


# ---------------------------------------------------------------------------
# Full SoM Model
# ---------------------------------------------------------------------------

SPECIALIST_NAMES = ["rewrite", "structural", "solver", "apply", "closer"]

# Domain-specific feature dimensions for each specialist
# Each specialist gets goal_shape features relevant to its domain
# at HIGH capacity, plus the full goal embedding at LOW capacity
DOMAIN_DIMS = {
    "rewrite": 384 + 12,     # goal_emb + shape (equality, iff focus)
    "structural": 384 + 12,  # goal_emb + shape (forall, exists, connective focus)
    "solver": 384 + 12,      # goal_emb + shape (inequality, numeric focus)
    "apply": 384 + 12,       # goal_emb + shape (type matching focus)
    "closer": 384 + 12,      # goal_emb + shape (simplicity focus)
}


class SoMModel:
    """Complete Society of Mind model for tactic family selection.

    Forward pass:
    1. Each specialist scores the goal from its domain perspective
    2. Orchestrator combines specialist scores with goal context
    3. Softmax trust weights select the winning specialist

    The output is a distribution over 5 specialist families.
    At inference, we invoke the tactic family of the highest-trust specialist.
    """

    def __init__(self, cfg: SoMConfig | None = None, seed: int = 42):
        self.cfg = cfg or SoMConfig()
        self.rng = np.random.default_rng(seed)

        # Build specialists
        context_dim = self.cfg.goal_emb_dim + self.cfg.goal_shape_dim + self.cfg.step_context_dim
        self.specialists: dict[str, SpecialistAgent] = {}
        for name in SPECIALIST_NAMES:
            domain_dim = DOMAIN_DIMS[name]
            self.specialists[name] = SpecialistAgent(
                name=name,
                domain_dim=domain_dim,
                context_dim=context_dim,
                cfg=self.cfg,
                rng=self.rng,
            )

        # Build orchestrator
        self.orchestrator = Orchestrator(self.cfg, self.rng)

    def forward(
        self,
        goal_emb: np.ndarray,
        goal_shape: np.ndarray,
        step_context: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Full forward pass.

        Args:
            goal_emb: (batch, 384) goal embedding from MiniLM
            goal_shape: (batch, 12) structural features
            step_context: (batch, 4) step_index, proof_length, hyp_count, target_len

        Returns:
            trust_weights: (batch, 5) probability over specialists
            info: dict with specialist scores, override flags, aux predictions
        """
        batch = goal_emb.shape[0]

        # Full context (shared across specialists at low capacity)
        context = np.concatenate([goal_emb, goal_shape, step_context], axis=-1)

        # Each specialist scores the goal
        specialist_scores = np.zeros((batch, self.cfg.n_specialists), dtype=np.float32)
        specialist_aux = {}

        for i, name in enumerate(SPECIALIST_NAMES):
            agent = self.specialists[name]
            # Domain features: goal_emb + goal_shape (all specialists see the same
            # features but learn to weight them differently via high-capacity path)
            domain_feats = np.concatenate([goal_emb, goal_shape], axis=-1)

            confidence, aux_pred = agent.forward(domain_feats, context)
            specialist_scores[:, i] = confidence
            specialist_aux[name] = aux_pred

        # Orchestrator
        trust_weights, override_flags, outcome_pred = self.orchestrator.forward(
            context, specialist_scores,
        )

        info = {
            "specialist_scores": specialist_scores,
            "specialist_aux": specialist_aux,
            "override_flags": override_flags,
            "outcome_pred": outcome_pred,
        }

        return trust_weights, info

    def predict(
        self,
        goal_emb: np.ndarray,
        goal_shape: np.ndarray,
        step_context: np.ndarray,
    ) -> tuple[str, float]:
        """Predict the best specialist for a single goal.

        Returns:
            specialist_name: name of the selected specialist
            confidence: trust weight for the selected specialist
        """
        if goal_emb.ndim == 1:
            goal_emb = goal_emb[np.newaxis, :]
            goal_shape = goal_shape[np.newaxis, :]
            step_context = step_context[np.newaxis, :]

        trust_weights, _ = self.forward(goal_emb, goal_shape, step_context)
        best_idx = int(np.argmax(trust_weights[0]))
        return SPECIALIST_NAMES[best_idx], float(trust_weights[0, best_idx])

    def param_count(self) -> int:
        total = sum(s.param_count() for s in self.specialists.values())
        total += self.orchestrator.param_count()
        return total

    def save(self, path: str) -> None:
        """Save model weights."""
        state = {}
        for name, agent in self.specialists.items():
            for i, layer in enumerate(agent.all_layers()):
                for j, p in enumerate(layer.parameters()):
                    state[f"specialist.{name}.{i}.{j}"] = p
        for i, layer in enumerate(self.orchestrator.all_layers()):
            for j, p in enumerate(layer.parameters()):
                state[f"orchestrator.{i}.{j}"] = p
        np.savez(path, **state)

    def load(self, path: str) -> None:
        """Load model weights."""
        state = np.load(path)
        for name, agent in self.specialists.items():
            for i, layer in enumerate(agent.all_layers()):
                params = layer.parameters()
                for j, p in enumerate(params):
                    key = f"specialist.{name}.{i}.{j}"
                    if key in state:
                        p[:] = state[key]
        for i, layer in enumerate(self.orchestrator.all_layers()):
            params = layer.parameters()
            for j, p in enumerate(params):
                key = f"orchestrator.{i}.{j}"
                if key in state:
                    p[:] = state[key]
