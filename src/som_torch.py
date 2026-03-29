"""PyTorch Society of Mind model for multi-step proof tactic selection.

Architecture adapted from Yami chess SoM (SOM_TRAINING_ANALYSIS):
- 5 specialist agents (rewrite, structural, solver, apply, closer)
- 1 orchestrator with trust-weight softmax + override mechanism
- Three-stage training: specialists -> orchestrator -> joint fine-tuning

Each specialist sees:
  - Domain signals at HIGH capacity (specialist-specific goal features)
  - Full goal context at LOW capacity (shared embedding)

The orchestrator sees:
  - Goal context + all specialist confidence scores
  - Outputs softmax trust weights over specialists

Total parameters: ~1.2M (5 x ~210K + ~72K orchestrator)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECIALIST_NAMES = ["rewrite", "structural", "solver", "apply", "closer"]


@dataclass
class SoMConfig:
    """Configuration for the SoM model."""
    goal_emb_dim: int = 384
    goal_shape_dim: int = 12
    step_context_dim: int = 4

    n_specialists: int = 5
    domain_hidden: int = 256
    context_hidden: int = 128
    specialist_hidden: int = 128

    orch_hidden: int = 128
    override_threshold: float = 0.8
    override_boost: float = 5.0

    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-3
    lr_stage3: float = 1e-4
    weight_decay: float = 1e-4
    aux_weight: float = 0.05


class SpecialistAgent(nn.Module):
    """One specialist in the SoM ensemble.

    Dual-path architecture (Decision 2 from SOM_TRAINING_ANALYSIS):
    - Domain encoder: specialist-specific features at HIGH capacity
    - Context encoder: full goal embedding at LOW capacity
    - Scoring head: confidence score for "should I handle this?"
    """

    def __init__(self, name: str, domain_dim: int, context_dim: int, cfg: SoMConfig):
        super().__init__()
        self.name = name

        # Domain encoder (HIGH capacity)
        self.domain_net = nn.Sequential(
            nn.Linear(domain_dim, cfg.domain_hidden),
            nn.LayerNorm(cfg.domain_hidden),
            nn.GELU(),
            nn.Linear(cfg.domain_hidden, cfg.specialist_hidden),
            nn.LayerNorm(cfg.specialist_hidden),
            nn.GELU(),
        )

        # Context encoder (LOW capacity)
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, cfg.context_hidden),
            nn.LayerNorm(cfg.context_hidden),
            nn.GELU(),
        )

        # Scoring head: domain + context -> confidence
        score_in = cfg.specialist_hidden + cfg.context_hidden
        self.score_net = nn.Sequential(
            nn.Linear(score_in, cfg.specialist_hidden),
            nn.GELU(),
            nn.Linear(cfg.specialist_hidden, 1),
        )

        # Auxiliary head: predict proof progress (weak supervisory signal)
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.specialist_hidden, 1),
            nn.Tanh(),
        )

    def forward(
        self, domain_feats: torch.Tensor, context_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (confidence, aux_pred), both (batch,)."""
        d = self.domain_net(domain_feats)
        c = self.context_net(context_feats)
        combined = torch.cat([d, c], dim=-1)
        confidence = self.score_net(combined).squeeze(-1)
        aux_pred = self.aux_head(d).squeeze(-1)
        return confidence, aux_pred


class Orchestrator(nn.Module):
    """Orchestrator that learns trust weights over specialists.

    Decision 3 (Override): When override > threshold, specialist's trust
    logit gets boosted. Handles cases where one specialist should dominate.

    Decision 4 (Trust as softmax): Trust weights sum to 1.0 via softmax.
    Override as logit boost preserves gradient flow.
    """

    def __init__(self, cfg: SoMConfig):
        super().__init__()
        self.cfg = cfg

        ctx_in = cfg.goal_emb_dim + cfg.goal_shape_dim + cfg.step_context_dim
        self.ctx_net = nn.Sequential(
            nn.Linear(ctx_in, cfg.orch_hidden),
            nn.LayerNorm(cfg.orch_hidden),
            nn.GELU(),
            nn.Linear(cfg.orch_hidden, cfg.orch_hidden // 2),
            nn.GELU(),
        )

        trust_in = cfg.orch_hidden // 2 + cfg.n_specialists
        self.trust_net = nn.Sequential(
            nn.Linear(trust_in, cfg.orch_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.orch_hidden // 2, cfg.n_specialists),
        )

        # Override head: sigmoid per specialist
        self.override_head = nn.Linear(cfg.orch_hidden // 2, cfg.n_specialists)

        # Auxiliary: predict proof outcome
        self.outcome_head = nn.Sequential(
            nn.Linear(cfg.orch_hidden // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        goal_context: torch.Tensor,
        specialist_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (trust_weights, override_flags, outcome_pred)."""
        c = self.ctx_net(goal_context)

        trust_in = torch.cat([c, specialist_scores], dim=-1)
        trust_logits = self.trust_net(trust_in)

        # Override mechanism (Decision 3)
        override_flags = torch.sigmoid(self.override_head(c))
        boost_mask = (override_flags > self.cfg.override_threshold).float()
        trust_logits = trust_logits + boost_mask * self.cfg.override_boost

        trust_weights = F.softmax(trust_logits, dim=-1)
        outcome_pred = self.outcome_head(c).squeeze(-1)

        return trust_weights, override_flags, outcome_pred


class SoMModel(nn.Module):
    """Complete Society of Mind model for tactic family selection.

    Forward: specialists score goal -> orchestrator combines -> softmax trust weights.
    Output: distribution over 5 specialist families.
    """

    def __init__(self, cfg: SoMConfig | None = None):
        super().__init__()
        self.cfg = cfg or SoMConfig()

        domain_dim = self.cfg.goal_emb_dim + self.cfg.goal_shape_dim
        context_dim = self.cfg.goal_emb_dim + self.cfg.goal_shape_dim + self.cfg.step_context_dim

        # Keys prefixed with "spec_" to avoid conflict with nn.Module.apply()
        self.specialists = nn.ModuleDict({
            f"spec_{name}": SpecialistAgent(name, domain_dim, context_dim, self.cfg)
            for name in SPECIALIST_NAMES
        })
        self.orchestrator = Orchestrator(self.cfg)

    def forward(
        self,
        goal_emb: torch.Tensor,
        goal_shape: torch.Tensor,
        step_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Full forward pass.

        Args:
            goal_emb: (batch, 384)
            goal_shape: (batch, 12)
            step_context: (batch, 4)

        Returns:
            trust_weights: (batch, 5) probability over specialists
            info: specialist scores, aux predictions, override flags
        """
        context = torch.cat([goal_emb, goal_shape, step_context], dim=-1)
        domain_feats = torch.cat([goal_emb, goal_shape], dim=-1)

        specialist_scores = []
        specialist_aux = {}
        for name in SPECIALIST_NAMES:
            agent = self.specialists[f"spec_{name}"]
            confidence, aux_pred = agent(domain_feats, context)
            specialist_scores.append(confidence)
            specialist_aux[name] = aux_pred

        scores_tensor = torch.stack(specialist_scores, dim=-1)
        trust_weights, override_flags, outcome_pred = self.orchestrator(
            context, scores_tensor,
        )

        info = {
            "specialist_scores": scores_tensor,
            "specialist_aux": specialist_aux,
            "override_flags": override_flags,
            "outcome_pred": outcome_pred,
        }
        return trust_weights, info

    def predict(
        self,
        goal_emb: torch.Tensor,
        goal_shape: torch.Tensor,
        step_context: torch.Tensor,
    ) -> tuple[str, float]:
        """Predict best specialist for a single goal."""
        if goal_emb.dim() == 1:
            goal_emb = goal_emb.unsqueeze(0)
            goal_shape = goal_shape.unsqueeze(0)
            step_context = step_context.unsqueeze(0)

        with torch.no_grad():
            trust_weights, _ = self(goal_emb, goal_shape, step_context)
        best_idx = int(trust_weights[0].argmax())
        return SPECIALIST_NAMES[best_idx], float(trust_weights[0, best_idx])
