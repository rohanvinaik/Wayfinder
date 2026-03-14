"""
Proof sketch predictor — PLANNING slot (Slot 3) for Wayfinder v2.

Generates proof sketches: ordered sequences of subgoals with
estimated difficulty and anchor targets. Two modes:

1. Deterministic: For simple templates (DECIDE, REWRITE_CHAIN, APPLY_CHAIN,
   HAMMER_DELEGATE), the sketch IS the template — a fixed sequence.

2. Learned: For complex templates (INDUCT_THEN_CLOSE, DECOMPOSE_AND_CONQUER,
   CASE_ANALYSIS, CONTRAPOSITIVE, EPSILON_DELTA), a small model predicts
   the subgoal sequence from goal embedding + template features.

See DESIGN §10.3 for specification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.som_contracts import PlanningOutput, RecognitionOutput, SubgoalSpec
from src.story_templates import SIMPLE_TEMPLATES, TEMPLATE_TAXONOMY

# Deterministic sketches for simple templates
_SIMPLE_SKETCHES: dict[str, list[SubgoalSpec]] = {
    "DECIDE": [
        SubgoalSpec(
            subgoal_type="automation_close",
            estimated_steps=1,
            bank_hints={"automation": -1, "depth": -1},
        ),
    ],
    "REWRITE_CHAIN": [
        SubgoalSpec(
            subgoal_type="rewrite_normalize",
            estimated_steps=2,
            bank_hints={"structure": 0, "automation": 0},
        ),
        SubgoalSpec(
            subgoal_type="close_normal_form",
            estimated_steps=1,
            bank_hints={"automation": -1},
        ),
    ],
    "APPLY_CHAIN": [
        SubgoalSpec(
            subgoal_type="apply_lemma",
            estimated_steps=2,
            bank_hints={"structure": 0, "automation": 1},
        ),
        SubgoalSpec(
            subgoal_type="close_subgoal",
            estimated_steps=1,
            bank_hints={"automation": -1},
        ),
    ],
    "HAMMER_DELEGATE": [
        SubgoalSpec(
            subgoal_type="hammer",
            estimated_steps=1,
            bank_hints={"automation": -1},
        ),
    ],
}

# Subgoal type vocabulary for complex templates
SUBGOAL_TYPES: list[str] = [
    "base_case",
    "inductive_step",
    "decompose_have",
    "decompose_suffices",
    "case_split",
    "case_branch",
    "contrapositive_setup",
    "derive_contradiction",
    "introduce_witness",
    "bound_distance",
    "automation_close",
    "rewrite_normalize",
    "apply_lemma",
    "close_subgoal",
    "hammer",
]

_MAX_SUBGOALS = 6


class SketchPredictor(nn.Module):
    """PLANNING slot: predict proof sketch from goal embedding + template features.

    For simple templates, returns deterministic sketches (no learning).
    For complex templates, uses a learned model to predict subgoal sequences.

    Args:
        embedding_dim: Goal embedding dimension (default 384).
        template_feature_dim: Template feature dimension from classifier (default 64).
        hidden_dim: Hidden layer dimension (default 256).
        max_subgoals: Maximum number of subgoals in a sketch (default 6).
        num_subgoal_types: Size of subgoal type vocabulary.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        template_feature_dim: int = 64,
        hidden_dim: int = 256,
        max_subgoals: int = _MAX_SUBGOALS,
        num_subgoal_types: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.template_feature_dim = template_feature_dim
        self.max_subgoals = max_subgoals
        self.num_subgoal_types = num_subgoal_types or len(SUBGOAL_TYPES)

        input_dim = embedding_dim + template_feature_dim  # 448
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Predict number of subgoals (1 to max_subgoals)
        self.length_head = nn.Linear(hidden_dim, max_subgoals)

        # Predict subgoal types (one head per position)
        self.type_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, self.num_subgoal_types) for _ in range(max_subgoals)]
        )

        # Predict estimated steps per subgoal (scalar)
        self.step_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(max_subgoals)])

    def forward(
        self,
        goal_embedding: torch.Tensor,
        template_features: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass returning raw predictions.

        Args:
            goal_embedding: [batch, embedding_dim]
            template_features: [batch, template_feature_dim]

        Returns:
            Tuple of (length_logits, type_logits_list, step_preds_list):
                length_logits: [batch, max_subgoals]
                type_logits_list: list of [batch, num_subgoal_types] per position
                step_preds_list: list of [batch, 1] per position
        """
        x = torch.cat([goal_embedding, template_features], dim=-1)
        hidden = self.encoder(x)

        length_logits = self.length_head(hidden)
        type_logits = [head(hidden) for head in self.type_heads]
        step_preds = [F.softplus(head(hidden)) for head in self.step_heads]

        return length_logits, type_logits, step_preds

    def predict(
        self,
        goal_embedding: torch.Tensor,
        recognition: RecognitionOutput,
    ) -> PlanningOutput:
        """Produce a PlanningOutput (inference mode).

        For simple templates, returns deterministic sketch.
        For complex templates, uses learned predictions.
        """
        template_name = recognition.template_name

        # Simple templates: deterministic sketch
        if template_name in SIMPLE_TEMPLATES:
            simple_sketch = list(_SIMPLE_SKETCHES.get(template_name, []))
            return PlanningOutput(
                sketch=simple_sketch,
                total_estimated_depth=sum(s.estimated_steps for s in simple_sketch),
                template_id=recognition.template_id,
            )

        # Complex templates: learned prediction
        learned_sketch = self._predict_complex(goal_embedding, recognition, template_name)
        return PlanningOutput(
            sketch=learned_sketch,
            total_estimated_depth=sum(s.estimated_steps for s in learned_sketch),
            template_id=recognition.template_id,
        )

    def _predict_complex(
        self,
        goal_embedding: torch.Tensor,
        recognition: RecognitionOutput,
        template_name: str,
    ) -> list[SubgoalSpec]:
        """Predict subgoal sequence for complex templates."""
        with torch.no_grad():
            template_features = recognition.template_features.unsqueeze(0)
            if goal_embedding.dim() == 1:
                goal_embedding = goal_embedding.unsqueeze(0)

            length_logits, type_logits, step_preds = self.forward(goal_embedding, template_features)

        num_subgoals = int(length_logits[0].argmax().item()) + 1  # 1-indexed
        template_info = TEMPLATE_TAXONOMY.get(template_name)
        bank_hints = dict(template_info.bank_signature) if template_info else {}

        sketch: list[SubgoalSpec] = []
        for i in range(num_subgoals):
            type_idx = int(type_logits[i][0].argmax().item())
            subgoal_type = (
                SUBGOAL_TYPES[type_idx] if type_idx < len(SUBGOAL_TYPES) else "automation_close"
            )
            est_steps = max(1, int(step_preds[i][0].item() + 0.5))
            sketch.append(
                SubgoalSpec(
                    subgoal_type=subgoal_type,
                    estimated_steps=est_steps,
                    bank_hints=dict(bank_hints),
                )
            )
        return sketch
