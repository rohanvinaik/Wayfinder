"""
Template classifier — RECOGNITION slot (Slot 2) for Wayfinder v2.

Lightweight classifier over GoalAnalyzer features (256d) that predicts
proof strategy templates. This is the key regime conversion: raw proof
structure (Regime B) becomes template classification (Regime A).

Architecture: features → Linear → ReLU → Linear → ReLU (template_features)
              → Linear → softmax (classification)

The intermediate 64d features serve as template_features for the PLANNING slot,
giving it a learned representation of template identity.

See DESIGN §10.2 for specification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.som_contracts import RecognitionOutput
from src.story_templates import TEMPLATE_NAMES, get_num_templates


class TemplateClassifier(nn.Module):
    """RECOGNITION slot: classify proof goals into strategy templates.

    Args:
        input_dim: GoalAnalyzer feature dimension (default 256).
        hidden_dim: Hidden layer dimension (default 128).
        feature_dim: Template feature dimension for planning (default 64).
        num_templates: Number of templates (default: from taxonomy).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        feature_dim: int = 64,
        num_templates: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_templates = num_templates or get_num_templates()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, self.num_templates)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and template features.

        Args:
            features: [batch, input_dim] from GoalAnalyzer.

        Returns:
            Tuple of (logits, template_features):
                logits: [batch, num_templates]
                template_features: [batch, feature_dim]
        """
        hidden = F.relu(self.layer1(features))
        template_features = F.relu(self.layer2(hidden))
        logits = self.classifier(template_features)
        return logits, template_features

    def predict(self, features: torch.Tensor) -> RecognitionOutput:
        """Produce a RecognitionOutput for the planning slot (inference mode).

        Args:
            features: [1, input_dim] single goal features.

        Returns:
            RecognitionOutput with template ID, confidence, and features.
        """
        with torch.no_grad():
            logits, template_features = self.forward(features)

        probs = F.softmax(logits[0], dim=-1)
        top_k_values, top_k_indices = torch.topk(probs, min(3, self.num_templates))

        best_idx = int(top_k_indices[0].item())
        best_conf = float(top_k_values[0].item())
        best_name = TEMPLATE_NAMES[best_idx]

        top_k_list = [
            (int(idx.item()), TEMPLATE_NAMES[int(idx.item())], float(val.item()))
            for idx, val in zip(top_k_indices, top_k_values)
        ]

        return RecognitionOutput(
            template_id=best_idx,
            template_name=best_name,
            template_confidence=best_conf,
            template_features=template_features[0],
            top_k_templates=top_k_list,
        )
