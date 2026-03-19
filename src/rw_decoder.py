"""Rewrite decoder — predicts ActionIR for rw tactics.

Two-stage decoder (GSE pattern):
    Stage 1: Shape — predict rewrite count + direction flags
    Stage 2: Leaves — select premise symbols from local vocabulary

Input: goal embedding + local vocabulary embeddings
Output: ActionIR with RewriteAtom list

This is the simplest family decoder and should be built first:
    - Grammar is fixed: rw [expr1, expr2, ...]
    - Each atom has one direction bit and one TermExpr
    - Highest premise sensitivity (oracle +16pp recall)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.tactic_ir import (
    ActionIR,
    Direction,
    RewriteAtom,
    const,
    var,
)

MAX_REWRITES = 6  # Cover 97% of cases (1-6 rewrites)


@dataclass
class RwPrediction:
    """Decoded rw ActionIR prediction."""

    n_rewrites: int
    directions: list[str]  # "forward" | "backward" per atom
    symbol_indices: list[int]  # index into local vocabulary
    symbol_names: list[str]  # resolved names
    confidence: float = 0.0

    def to_action_ir(self) -> ActionIR:
        """Convert to ActionIR for deterministic lowering."""
        rewrites = []
        for i in range(self.n_rewrites):
            direction = (
                Direction.BACKWARD
                if i < len(self.directions) and self.directions[i] == "backward"
                else Direction.FORWARD
            )
            name = self.symbol_names[i] if i < len(self.symbol_names) else "_"
            # Use var() for hypothesis-like names, const() for lemma names
            expr = var(name) if len(name) <= 3 and name[0].islower() else const(name)
            rewrites.append(RewriteAtom(direction=direction, expr=expr))
        return ActionIR(family="rw", rewrites=rewrites)


class RwDecoder(nn.Module):
    """Two-stage rw tactic decoder.

    Stage 1 (shape): goal_emb → (n_rewrites, direction_flags)
    Stage 2 (leaves): goal_emb + shape → pointer over local vocabulary

    Args:
        goal_dim: dimension of goal embedding (384 for MiniLM)
        hidden_dim: internal hidden dimension
        max_rewrites: maximum number of rewrite atoms to predict
    """

    def __init__(
        self,
        goal_dim: int = 384,
        hidden_dim: int = 256,
        max_rewrites: int = MAX_REWRITES,
    ) -> None:
        super().__init__()
        self.max_rewrites = max_rewrites

        # Stage 1: Shape prediction
        self.shape_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Predict number of rewrites (1..max_rewrites)
        self.count_head = nn.Linear(hidden_dim, max_rewrites)
        # Predict direction per position (forward/backward)
        self.direction_head = nn.Linear(hidden_dim, max_rewrites)  # sigmoid per position

        # Stage 2: Leaf selection (pointer over local vocabulary)
        # Query: goal_emb + position encoding
        # Key: local vocabulary embeddings
        self.position_emb = nn.Embedding(max_rewrites, hidden_dim)
        self.query_proj = nn.Linear(goal_dim + hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(goal_dim, hidden_dim)  # vocab items are also goal_dim

    def forward_shape(self, goal_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: predict rewrite count and directions.

        Args:
            goal_emb: [batch, goal_dim]

        Returns:
            count_logits: [batch, max_rewrites] — softmax over 1..max_rewrites
            direction_logits: [batch, max_rewrites] — sigmoid per position
        """
        h = self.shape_encoder(goal_emb)
        return self.count_head(h), self.direction_head(h)

    def forward_leaves(
        self,
        goal_emb: torch.Tensor,
        vocab_emb: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage 2: pointer attention over local vocabulary.

        Args:
            goal_emb: [batch, goal_dim]
            vocab_emb: [batch, vocab_size, goal_dim] — local vocabulary embeddings
            positions: [batch, n_positions] — which positions to decode (default: all)

        Returns:
            pointer_logits: [batch, n_positions, vocab_size]
        """
        batch_size = goal_emb.shape[0]
        if positions is None:
            positions = torch.arange(self.max_rewrites, device=goal_emb.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)

        n_pos = positions.shape[1]

        # Position embeddings
        pos_emb = self.position_emb(positions)  # [batch, n_pos, hidden]

        # Query: goal + position
        goal_expanded = goal_emb.unsqueeze(1).expand(-1, n_pos, -1)  # [batch, n_pos, goal_dim]
        query_input = torch.cat(
            [goal_expanded, pos_emb], dim=-1
        )  # [batch, n_pos, goal_dim + hidden]
        queries = self.query_proj(query_input)  # [batch, n_pos, hidden]

        # Keys: vocabulary items
        keys = self.key_proj(vocab_emb)  # [batch, vocab_size, hidden]

        # Pointer attention: dot product
        pointer_logits = torch.bmm(queries, keys.transpose(1, 2))  # [batch, n_pos, vocab_size]
        return pointer_logits

    def predict(
        self,
        goal_emb: torch.Tensor,
        vocab_emb: torch.Tensor,
        vocab_names: list[str],
    ) -> RwPrediction:
        """Full prediction: shape + leaves → RwPrediction."""
        self.eval()
        with torch.no_grad():
            # Stage 1: shape
            count_logits, dir_logits = self.forward_shape(goal_emb.unsqueeze(0))
            n_rewrites = int(count_logits.argmax(dim=-1).item()) + 1  # 1-indexed
            directions = [
                "backward" if torch.sigmoid(dir_logits[0, i]).item() > 0.5 else "forward"
                for i in range(n_rewrites)
            ]

            # Stage 2: leaves
            positions = torch.arange(n_rewrites, device=goal_emb.device).unsqueeze(0)
            pointer_logits = self.forward_leaves(
                goal_emb.unsqueeze(0), vocab_emb.unsqueeze(0), positions
            )
            symbol_indices = pointer_logits.argmax(dim=-1).squeeze(0).tolist()
            symbol_names = [
                vocab_names[idx] if idx < len(vocab_names) else "_" for idx in symbol_indices
            ]

        return RwPrediction(
            n_rewrites=n_rewrites,
            directions=directions,
            symbol_indices=symbol_indices,
            symbol_names=symbol_names,
        )
