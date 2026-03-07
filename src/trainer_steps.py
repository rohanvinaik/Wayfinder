"""Training step logic and batch-building helpers for proof synthesis."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from src.trainer_constants import _REPAIR_SEVERITY, infer_domain

if TYPE_CHECKING:
    import torch


class TrainerStepsMixin:
    """Mixin providing train_step, domain gate step, and batch-building helpers.

    Expects the host class to provide: pipeline, vocabs, infra, config, device,
    _rng, _ood, _tracking, step.
    """

    pipeline: Any
    vocabs: Any
    infra: Any
    config: dict
    device: str
    step: int
    _rng: Any
    _ood: Any
    _tracking: dict[str, Any]

    def _trainable_params(self, composite_loss: Any = None) -> list:
        raise NotImplementedError

    def _gate_trainable_params(self) -> list:
        raise NotImplementedError

    @staticmethod
    def _tier1_token(example: Any) -> str:
        """Extract first tactic token target for an example."""
        tokens = getattr(example, "tier1_tokens", [])
        if not tokens:
            return "<UNK>"
        # Skip BOS if present
        return tokens[1] if len(tokens) > 1 else tokens[0]

    def _build_tier1_targets(self, batch: list) -> torch.Tensor:
        """Build tier1 target indices from batch examples."""
        import torch

        unk_idx = self.vocabs.tier1.get("<UNK>", 1)
        targets = []
        for ex in batch:
            token = self._tier1_token(ex)
            targets.append(self.vocabs.tier1.get(token, unk_idx))
        return torch.tensor(targets, dtype=torch.long, device=self.device)

    def _index_negative_bank(
        self, neg_dataset: Any
    ) -> tuple[dict[str, list[int]], dict[str, float]]:
        """Build goal-indexed negative targets and repair weights."""
        prompt_to_negatives: dict[str, list[int]] = defaultdict(list)
        prompt_to_repair_weight: dict[str, float] = {}
        unk_idx = self.vocabs.tier1.get("<UNK>", 1)

        for entry in getattr(neg_dataset, "entries", []):
            goal = (entry.goal_state or "").strip()
            if not goal:
                continue

            if entry.negative is not None:
                neg_token = self._tier1_token(entry.negative)
                prompt_to_negatives[goal].append(self.vocabs.tier1.get(neg_token, unk_idx))

            sev = max((_REPAIR_SEVERITY.get(tag, 0.5) for tag in entry.error_tags), default=0.5)
            repair_weight = 1.0 + 0.5 * sev
            prompt_to_repair_weight[goal] = max(
                repair_weight, prompt_to_repair_weight.get(goal, 1.0)
            )

        return dict(prompt_to_negatives), dict(prompt_to_repair_weight)

    def _sample_negative_index(self, positive_index: int) -> int:
        """Sample a random tier1 index different from positive_index."""
        vocab_size = max(2, len(self.vocabs.tier1))
        candidate = positive_index
        for _ in range(4):
            candidate = int(self._rng.integers(0, vocab_size))
            if candidate != positive_index:
                return candidate
        return (positive_index + 1) % vocab_size

    def _build_negative_targets(self, batch: list, tier1_targets: torch.Tensor) -> torch.Tensor:
        """Select negative targets from negative-bank map."""
        import torch

        negative_targets: list[int] = []
        neg_map = self.infra.negative_bank_by_prompt or {}
        for i, ex in enumerate(batch):
            positive_idx = int(tier1_targets[i].item())
            goal = (ex.goal_state or "").strip()
            candidates = neg_map.get(goal, [])

            if candidates:
                choice = int(candidates[int(self._rng.integers(0, len(candidates)))])
                if choice == positive_idx and len(candidates) > 1:
                    choice = int(candidates[(candidates.index(choice) + 1) % len(candidates)])
                negative_targets.append(choice)
            else:
                negative_targets.append(self._sample_negative_index(positive_idx))

        return torch.tensor(negative_targets, dtype=torch.long, device=self.device)

    def _build_repair_weights(self, batch: list) -> torch.Tensor:
        """Build repair-weight vector from negative-bank tags."""
        import torch

        repair_map = self.infra.repair_weight_by_prompt or {}
        weights = []
        for ex in batch:
            goal = (ex.goal_state or "").strip()
            weight = repair_map.get(goal, 1.0)
            weights.append(float(weight))

        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _track_batch_metrics(
        self,
        batch: list,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Track tier/domain/tactic metrics from the latest batch."""
        correct = predictions == targets
        tier1_acc = float(np.mean(correct)) if len(correct) else 0.0

        domain_correct: dict[str, list[float]] = defaultdict(list)
        tactic_correct: dict[str, list[float]] = defaultdict(list)
        idx2token = {v: k for k, v in self.vocabs.tier1.items()}
        for i, ex in enumerate(batch):
            domain = infer_domain(ex)
            domain_correct[domain].append(float(correct[i]))
            tactic_token = idx2token.get(int(targets[i]), "<UNK>")
            tactic_correct[tactic_token].append(float(correct[i]))

        prev_tiers = self._tracking["tier_accuracies"]
        self._tracking["tier_accuracies"] = {
            "tier1": tier1_acc,
            "tier2": prev_tiers.get("tier2", tier1_acc),
            "tier3": prev_tiers.get("tier3", tier1_acc),
        }
        self._tracking["domain_accuracies"] = {
            d: float(np.mean(vals)) for d, vals in domain_correct.items() if vals
        }
        self._tracking["tactic_accuracies"] = {
            a: float(np.mean(vals)) for a, vals in tactic_correct.items() if vals
        }

    def _sample_ood_examples(self, n: int, in_domain: bool) -> list[Any]:
        """Sample OOD prompt records by label with replacement."""
        pool = self._ood.in_domain if in_domain else self._ood.out_domain
        if not pool:
            return []
        indices = self._rng.integers(0, len(pool), size=n)
        return [pool[int(i)] for i in indices]

    def _domain_gate_step(self, in_domain_prompts: list[str]) -> tuple[float, float]:
        """Run a separate OOD training step for the domain gate head."""
        import torch

        gate_optimizer = self.infra.gate_optimizer
        if gate_optimizer is None:
            return 0.0, 0.0

        bs = max(1, len(in_domain_prompts))
        ood_n = max(1, bs // 2)
        id_n = max(1, bs - ood_n)

        sampled_ood = self._sample_ood_examples(ood_n, in_domain=False)
        sampled_id = self._sample_ood_examples(id_n, in_domain=True)

        prompts = list(in_domain_prompts)
        labels = [1.0] * len(in_domain_prompts)

        for rec in sampled_ood + sampled_id:
            prompts.append(rec.prompt)
            labels.append(1.0 if rec.label == "in_domain" else 0.0)

        if not prompts:
            return 0.0, 0.0

        embeddings = self.pipeline.encoder.encode(prompts).clone().detach().requires_grad_(False)
        logits = self.pipeline.domain_gate(embeddings)
        label_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device).unsqueeze(1)

        gate_loss = self.infra.ood_loss(logits, label_tensor)
        gate_optimizer.zero_grad()
        gate_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._gate_trainable_params(), self.config["safety"]["max_grad_norm"]
        )
        gate_optimizer.step()

        gate_pred = (torch.sigmoid(logits) > 0.5).float()
        gate_acc = float((gate_pred == label_tensor).float().mean().item())
        return float(gate_loss.item()), gate_acc

    def train_step(self, batch: list, return_per_example: bool = False) -> dict[str, float]:
        """Execute one training step."""
        import torch
        import torch.nn.functional as F

        goal_states = [ex.goal_state for ex in batch]
        embeddings = (
            self.pipeline.encoder.encode(goal_states).clone().detach().requires_grad_(False)
        )

        analyzer_features = self.pipeline.goal_analyzer(embeddings)
        bridge_features = self.pipeline.bridge(analyzer_features)
        decoder_output = self.pipeline.decoder(bridge_features)
        tier1_logits = decoder_output["tier1_logits"]

        self._tracking["bridge_features"] = bridge_features.detach().cpu().numpy()
        tier1_targets = self._build_tier1_targets(batch)
        negative_targets = self._build_negative_targets(batch, tier1_targets)
        repair_weights = self._build_repair_weights(batch)

        loss_dict = self.infra.composite_loss(
            tier1_logits,
            tier1_targets,
            negative_targets=negative_targets,
            repair_weights=repair_weights,
            margin=self.config.get("training", {}).get("loss", {}).get("margin", 0.5),
        )

        total_loss = loss_dict["L_total"]
        self.infra.optimizer.zero_grad()
        total_loss.backward()

        max_grad = self.config["safety"]["max_grad_norm"]
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._trainable_params() if p.grad is not None],
            max_grad,
        )
        self.infra.optimizer.step()

        gate_loss, gate_acc = self._domain_gate_step(goal_states)
        self.step += 1

        per_example_ce = F.cross_entropy(tier1_logits, tier1_targets, reduction="none")
        pred_np = tier1_logits.argmax(dim=-1).detach().cpu().numpy()
        target_np = tier1_targets.detach().cpu().numpy()
        self._track_batch_metrics(batch, pred_np, target_np)

        metrics: dict[str, float] = {
            "L_total": float(total_loss.item()),
            "L_ce": float(loss_dict["L_ce"].item()),
            "L_margin": float(loss_dict["L_margin"].item()),
            "L_repair": float(loss_dict["L_repair"].item()),
            "L_gate": float(gate_loss),
            "gate_accuracy": float(gate_acc),
            "batch_tier1_accuracy": float(self._tracking["tier_accuracies"].get("tier1", 0.0)),
            "L_total_joint": float(total_loss.item() + 0.1 * gate_loss),
            "w_ce": float(loss_dict["w_ce"].item()),
            "w_margin": float(loss_dict["w_margin"].item()),
            "w_repair": float(loss_dict["w_repair"].item()),
            "sigma_ce": float(loss_dict["sigma_ce"].item()),
            "sigma_margin": float(loss_dict["sigma_margin"].item()),
            "sigma_repair": float(loss_dict["sigma_repair"].item()),
        }

        if return_per_example:
            metrics["per_example_loss"] = per_example_ce.detach().cpu().tolist()  # type: ignore[assignment]

        return metrics
