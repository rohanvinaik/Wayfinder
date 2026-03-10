"""Training loop for Balanced Sashimi proof synthesis models."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.trainer_checks import (
    PABMetricsSnapshot,
    check_gradient_abort,
    check_gradient_health,
    collect_decoder_weight_signs,
    log_ternary_distribution,
    record_pab_checkpoint,
    save_checkpoint,
)
from src.trainer_constants import infer_domain
from src.trainer_setup import (
    build_losses,
    build_pab_tracker,
    build_pipeline_modules,
    load_datasets,
    load_vocabs,
    setup_run_dirs,
)
from src.trainer_steps import TrainerStepsMixin

_Pipeline = namedtuple("_Pipeline", "encoder domain_gate goal_analyzer bridge decoder")
_Vocabs = namedtuple("_Vocabs", "tier1 tier2")
_TrainInfra = namedtuple(
    "_TrainInfra",
    (
        "composite_loss "
        "ood_loss "
        "optimizer "
        "dataset "
        "pab_tracker "
        "gate_optimizer "
        "negative_bank_by_prompt "
        "repair_weight_by_prompt "
        "ood_examples "
        "eval_examples"
    ),
)
_TrainInfra.__new__.__defaults__ = (None, None, None, None, None)
_TrainerInit = namedtuple("_TrainerInit", "config run_id device seed encoder_override")
_RunPaths = namedtuple("_RunPaths", "run_dir checkpoint_dir")
_OODExamples = namedtuple("_OODExamples", "in_domain out_domain")


def load_config(path: Path) -> dict:
    """Load experiment configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


class BalancedSashimiTrainer(TrainerStepsMixin):
    """Training orchestrator for the Balanced Sashimi proof synthesis pipeline."""

    def __init__(
        self,
        config: dict,
        run_id: str,
        device: str = "mps",
        seed: int = 42,
        encoder_override: object | None = None,
    ) -> None:
        self._init = _TrainerInit(config, run_id, device, seed, encoder_override)
        self._paths = _RunPaths(run_dir=Path("."), checkpoint_dir=Path("."))
        self._rng = np.random.default_rng(seed)
        self._output_mode = "proof"
        self._ood = _OODExamples(in_domain=[], out_domain=[])
        self._tracking: dict[str, Any] = {
            "bridge_features": None,
            "tier_accuracies": {"tier1": 0.0, "tier2": 0.0, "tier3": 0.0},
            "domain_accuracies": {},
            "tactic_accuracies": {},
            "validation_snapshot": {},
        }

    @property
    def config(self) -> dict:
        return self._init.config

    @property
    def run_id(self) -> str:
        return self._init.run_id

    @property
    def device(self) -> str:
        return self._init.device

    @property
    def seed(self) -> int:
        return self._init.seed

    @property
    def encoder_override(self) -> object | None:
        return self._init.encoder_override

    @property
    def run_dir(self) -> Path:
        return self._paths.run_dir

    @property
    def checkpoint_dir(self) -> Path:
        return self._paths.checkpoint_dir

    def setup(self) -> None:
        """Initialize model, optimizer, data loaders, and logging."""
        import torch

        torch.manual_seed(self.seed)

        modules, self._output_mode = build_pipeline_modules(
            self.config,
            self.device,
            self.encoder_override,
        )
        self.pipeline = _Pipeline(*modules)
        tier1, tier2 = load_vocabs(self.config)
        self.vocabs = _Vocabs(tier1=tier1, tier2=tier2)

        composite_loss, ood_loss = build_losses(self.config, self.device)
        pab_tracker = build_pab_tracker(self.config, self.run_id)

        train_dataset, neg_dataset, ood_examples, eval_examples = load_datasets(self.config)
        negative_bank_by_prompt: dict[str, list[int]] = {}
        repair_weight_by_prompt: dict[str, float] = {}
        if neg_dataset is not None:
            negative_bank_by_prompt, repair_weight_by_prompt = self._index_negative_bank(
                neg_dataset
            )

        self._ood = _OODExamples(
            in_domain=[p for p in ood_examples if p.label == "in_domain"],
            out_domain=[p for p in ood_examples if p.label != "in_domain"],
        )

        train_cfg = self.config["training"]
        optimizer = torch.optim.AdamW(
            list(self._trainable_params(composite_loss)),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        gate_lr = train_cfg.get("domain_gate_learning_rate", train_cfg["learning_rate"])
        gate_optimizer = torch.optim.AdamW(
            list(self._gate_trainable_params()),
            lr=gate_lr,
            weight_decay=train_cfg["weight_decay"],
        )

        self.infra = _TrainInfra(
            composite_loss=composite_loss,
            ood_loss=ood_loss,
            optimizer=optimizer,
            dataset=train_dataset,
            pab_tracker=pab_tracker,
            gate_optimizer=gate_optimizer,
            negative_bank_by_prompt=negative_bank_by_prompt,
            repair_weight_by_prompt=repair_weight_by_prompt,
            ood_examples=ood_examples,
            eval_examples=eval_examples,
        )

        self.step = 0
        run_dir, checkpoint_dir = setup_run_dirs(self.config, self.run_id)
        self._paths = _RunPaths(run_dir=run_dir, checkpoint_dir=checkpoint_dir)

    def _trainable_params(self, composite_loss=None) -> list:
        """Collect generation-side trainable parameters (excluding domain gate)."""
        loss = composite_loss if composite_loss is not None else self.infra.composite_loss
        params: list[Any] = []
        for module in [
            self.pipeline.goal_analyzer,
            self.pipeline.bridge,
            self.pipeline.decoder,
            loss,
        ]:
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    def _gate_trainable_params(self) -> list:
        """Collect domain gate trainable parameters."""
        return [p for p in self.pipeline.domain_gate.parameters() if p.requires_grad]

    def _run_inference_on_subset(self, subset: list) -> tuple[float, np.ndarray, np.ndarray]:
        """Run forward pass on a subset and return (val_loss, pred_indices, target_indices)."""
        import torch
        import torch.nn.functional as F

        goal_states = [ex.goal_state for ex in subset]
        with torch.no_grad():
            emb = self.pipeline.encoder.encode(goal_states)
            frame = self.pipeline.goal_analyzer(emb)
            bridge = self.pipeline.bridge(frame)
            out = self.pipeline.decoder(bridge)
            logits = out["tier1_logits"]
            targets = self._build_tier1_targets(subset)
            val_loss = float(F.cross_entropy(logits, targets).item())
            pred_idx = logits.argmax(dim=-1).detach().cpu().numpy()
            target_idx = targets.detach().cpu().numpy()
        return val_loss, pred_idx, target_idx

    def _compute_validation_snapshot(self) -> dict[str, Any]:
        """Compute lightweight eval metrics for PAB checkpoints."""
        eval_examples = self.infra.eval_examples or []
        if not eval_examples:
            return {}

        pab_cfg = self.config.get("pab", {})
        subset_n = max(1, int(pab_cfg.get("validation_subset_size", 16)))
        if len(eval_examples) <= subset_n:
            subset = list(eval_examples)
        else:
            idx = self._rng.integers(0, len(eval_examples), size=subset_n)
            subset = [eval_examples[int(i)] for i in idx]

        val_loss, pred_idx, target_idx = self._run_inference_on_subset(subset)

        idx2token = {v: k for k, v in self.vocabs.tier1.items()}
        predicted_tokens = [idx2token.get(int(i), "<UNK>") for i in pred_idx]
        domains = [infer_domain(ex) for ex in subset]

        domain_acc: dict[str, list[float]] = defaultdict(list)
        tactic_acc: dict[str, list[float]] = defaultdict(list)

        for i, ex in enumerate(subset):
            expected = self._tier1_token(ex)
            predicted = predicted_tokens[i]
            tactic_acc[expected].append(float(predicted == expected))
            domain_acc[domains[i]].append(float(predicted == expected))

        return {
            "val_loss": val_loss,
            "tier1_accuracy": float(np.mean(pred_idx == target_idx)) if len(target_idx) else 0.0,
            "domain_accuracies": {d: float(np.mean(v)) for d, v in domain_acc.items() if v},
            "tactic_accuracies": {a: float(np.mean(v)) for a, v in tactic_acc.items() if v},
        }

    def check_gradient_health(self) -> bool:
        healthy, _msg = check_gradient_health(self.pipeline)
        return healthy

    def _collect_decoder_weight_signs(self) -> np.ndarray | None:
        return collect_decoder_weight_signs(self.pipeline.decoder)

    def log_ternary_distribution(self, _step: int) -> dict[str, dict[str, float]]:
        return log_ternary_distribution(self.pipeline.decoder)

    def save_checkpoint(self, step: int) -> Path:
        return save_checkpoint(
            self.checkpoint_dir,
            self.run_id,
            step,
            self.config,
            self.pipeline,
            self.infra,
        )

    def _run_step_checks(self, loss_dict: dict, safety: dict) -> dict | None:
        abort = check_gradient_abort(self.step, safety, self.pipeline, self.save_checkpoint)
        if abort is not None:
            return abort

        if self.step % safety["ternary_distribution_log_every_n_steps"] == 0:
            ternary_dist = self.log_ternary_distribution(self.step)
            if ternary_dist:
                print(f"  Step {self.step} ternary: {ternary_dist}")

        if self.step % 50 == 0:
            print(
                f"  Step {self.step}/{self.config['training']['max_iterations']}"
                f": L_total={loss_dict['L_total']:.4f}"
                f", gate={loss_dict.get('L_gate', 0.0):.4f}"
            )

        snapshot = {}
        tracker = self.infra.pab_tracker
        if tracker is not None:
            interval = self.config.get("pab", {}).get("checkpoint_interval", 50)
            if self.step % interval == 0:
                snapshot = self._compute_validation_snapshot()
                self._tracking["validation_snapshot"] = snapshot
                if snapshot:
                    self._tracking["tier_accuracies"] = {
                        "tier1": float(snapshot.get("tier1_accuracy", 0.0)),
                        "tier2": 0.0,
                        "tier3": 0.0,
                    }
                    self._tracking["domain_accuracies"] = dict(
                        snapshot.get("domain_accuracies", {})
                    )
                    self._tracking["tactic_accuracies"] = dict(
                        snapshot.get("tactic_accuracies", {})
                    )

        metrics = PABMetricsSnapshot(
            val_loss=snapshot.get("val_loss") if snapshot else None,
            tier_accuracies=self._tracking["tier_accuracies"],
            domain_accuracies=self._tracking["domain_accuracies"],
            tactic_accuracies=self._tracking.get("tactic_accuracies", {}),
        )
        return record_pab_checkpoint(
            self.step,
            loss_dict,
            self.config,
            self.infra,
            self._tracking.get("bridge_features"),
            self.pipeline.decoder,
            metrics=metrics,
        )

    @staticmethod
    def _build_training_result(
        step: int,
        all_losses: list[dict],
        epoch: int,
        elapsed: float,
        checkpoint_path: str,
        validation_snapshot: dict | None = None,
        pab_profile_path: str | None = None,
    ) -> dict:
        """Pure construction of training result summary dict."""
        result = {
            "status": "complete",
            "steps": step,
            "epochs": epoch,
            "elapsed_s": round(elapsed, 1),
            "final_loss": all_losses[-1] if all_losses else {},
            "checkpoint": checkpoint_path,
        }
        if validation_snapshot:
            result["final_validation_snapshot"] = validation_snapshot
        if pab_profile_path:
            result["pab_profile"] = pab_profile_path
        return result

    def _finalize_training(self, all_losses, epoch, elapsed, log_path) -> dict:
        ckpt_path = self.save_checkpoint(self.step)
        with open(log_path, "w") as f:
            for entry in all_losses:
                f.write(json.dumps(entry) + "\n")

        print(f"\nTraining complete: {self.step} steps in {elapsed:.1f}s")
        if all_losses:
            print(f"Final loss: {all_losses[-1]['L_total']:.4f}")
        else:
            print("No losses recorded")
        print(f"Checkpoint: {ckpt_path}")

        pab_profile_path = None
        tracker = self.infra.pab_tracker
        if tracker is not None:
            profile = tracker.finalize()
            pab_cfg = self.config.get("pab", {})
            if pab_cfg.get("save_profiles", True):
                profile_path = self.run_dir / f"{self.run_id}_pab_profile.json"
                profile.save(profile_path)
                pab_profile_path = str(profile_path)
            print(f"  PAB regime: {profile.summary.stability_regime}")
            print(f"  PAB stability_mean: {profile.summary.stability_mean:.4f}")

        return self._build_training_result(
            self.step, all_losses, epoch, elapsed, str(ckpt_path),
            validation_snapshot=self._tracking["validation_snapshot"] or None,
            pab_profile_path=pab_profile_path,
        )

    def train(self, dry_run: bool = False) -> dict:
        """Main training loop."""
        import time

        from torch.utils.data import DataLoader

        self.setup()

        max_iters = self.config["training"]["max_iterations"]
        safety = self.config["safety"]
        log_path = self.run_dir / f"{self.run_id}{self.config['logging']['training_log_suffix']}"

        bs = self.config["training"]["batch_size"]
        print(f"Training run: {self.run_id}")
        print(f"  Device: {self.device}, Max iters: {max_iters}, Batch: {bs}")
        print(f"  Train examples: {len(self.infra.dataset)}")
        if dry_run:
            print("  [DRY RUN] Will run 1 batch only.")

        start = time.time()
        all_losses: list[dict[str, float]] = []

        loader = DataLoader(  # type: ignore[call-overload]
            self.infra.dataset,
            batch_size=bs,
            shuffle=True,
            collate_fn=lambda batch: batch,
            num_workers=0,
        )

        epoch = 0
        while self.step < max_iters:
            epoch += 1
            for batch in loader:
                if self.step >= max_iters:
                    break

                loss_dict = self.train_step(batch)
                all_losses.append(loss_dict)

                abort = self._run_step_checks(loss_dict, safety)
                if abort is not None:
                    return abort

                if dry_run:
                    print(f"  Dry run complete after 1 step. Loss: {loss_dict['L_total']:.4f}")
                    return {"status": "dry_run", "step": 1, "loss": loss_dict}

        return self._finalize_training(all_losses, epoch, time.time() - start, log_path)


def main():
    parser = argparse.ArgumentParser(description="Balanced Sashimi proof synthesis training")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Run one batch and exit")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = BalancedSashimiTrainer(
        config=config,
        run_id=args.run_id,
        device=args.device,
        seed=args.seed,
    )
    trainer.train(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
