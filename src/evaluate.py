"""Evaluation harness for Balanced Sashimi proof synthesis checkpoints.

Usage:
    python src/evaluate.py \
      --config configs/base.yaml \
      --checkpoint path/to/ckpt.pt \
      --eval-file data/proof_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import yaml

from src.behavioral_fingerprint import BehavioralFingerprint

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _VocabInfo(NamedTuple):
    tier1_vocab: dict[str, int]
    tier1_idx2token: dict[int, str]
    tier2_vocab_size: int


class _EvalPipeline(NamedTuple):
    encoder: Any
    domain_gate: Any
    goal_analyzer: Any
    bridge: Any
    decoder: Any
    checkpoint_meta: dict[str, Any]


def _load_vocabs(config: dict) -> _VocabInfo:
    data_cfg = config["data"]
    tier1_vocab = json.loads((PROJECT_ROOT / data_cfg["tier1_vocab"]).read_text())
    tier2_path = PROJECT_ROOT / data_cfg.get("tier2_vocab", "")
    tier2_vocab = json.loads(tier2_path.read_text()) if tier2_path.exists() else {}
    tier1_idx2token = {v: k for k, v in tier1_vocab.items()}
    return _VocabInfo(
        tier1_vocab=tier1_vocab,
        tier1_idx2token=tier1_idx2token,
        tier2_vocab_size=len(tier2_vocab),
    )


def _build_pipeline(config: dict, checkpoint_path: Path, vocab: _VocabInfo) -> _EvalPipeline:
    import torch

    from src.bridge import InformationBridge
    from src.domain_gate import DomainGate
    from src.encoder import GoalEncoder
    from src.goal_analyzer import GoalAnalyzer
    from src.ternary_decoder import TernaryDecoder

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_config = ckpt.get("config", {})
    if isinstance(checkpoint_config, dict) and isinstance(checkpoint_config.get("model"), dict):
        model_cfg = checkpoint_config["model"]
    else:
        model_cfg = config["model"]
    decoder_cfg = model_cfg["decoder"]

    encoder = GoalEncoder.from_config(model_cfg["encoder"], device="cpu")
    encoder.ensure_loaded()
    domain_gate = DomainGate(
        input_dim=encoder.output_dim,
        hidden_dim=model_cfg["domain_gate"]["hidden_dim"],
    )
    goal_analyzer = GoalAnalyzer(
        input_dim=encoder.output_dim,
        feature_dim=model_cfg["goal_analyzer"]["feature_dim"],
    )
    bridge = InformationBridge(
        input_dim=model_cfg["goal_analyzer"]["feature_dim"],
        bridge_dim=model_cfg["bridge"]["bridge_dim"],
    )
    decoder = TernaryDecoder(
        input_dim=model_cfg["bridge"]["bridge_dim"],
        hidden_dim=decoder_cfg["hidden_dim"],
        tier1_vocab_size=len(vocab.tier1_vocab),
        tier2_vocab_size=vocab.tier2_vocab_size or decoder_cfg.get("tier2_vocab_size", 256),
        num_layers=decoder_cfg["num_layers"],
        ternary_enabled=decoder_cfg.get("ternary_enabled", True),
        partial_ternary=decoder_cfg.get("partial_ternary", False),
    )

    domain_gate.load_state_dict(ckpt["domain_gate"])
    goal_analyzer.load_state_dict(ckpt["goal_analyzer"])
    bridge.load_state_dict(ckpt["bridge"])
    decoder.load_state_dict(ckpt["decoder"])

    for module in [domain_gate, goal_analyzer, bridge, decoder]:
        module.eval()

    return _EvalPipeline(
        encoder=encoder,
        domain_gate=domain_gate,
        goal_analyzer=goal_analyzer,
        bridge=bridge,
        decoder=decoder,
        checkpoint_meta=ckpt,
    )


def _target_tier1_token(example: Any) -> str:
    tokens = getattr(example, "tier1_tokens", [])
    if not tokens:
        return "<UNK>"
    return tokens[1] if len(tokens) > 1 else tokens[0]


def _infer_domain(example: Any) -> str:
    from src.trainer_constants import infer_domain

    return infer_domain(example)


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: Path,
    eval_file: Path,
    output_json: Path | None = None,
) -> dict:
    """Evaluate a trained checkpoint against proof eval data."""
    import torch

    from src.data import load_proofs_jsonl

    examples = load_proofs_jsonl(eval_file)
    vocab = _load_vocabs(config)
    pipeline = _build_pipeline(config, checkpoint_path, vocab)
    eval_cfg = config.get("evaluation", {})
    fingerprint_enabled = bool(eval_cfg.get("fingerprint_enabled", False))

    tier1_correct = 0
    domain_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    tactic_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    logits_rows: list[np.ndarray] = []
    action_predictions: list[str] = []
    probe_labels: list[str] = []

    total = len(examples)
    print(f"Evaluating {total} examples...")

    with torch.no_grad():
        for i, ex in enumerate(examples):
            expected = _target_tier1_token(ex)
            domain = _infer_domain(ex)

            emb = pipeline.encoder.encode([ex.goal_state])
            frame = pipeline.goal_analyzer(emb)
            bridge_out = pipeline.bridge(frame)
            dec_out = pipeline.decoder(bridge_out)
            tier1_logits = dec_out["tier1_logits"]
            pred_idx = int(tier1_logits.argmax(dim=-1).item())
            pred_tactic = vocab.tier1_idx2token.get(pred_idx, "<UNK>")

            correct = pred_tactic == expected
            tier1_correct += int(correct)

            domain_stats[domain]["total"] += 1
            domain_stats[domain]["correct"] += int(correct)
            tactic_stats[expected]["total"] += 1
            tactic_stats[expected]["correct"] += int(correct)

            if fingerprint_enabled:
                logits_rows.append(tier1_logits.squeeze(0).cpu().numpy())
                action_predictions.append(pred_tactic)
                probe_labels.append(ex.goal_state[:80])

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{total} evaluated...")

    denom = total if total > 0 else 1
    results: dict[str, Any] = {
        "total_examples": total,
        "checkpoint": str(checkpoint_path),
        "eval_file": str(eval_file),
        "tier1_exact_match_rate": tier1_correct / denom,
        "domain_accuracy": {
            d: (v["correct"] / v["total"] if v["total"] else 0.0)
            for d, v in sorted(domain_stats.items())
        },
        "tactic_accuracy": {
            a: (v["correct"] / v["total"] if v["total"] else 0.0)
            for a, v in sorted(tactic_stats.items())
        },
    }

    if fingerprint_enabled and logits_rows:
        fingerprint = BehavioralFingerprint.from_outputs(
            experiment_id=pipeline.checkpoint_meta.get("run_id", "unknown"),
            step=pipeline.checkpoint_meta.get("step", 0),
            output_logits=np.stack(logits_rows),
            action_predictions=action_predictions,
            probe_labels=probe_labels,
        )
        results["fingerprint_entropy"] = fingerprint.action_entropy
        results["fingerprint_discreteness"] = fingerprint.discreteness_score

        if eval_cfg.get("fingerprint_save", False) and output_json:
            fp_path = Path(output_json).with_suffix(".fingerprint.json")
            fingerprint.save(fp_path)

    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Results written to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Balanced Sashimi checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    config_data = yaml.safe_load(open(args.config))
    results = evaluate_checkpoint(
        config=config_data,
        checkpoint_path=args.checkpoint,
        eval_file=args.eval_file,
        output_json=args.output_json,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
