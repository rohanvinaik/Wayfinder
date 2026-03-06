"""Setup and construction logic for the proof synthesis training pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.pab_tracker import PABTracker

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_vocabs(config: dict) -> tuple[dict, dict]:
    """Load tier1 (tactic) and tier2 (premise) vocabularies."""
    data_cfg = config["data"]
    tier1_vocab = json.loads((PROJECT_ROOT / data_cfg["tier1_vocab"]).read_text())
    tier2_vocab_path = PROJECT_ROOT / data_cfg.get("tier2_vocab", "")
    if tier2_vocab_path.exists():
        tier2_vocab = json.loads(tier2_vocab_path.read_text())
    else:
        tier2_vocab = {}
    return tier1_vocab, tier2_vocab


def build_pipeline_modules(
    config: dict,
    device: str,
    encoder_override: object | None = None,
) -> tuple[tuple, str]:
    """Construct model components.

    Returns ((encoder, gate, analyzer, bridge, decoder), output_mode).
    """
    from src.bridge import InformationBridge
    from src.domain_gate import DomainGate
    from src.encoder import GoalEncoder
    from src.goal_analyzer import GoalAnalyzer
    from src.ternary_decoder import TernaryDecoder

    model_cfg = config["model"]
    decoder_cfg = model_cfg["decoder"]
    tier1_vocab, tier2_vocab = load_vocabs(config)

    encoder = (
        encoder_override
        if encoder_override is not None
        else GoalEncoder(model_name=model_cfg["encoder"]["model_name"], device=device)
    )
    modules = (
        encoder,
        DomainGate(
            input_dim=model_cfg["encoder"]["output_dim"],
            hidden_dim=model_cfg["domain_gate"]["hidden_dim"],
        ).to(device),
        GoalAnalyzer(
            input_dim=model_cfg["encoder"]["output_dim"],
            feature_dim=model_cfg["goal_analyzer"]["feature_dim"],
        ).to(device),
        InformationBridge(
            input_dim=model_cfg["goal_analyzer"]["feature_dim"],
            bridge_dim=model_cfg["bridge"]["bridge_dim"],
        ).to(device),
        TernaryDecoder(
            input_dim=model_cfg["bridge"]["bridge_dim"],
            hidden_dim=decoder_cfg["hidden_dim"],
            tier1_vocab_size=len(tier1_vocab),
            tier2_vocab_size=len(tier2_vocab) if tier2_vocab else decoder_cfg.get("tier2_vocab_size", 256),
            num_layers=decoder_cfg["num_layers"],
            ternary_enabled=decoder_cfg.get("ternary_enabled", True),
            partial_ternary=decoder_cfg.get("partial_ternary", False),
        ).to(device),
    )
    return modules, decoder_cfg.get("output_mode", "proof")


def build_pab_tracker(config: dict, run_id: str) -> PABTracker | None:
    """Create a PABTracker if enabled in config."""
    pab_cfg = config.get("pab", {})
    if not pab_cfg.get("enabled", False):
        return None
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    return PABTracker(
        experiment_id=run_id,
        config_hash=config_hash,
        checkpoint_interval=pab_cfg.get("checkpoint_interval", 50),
    )


def load_datasets(config: dict) -> tuple[Any, Any, list[Any], list[Any]]:
    """Load training dataset, negative bank dataset, OOD examples, and eval examples."""
    from src.data import NegativeBankDataset, ProofDataset, load_ood_prompts_jsonl

    data_cfg = config["data"]
    train_dataset = ProofDataset(PROJECT_ROOT / data_cfg["proof_train"])

    neg_dataset = None
    negative_bank_path = PROJECT_ROOT / data_cfg.get("negative_bank", "")
    if negative_bank_path.exists():
        neg_dataset = NegativeBankDataset(negative_bank_path)

    ood_examples: list[Any] = []
    ood_path = PROJECT_ROOT / data_cfg.get("ood_prompts", "")
    if ood_path.exists():
        ood_examples = load_ood_prompts_jsonl(ood_path)

    eval_examples: list[Any] = []
    eval_path = PROJECT_ROOT / data_cfg.get("proof_eval", "")
    if eval_path.exists():
        eval_examples = ProofDataset(eval_path).examples

    return train_dataset, neg_dataset, ood_examples, eval_examples


def build_losses(config: dict, device: str) -> tuple:
    """Create composite loss and OOD loss."""
    from src.losses import CompositeLoss, OODLoss

    loss_cfg = config["training"].get("loss", {})
    composite = CompositeLoss(
        initial_log_sigma=loss_cfg.get("initial_log_sigma", 0.0),
        margin=loss_cfg.get("margin", 0.5),
    ).to(device)
    ood = OODLoss().to(device)
    return composite, ood


def setup_run_dirs(config: dict, run_id: str) -> tuple[Path, Path]:
    """Create and return (run_dir, checkpoint_dir)."""
    run_dir = PROJECT_ROOT / config["logging"]["run_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = PROJECT_ROOT / config["logging"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, checkpoint_dir
