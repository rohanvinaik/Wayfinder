"""Data loading and dataset classes for proof synthesis training.

Includes legacy Balanced Sashimi datasets and Wayfinder NavigationalDataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from src.contracts import NegativeBankEntry, OODPrompt, ProofExample
from src.nav_contracts import NavigationalExample


def load_proofs_jsonl(path: Path) -> list[ProofExample]:
    """Load ProofExample records from a JSONL file."""
    examples = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                examples.append(ProofExample.from_dict(d))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Malformed record at line {line_num}: {e}") from e
    return examples


def save_proofs_jsonl(examples: list[ProofExample], path: Path) -> int:
    """Save ProofExample records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return len(examples)


def load_ood_prompts_jsonl(path: Path) -> list[OODPrompt]:
    """Load OODPrompt records from a JSONL file."""
    prompts: list[OODPrompt] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                prompts.append(OODPrompt.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Malformed OOD prompt record at line {line_num}: {e}") from e
    return prompts


class ProofDataset(Dataset):
    """PyTorch Dataset wrapping ProofExample records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.examples = load_proofs_jsonl(path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ProofExample:  # type: ignore[override]
        return self.examples[idx]


class NegativeBankDataset(Dataset):
    """PyTorch Dataset wrapping NegativeBankEntry records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[NegativeBankEntry] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.entries.append(NegativeBankEntry.from_dict(json.loads(line)))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> NegativeBankEntry:  # type: ignore[override]
        return self.entries[idx]


class OODPromptDataset(Dataset):
    """PyTorch Dataset wrapping OODPrompt records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.prompts = load_ood_prompts_jsonl(path)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> OODPrompt:  # type: ignore[override]
        return self.prompts[idx]


# ---------------------------------------------------------------------------
# Wayfinder navigational dataset
# ---------------------------------------------------------------------------


def load_nav_examples_jsonl(path: Path) -> list[NavigationalExample]:
    """Load NavigationalExample records from a JSONL file."""
    examples: list[NavigationalExample] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(NavigationalExample.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Malformed nav example at line {line_num}: {e}") from e
    return examples


class NavigationalDataset(Dataset):
    """PyTorch Dataset for Wayfinder navigational training data.

    Supports curriculum filtering by max proof length.

    Args:
        path: Path to nav_training.jsonl.
        max_steps: Only include examples from proofs with <= max_steps total.
            None means no filtering (all examples).
    """

    def __init__(self, path: Path, max_steps: int | None = None) -> None:
        self.path = path
        all_examples = load_nav_examples_jsonl(path)
        if max_steps is not None:
            self.examples = [e for e in all_examples if e.total_steps <= max_steps]
        else:
            self.examples = all_examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> NavigationalExample:  # type: ignore[override]
        return self.examples[idx]
