"""Flip ternary decoder weight signs for EXP-6.3b falsification test.

Takes a navigator checkpoint, randomly flips a fraction of the decoder
direction head weight signs, and saves a new checkpoint. Used to test
whether ternary crystallization encodes information or is dead weight.

Usage:
    python scripts/flip_decoder_signs.py \
        --input models/NAV-002_step5000.pt \
        --output models/NAV-002_step5000_flip50.pt \
        --flip-ratio 0.5 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def flip_decoder_signs(
    input_path: Path,
    output_path: Path,
    flip_ratio: float,
    seed: int,
) -> dict:
    """Flip a fraction of decoder direction head weight signs.

    Returns summary of what was flipped.
    """
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    rng = np.random.default_rng(seed)

    nav_state = checkpoint["modules"]["navigator"]
    flipped_count = 0
    total_count = 0

    # Direction heads are keyed as "direction_heads.{bank}.weight" and ".bias"
    for key in list(nav_state.keys()):
        if "direction_heads" in key and key.endswith(".weight"):
            w = nav_state[key].clone()
            mask = torch.from_numpy(rng.random(w.shape) < flip_ratio)
            w[mask] *= -1
            nav_state[key] = w
            flipped_count += int(mask.sum().item())
            total_count += w.numel()

    checkpoint["modules"]["navigator"] = nav_state
    torch.save(checkpoint, output_path)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "flip_ratio": flip_ratio,
        "seed": seed,
        "weights_flipped": flipped_count,
        "weights_total": total_count,
        "actual_flip_pct": round(100 * flipped_count / max(total_count, 1), 1),
    }
    print(
        f"Flipped {flipped_count}/{total_count} direction head weights "
        f"({summary['actual_flip_pct']}%)"
    )
    print(f"Saved to {output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Flip decoder signs for EXP-6.3b")
    parser.add_argument("--input", type=Path, required=True, help="Input checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint")
    parser.add_argument("--flip-ratio", type=float, default=0.5, help="Fraction to flip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    flip_decoder_signs(args.input, args.output, args.flip_ratio, args.seed)


if __name__ == "__main__":
    main()
