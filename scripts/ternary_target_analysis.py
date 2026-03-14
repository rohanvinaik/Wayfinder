#!/usr/bin/env python3
"""Phase 7.1c: Ternary Target Distribution Analysis.

Analyzes nav_directions from training data to characterize the
ternary direction space: per-bank distributions, 729-bin occupancy,
OTP dimensionality, and bank co-activation patterns.
"""

import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

BANKS = ["structure", "domain", "depth", "automation", "context", "decomposition"]
MAX_LINES = 50_000
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nav_train.jsonl"


def load_directions(path: Path, max_lines: int) -> list[dict[str, int]]:
    """Load nav_directions from first max_lines of training data."""
    directions = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            obj = json.loads(line)
            directions.append(obj["nav_directions"])
    return directions


def per_bank_distribution(directions: list[dict[str, int]]) -> None:
    """Section 1: Per-bank ternary value distribution."""
    n = len(directions)
    print("=" * 72)
    print("1. PER-BANK TERNARY DISTRIBUTION")
    print("=" * 72)
    print(f"   (N = {n:,} examples, first {MAX_LINES:,} lines of nav_train.jsonl)")
    print()
    print(
        f"{'Bank':<16} {'  -1':>8} {'   0':>8} {'  +1':>8}"
        f"  {'%-1':>6} {'%0':>6} {'%+1':>6}  Zero-Heavy?"
    )
    print("-" * 90)

    zero_heavy_banks = []
    for bank in BANKS:
        counts = Counter(d[bank] for d in directions)
        c_neg = counts.get(-1, 0)
        c_zero = counts.get(0, 0)
        c_pos = counts.get(1, 0)
        p_neg = 100.0 * c_neg / n
        p_zero = 100.0 * c_zero / n
        p_pos = 100.0 * c_pos / n
        is_zero_heavy = p_zero > 50.0
        marker = "YES" if is_zero_heavy else "no"
        if is_zero_heavy:
            zero_heavy_banks.append(bank)
        print(
            f"{bank:<16} {c_neg:>8,} {c_zero:>8,} {c_pos:>8,}"
            f"  {p_neg:>5.1f}% {p_zero:>5.1f}% {p_pos:>5.1f}%  {marker}"
        )

    print()
    if zero_heavy_banks:
        print(f"   Zero-heavy banks (>50% zero): {', '.join(zero_heavy_banks)}")
    else:
        print("   No banks are zero-heavy (all have <50% zero).")
    print()


def direction_bin_analysis(directions: list[dict[str, int]]) -> None:
    """Section 2: 729-bin direction space analysis."""
    n = len(directions)
    print("=" * 72)
    print("2. 729-BIN DIRECTION SPACE")
    print("=" * 72)
    print()

    # Convert each example to a tuple for counting
    bin_counter: Counter[tuple[int, ...]] = Counter()
    for d in directions:
        key = tuple(d[bank] for bank in BANKS)
        bin_counter[key] += 1

    n_unique = len(bin_counter)
    occupancy_pct = 100.0 * n_unique / 729
    print(f"   Unique bins occupied: {n_unique} / 729 ({occupancy_pct:.1f}%)")
    print(f"   Empty bins: {729 - n_unique}")
    print()

    # Entropy
    probs = [count / n for count in bin_counter.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(729)
    print(f"   Shannon entropy: {entropy:.3f} bits (max = {max_entropy:.3f} bits)")
    print(f"   Normalized entropy: {entropy / max_entropy:.3f}")
    print()

    # Top 20 bins
    print("   Top 20 most frequent bins:")
    print(
        f"   {'Rank':<6} {'S':>3} {'D':>3} {'Dp':>3} {'A':>3} {'C':>3} {'Dc':>3}"
        f"   {'Count':>7} {'%':>6}  {'Cum%':>6}"
    )
    print("   " + "-" * 55)
    cum = 0.0
    for rank, (key, count) in enumerate(bin_counter.most_common(20), 1):
        pct = 100.0 * count / n
        cum += pct
        v_s, v_d, v_dp, v_a, v_c, v_dc = key

        # Format each value with +/- sign
        def fmt(v: int) -> str:
            if v == 1:
                return "+1"
            elif v == -1:
                return "-1"
            return " 0"

        print(
            f"   {rank:<6} {fmt(v_s):>3} {fmt(v_d):>3} {fmt(v_dp):>3}"
            f" {fmt(v_a):>3} {fmt(v_c):>3} {fmt(v_dc):>3}"
            f"   {count:>7,} {pct:>5.1f}%  {cum:>5.1f}%"
        )

    print()
    # Report concentration
    top5_pct = sum(count for _, count in bin_counter.most_common(5)) / n * 100
    top10_pct = sum(count for _, count in bin_counter.most_common(10)) / n * 100
    top20_pct = sum(count for _, count in bin_counter.most_common(20)) / n * 100
    top50_pct = sum(count for _, count in bin_counter.most_common(50)) / n * 100
    print(
        f"   Concentration: top-5={top5_pct:.1f}%,"
        f" top-10={top10_pct:.1f}%,"
        f" top-20={top20_pct:.1f}%,"
        f" top-50={top50_pct:.1f}%"
    )
    print()

    # Singleton and rare bins
    singletons = sum(1 for c in bin_counter.values() if c == 1)
    rare = sum(1 for c in bin_counter.values() if c <= 5)
    print(f"   Singletons (count=1): {singletons} bins")
    print(f"   Rare bins (count<=5): {rare} bins")
    print()


def otp_dimensionality(directions: list[dict[str, int]]) -> None:
    """Section 3: Active dimensionality distribution."""
    n = len(directions)
    print("=" * 72)
    print("3. OTP DIMENSIONALITY (Active Non-Zero Banks per Example)")
    print("=" * 72)
    print()

    dim_counter: Counter[int] = Counter()
    for d in directions:
        active = sum(1 for bank in BANKS if d[bank] != 0)
        dim_counter[active] += 1

    print(f"   {'Dim':>4}  {'Count':>8}  {'%':>6}  Bar")
    print("   " + "-" * 50)
    for dim in range(7):
        count = dim_counter.get(dim, 0)
        pct = 100.0 * count / n
        bar = "#" * int(pct / 2)
        print(f"   {dim:>4}  {count:>8,}  {pct:>5.1f}%  {bar}")

    print()
    # Mean and median
    dims = []
    for d in directions:
        dims.append(sum(1 for bank in BANKS if d[bank] != 0))
    mean_dim = sum(dims) / len(dims)
    sorted_dims = sorted(dims)
    median_dim = sorted_dims[len(sorted_dims) // 2]
    print(f"   Mean active dimensionality: {mean_dim:.2f}")
    print(f"   Median active dimensionality: {median_dim}")
    print()

    # Fraction in low-dim subspace
    low_dim = sum(dim_counter.get(d, 0) for d in range(4))
    high_dim = sum(dim_counter.get(d, 0) for d in range(4, 7))
    print(f"   Low-dim (0-3 banks): {low_dim:,} ({100.0 * low_dim / n:.1f}%)")
    print(f"   High-dim (4-6 banks): {high_dim:,} ({100.0 * high_dim / n:.1f}%)")
    print()


def _print_coactivation_details(
    abbrev: dict[str, str],
    activation: dict[str, int],
    coactivation: dict[tuple[str, str], int],
    n: int,
) -> None:
    """Print co-activation matrix, strongest/weakest pairs, and lift analysis."""
    header = "   " + " " * 16 + "  ".join(f"{abbrev[b]:>5}" for b in BANKS)
    print(header)
    print("   " + "-" * (16 + 7 * 6))

    for b1 in BANKS:
        row = f"   {abbrev[b1] + ' ' + b1:<16}"
        for b2 in BANKS:
            if b1 == b2:
                row += f"{'---':>7}"
            else:
                key = (b1, b2) if (b1, b2) in coactivation else (b2, b1)
                rate = 100.0 * coactivation[key] / n
                row += f"{rate:>6.1f}%"
        print(row)

    print()

    sorted_pairs = sorted(coactivation.items(), key=lambda x: x[1], reverse=True)
    print("   Strongest co-activation pairs:")
    for (b1, b2), count in sorted_pairs[:5]:
        rate = 100.0 * count / n
        print(f"     {b1} + {b2}: {rate:.1f}%")

    print()
    print("   Weakest co-activation pairs:")
    for (b1, b2), count in sorted_pairs[-5:]:
        rate = 100.0 * count / n
        print(f"     {b1} + {b2}: {rate:.1f}%")

    print()

    # Lift analysis
    print("   Co-activation lift (observed / expected if independent):")
    print(f"   {'Pair':<35} {'Obs%':>6} {'Exp%':>6} {'Lift':>6}")
    print("   " + "-" * 55)
    lifts = []
    for (b1, b2), count in sorted_pairs:
        obs_rate = count / n
        exp_rate = (activation[b1] / n) * (activation[b2] / n)
        lift = obs_rate / exp_rate if exp_rate > 0 else float("inf")
        lifts.append(((b1, b2), obs_rate, exp_rate, lift))

    lifts.sort(key=lambda x: x[3], reverse=True)
    for (b1, b2), obs, exp, lift in lifts:
        print(f"   {b1 + ' + ' + b2:<35} {obs * 100:>5.1f}% {exp * 100:>5.1f}% {lift:>5.2f}x")

    print()


def bank_coactivation(directions: list[dict[str, int]]) -> None:
    """Section 4: Pairwise bank co-activation rates."""
    n = len(directions)
    print("=" * 72)
    print("4. BANK CO-ACTIVATION (Pairwise)")
    print("=" * 72)
    print()
    print("   Co-activation = P(both banks non-zero)")
    print()

    # Individual activation rates first
    activation: dict[str, int] = {bank: 0 for bank in BANKS}
    coactivation: dict[tuple[str, str], int] = {}
    for b1, b2 in combinations(BANKS, 2):
        coactivation[(b1, b2)] = 0

    for d in directions:
        active_banks = [bank for bank in BANKS if d[bank] != 0]
        for bank in active_banks:
            activation[bank] += 1
        for b1, b2 in combinations(active_banks, 2):
            # Ensure canonical order
            if (b1, b2) in coactivation:
                coactivation[(b1, b2)] += 1
            elif (b2, b1) in coactivation:
                coactivation[(b2, b1)] += 1

    print("   Individual activation rates (P(bank != 0)):")
    print(f"   {'Bank':<16} {'Active':>8} {'Rate':>7}")
    print("   " + "-" * 35)
    for bank in BANKS:
        rate = 100.0 * activation[bank] / n
        print(f"   {bank:<16} {activation[bank]:>8,} {rate:>6.1f}%")

    print()
    print("   Pairwise co-activation matrix:")
    # Print as a matrix
    abbrev = {
        "structure": "S",
        "domain": "D",
        "depth": "Dp",
        "automation": "A",
        "context": "C",
        "decomposition": "Dc",
    }
    _print_coactivation_details(abbrev, activation, coactivation, n)


def main() -> None:
    print()
    print("=" * 72)
    print("  PHASE 7.1c: TERNARY TARGET DISTRIBUTION ANALYSIS")
    print("=" * 72)
    print()

    print(f"Data: {DATA_PATH}")
    directions = load_directions(DATA_PATH, MAX_LINES)
    print(f"Loaded {len(directions):,} examples (first {MAX_LINES:,} lines)")
    print()

    per_bank_distribution(directions)
    direction_bin_analysis(directions)
    otp_dimensionality(directions)
    bank_coactivation(directions)

    # Summary
    print("=" * 72)
    print("SUMMARY & DESIGN DOC PREDICTIONS")
    print("=" * 72)
    print()
    print("Predictions from WAYFINDER_RESEARCH.md / WAYFINDER_DESIGN.md:")
    print()
    print("  1. 'Zero should be the majority for most banks (Informational Zero)'")
    print("     → See Section 1 above.")
    print()
    print("  2. 'The 729-bin space should be very sparse'")
    print("     → See Section 2 above.")
    print()
    print("  3. 'Sparser bins carry higher discrimination'")
    print("     → Bins with fewer examples represent more specialized proof")
    print("       strategies. See rare/singleton counts in Section 2.")
    print()


if __name__ == "__main__":
    main()
