"""
Phase 0.6 encoder evaluation — benchmark candidate encoders on Wayfinder goal states.

Standalone worker (ModelAtlas pattern): reads nav_eval.jsonl, reports metrics.

Evaluates:
  1. Tokenizer coverage of math symbols (no UNK tokens for ∀∃→↔⊢ℕℤℝ)
  2. Intra-theorem vs inter-theorem cosine similarity (embedding quality)
  3. Encoding throughput (goals/sec on current device)
  4. Memory footprint (peak RSS delta)

Usage:
    python scripts/eval_encoders.py --eval-data data/nav_eval.jsonl --samples 500
    python scripts/eval_encoders.py --eval-data data/nav_eval.jsonl --model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import resource
from collections import defaultdict
from pathlib import Path

from scripts.encoder_backends import (
    _load_tokenizer,
    encode_goals,
)
from scripts.similarity_metrics import compute_similarity_metrics

# Math symbols that appear in Lean 4 goal states
MATH_SYMBOLS = [
    "∀",
    "∃",
    "→",
    "↔",
    "⊢",
    "ℕ",
    "ℤ",
    "ℝ",
    "≤",
    "≥",
    "∈",
    "∉",
    "⊆",
    "⊇",
    "∪",
    "∩",
    "⟨",
    "⟩",
    "·",
    "▸",
    "λ",
    "←",
    "↦",
    "⁻¹",
    "∘",
    "×",
    "⊥",
    "⊤",
    "¬",
    "∧",
    "∨",
    "⊕",
]

# Candidate models to evaluate — organized by size tier
DEFAULT_CANDIDATES = [
    # --- Baseline ---
    "all-MiniLM-L6-v2",  # 22M, 384d — current placeholder
    # --- Small (100-200M) ---
    "Alibaba-NLP/gte-modernbert-base",  # 149M, 768d — ModernBERT, math+code (ModelAtlas)
    "nomic-ai/modernbert-embed-base",  # 137M, 768d — ModernBERT, Matryoshka, 8K ctx
    # --- Medium (300M-500M) ---
    "google/byt5-small",  # 300M, 1472d — tokenizer-free (PLAN doc)
    "Snowflake/snowflake-arctic-embed-l-v2.0",  # 335M, 1024d — high quality, Matryoshka
    "Salesforce/SFR-Embedding-Code-400M_R",  # 400M, 1024d — code-specialized
    # --- Large (1B-2B) ---
    "dunzhang/stella_en_1.5B_v5",  # 1.5B, 8192d — strong MTEB
    # --- XL (7B-8B) ---
    "intfloat/e5-mistral-7b-instruct",  # 7B, 4096d — Mistral-based, MTEB leader
    "Alibaba-NLP/gte-Qwen2-7B-instruct",  # 7B, 3584d — Qwen2-based
    "nvidia/NV-Embed-v2",  # 7.85B, 4096d — MTEB #1 at release
]

# Extended candidates for Phase 0.6b — math-domain + novel architectures
EXTENDED_CANDIDATES = [
    # --- Math-domain specific ---
    # 218M, ByT5 fine-tuned for Lean 4 premise retrieval
    "kaiyuy/leandojo-lean4-retriever-byt5-small",
    "math-similarity/Bert-MLM_arXiv-MP-class_zbMath",  # 110M, BERT fine-tuned for math similarity
    "tbs17/MathBERT",  # 110M, BERT pre-trained on math curriculum
    # --- New embedding architectures ---
    "Qwen/Qwen3-Embedding-0.6B",  # 596M, Qwen3 causal→embedding, Matryoshka
    "perplexity-ai/pplx-embed-v1-0.6b",  # 596M, bidirectional Qwen3 (novel arch)
    # --- Domain-adapted retriever (7B, slow) ---
    "FrenzyMath/LeanSearch-PS",  # LoRA on e5-mistral-7b for Lean premise selection
]

# Phase 0.6b storage candidates — downloaded to external STORAGE
# Keys are directory names under --model-dir; values are original HF repo IDs (for reference)
STORAGE_CANDIDATES = {
    # Phase 0.6 encoders
    "jina-embeddings-v5-text-small": "jinaai/jina-embeddings-v5-text-small",
    "nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-code": "nomic-ai/nomic-embed-code",
    "ettin-encoder-1b": "jhu-clsp/ettin-encoder-1b",
    "ModernGBERT_1B": "LSX-UniWue/ModernGBERT_1B",
    # SoM planning models
    "DeepSeek-Prover-V1.5-SFT": "deepseek-ai/DeepSeek-Prover-V1.5-SFT",
    "deepseek-math-7b-rl": "deepseek-ai/deepseek-math-7b-rl",
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "stable-code-3b": "stabilityai/stable-code-3b",
    "next-270m": "thelamapi/next-270m",
    # Local reference models (quirky skills + architectures)
    "Qwen3-8B-Drama-Thinking": "Qwen/Qwen3-8B-Drama-Thinking",
    "MPT-7B-StoryWriter": "mosaicml/mpt-7b-storywriter",
    "Cerebrum-1.0-7b": "agentica-org/Cerebrum-1.0-7b",
    "rwkv7-world": "RWKV/rwkv-7-world",
    "nomos-1": "nomos-ai/nomos-1",
    "polymathic-ai": "polymathic-ai/aion-base",
}

# Default root for wayfinder model storage
_DEFAULT_MODEL_ROOT = "/Volumes/STORAGE/wayfinder_models"

# Subdirectories to search under model root
_MODEL_SUBDIRS = ["phase0_encoders", "som_planning", "local_refs", "som_quirky"]


def resolve_model_path(model_name: str, model_dir: str | None) -> str:
    """Resolve model name to local path if available.

    Searches model_dir directly, then all standard subdirectories under the
    wayfinder_models root. Falls back to the original model_name (for HF hub).
    """
    dirs_to_search: list[Path] = []
    if model_dir is not None:
        dirs_to_search.append(Path(model_dir))
    # Also search standard subdirectories under default root
    root = Path(_DEFAULT_MODEL_ROOT)
    if root.is_dir():
        for subdir in _MODEL_SUBDIRS:
            sub = root / subdir
            if sub.is_dir():
                dirs_to_search.append(sub)

    # Check each search directory for a matching model directory
    for dirname in [model_name, model_name.split("/")[-1]]:
        for search_dir in dirs_to_search:
            candidate = search_dir / dirname
            if candidate.is_dir():
                return str(candidate)
    return model_name


def load_goal_states(eval_path: str, sample_size: int) -> tuple[list[str], dict[str, list[int]]]:
    """Load goal states and group indices by theorem_id for similarity eval."""
    goals: list[str] = []
    theorem_groups: dict[str, list[int]] = defaultdict(list)

    with open(eval_path) as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            record = json.loads(line.strip())
            goals.append(record["goal_state"])
            theorem_groups[record["theorem_id"]].append(i)

    return goals, dict(theorem_groups)


def check_tokenizer_coverage(model_name: str) -> dict:
    """Check how the tokenizer handles math symbols — UNK tokens indicate poor coverage."""
    tokenizer = _load_tokenizer(model_name)
    unk_id = tokenizer.unk_token_id

    covered = 0
    uncovered = []
    for sym in MATH_SYMBOLS:
        ids = tokenizer.encode(sym, add_special_tokens=False)
        if unk_id is not None and all(tid == unk_id for tid in ids):
            uncovered.append(sym)
        else:
            covered += 1

    total = len(MATH_SYMBOLS)
    return {
        "covered": covered,
        "total": total,
        "coverage_pct": round(100 * covered / total, 1),
        "uncovered_symbols": uncovered,
    }


def evaluate_model(
    model_name: str,
    goals: list[str],
    theorem_groups: dict[str, list[int]],
    device: str,
) -> dict:
    """Run full evaluation for a single model."""
    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'=' * 60}")

    # 1. Tokenizer coverage
    print("  [1/3] Checking tokenizer coverage...")
    tok_result = check_tokenizer_coverage(model_name)
    print(
        f"    Math symbol coverage: {tok_result['coverage_pct']}%"
        f" ({tok_result['covered']}/{tok_result['total']})"
    )
    if tok_result["uncovered_symbols"]:
        print(f"    Uncovered: {tok_result['uncovered_symbols']}")

    # 2. Encode + throughput
    print("  [2/3] Encoding goal states...")
    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    embeddings, throughput, native_dim = encode_goals(model_name, goals, device)
    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_delta_mb = (mem_after - mem_before) / (1024 * 1024)  # macOS reports bytes
    print(f"    Native dim: {native_dim}, Throughput: {throughput:.1f} goals/sec")
    print(f"    Memory delta: {mem_delta_mb:.0f} MB")

    # 3. Similarity metrics
    print("  [3/3] Computing similarity metrics...")
    sim_result = compute_similarity_metrics(embeddings, theorem_groups)
    print(f"    Intra-theorem sim: {sim_result['intra_theorem_sim']:.4f}")
    print(f"    Inter-theorem sim: {sim_result['inter_theorem_sim']:.4f}")
    print(f"    Separation: {sim_result['separation']:.4f}")

    return {
        "model": model_name,
        "native_dim": native_dim,
        "tokenizer": tok_result,
        "throughput_goals_per_sec": round(throughput, 1),
        "memory_delta_mb": round(mem_delta_mb, 0),
        "similarity": sim_result,
    }


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("  ENCODER COMPARISON SUMMARY")
    print("=" * 90)
    print(
        f"  {'Model':<42} {'Dim':>4} {'Tok%':>5}"
        f" {'Intra':>6} {'Inter':>6} {'Sep':>6} {'Goals/s':>8}"
    )
    print(f"  {'-' * 42} {'-' * 4} {'-' * 5} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 8}")
    for r in sorted(results, key=lambda x: x["similarity"]["separation"], reverse=True):
        print(
            f"  {r['model']:<42} {r['native_dim']:>4} "
            f"{r['tokenizer']['coverage_pct']:>5.1f} "
            f"{r['similarity']['intra_theorem_sim']:>6.4f} "
            f"{r['similarity']['inter_theorem_sim']:>6.4f} "
            f"{r['similarity']['separation']:>6.4f} "
            f"{r['throughput_goals_per_sec']:>8.1f}"
        )
    print("=" * 90)
    print("  Key: Sep = Intra-Inter (higher = better discrimination)")
    print("  Key: Tok% = math symbol tokenizer coverage")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0.6 encoder evaluation")
    parser.add_argument("--eval-data", type=str, default="data/nav_eval.jsonl")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--model", type=str, default=None, help="Evaluate a single model")
    parser.add_argument(
        "--tier",
        type=str,
        default=None,
        choices=["small", "medium", "large", "xl", "all", "extended", "storage"],
        help="Run only models in a size tier (default: small+medium)",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="data/encoder_eval_results.json")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing downloaded models "
        "(e.g. /Volumes/STORAGE/wayfinder_models/phase0_encoders)",
    )
    args = parser.parse_args()

    if not Path(args.eval_data).exists():
        print(f"Error: {args.eval_data} not found")
        return

    print(f"Loading {args.samples} goal states from {args.eval_data}...")
    goals, theorem_groups = load_goal_states(args.eval_data, args.samples)
    print(f"  Loaded {len(goals)} goal states from {len(theorem_groups)} theorems")

    # Tier-based model selection
    tier_map: dict[str, list[str]] = {
        "small": DEFAULT_CANDIDATES[:3],  # baseline + ModernBERT pair
        "medium": DEFAULT_CANDIDATES[3:6],  # byt5 + arctic + SFR-Code
        "large": DEFAULT_CANDIDATES[6:7],  # stella 1.5B
        "xl": DEFAULT_CANDIDATES[7:],  # 7B+ models
        "storage": list(STORAGE_CANDIDATES.keys()),  # models on STORAGE
    }
    if args.model:
        candidates = [args.model]
    elif args.tier == "all":
        candidates = DEFAULT_CANDIDATES + EXTENDED_CANDIDATES + list(STORAGE_CANDIDATES.keys())
    elif args.tier:
        candidates = tier_map.get(args.tier, DEFAULT_CANDIDATES[:6])
    else:
        # Default: small + medium (no multi-GB downloads)
        candidates = DEFAULT_CANDIDATES[:6]

    results = []
    for model_name in candidates:
        resolved = resolve_model_path(model_name, args.model_dir)
        try:
            result = evaluate_model(resolved, goals, theorem_groups, args.device)
            result["model"] = model_name  # keep original name for display
            results.append(result)
        except Exception as e:
            print(f"\n  FAILED: {model_name} ({resolved}): {e}")
            results.append({"model": model_name, "error": str(e)})

    if len(results) > 1:
        successful = [r for r in results if "error" not in r]
        if successful:
            print_comparison_table(successful)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
