# main_summeval.py  — SummEval with §4.3 metric definitions injected
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from Domain.candidate import Candidate
from Domain.round import Round
from Pipeline.round_analysis import compute_round_analysis
from Pipeline.make_pairs import ensure_pairs

# -------------------------
# Global committee role
# -------------------------
MEMBER_NAME: str = "member"

# -------------------------
# Metric-specific settings
# -------------------------
METRIC_CLAIMS = {
    "coherence": "For the same source article, summary {A} is more coherent than summary {B}",
    "consistency": "For the same source article, summary {A} is more consistent than summary {B}",
    "fluency": "Summary {A} is more fluent than summary {B}",
    "relevance": "For the same source article, summary {A} is more relevant than summary {B}",
}

METRIC_TITLES = {
    "coherence": "Summary Coherence Evaluation Committee",
    "consistency": "Summary Consistency Evaluation Committee",
    "fluency": "Summary Fluency Evaluation Committee",
    "relevance": "Summary Relevance Evaluation Committee",
}

# -------------------------
# SummEval §4.3 Definitions (concise, faithful)
# -------------------------
METRIC_DEFINITIONS = {
    "coherence": (
        "Coherence — the collective quality of all sentences. "
        "Aligned with the DUC ‘structure and coherence’ guideline: "
        "the summary should be well-structured and well-organized, building from sentence to sentence "
        "into a coherent body of information about the topic (not just a heap of related facts)."
    ),
    "consistency": (
        "Consistency — factual alignment with the source. "
        "A consistent summary contains only statements entailed by the source document; "
        "penalize hallucinated or contradictory facts."
    ),
    "fluency": (
        "Fluency — quality of individual sentences. "
        "Following DUC guidelines: no formatting/capitalization problems or obvious grammatical errors "
        "(e.g., fragments, missing components) that hinder readability."
    ),
    "relevance": (
        "Relevance — selection of important content from the source. "
        "Include only important information; penalize redundancies and excess/unimportant content."
    ),
}


# -------------------------
# Helpers
# -------------------------
def escape_braces(s: str) -> str:
    """Make curly braces literal for f-strings / prompt templates."""
    return s.replace("{", "{{").replace("}", "}}")


def load_candidates_from_dir(metric_dir: Path) -> List[Candidate]:
    """Load all Candidate JSON files from a per-metric directory."""
    out: List[Candidate] = []
    for f in sorted(metric_dir.glob("*.json")):
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
            out.append(Candidate.model_validate(obj))
        except Exception as e:
            print(f"[warn] failed to load Candidate from {f.name}: {e}")
    return out


def build_committee_context(metric: str, safe_article: Optional[str]) -> str:
    """
    Build the committee prompt:
      • Title + decision target
      • Official SummEval §4.3 definition for the chosen metric
      • Source article (reference) for all metrics except fluency (kept per your prior behavior)
    """
    title = METRIC_TITLES[metric]
    base = f"""
    This is the {title}.
    The committee must decide which of two candidate summaries better satisfies the metric: {metric}.
    """.strip()

    definition = METRIC_DEFINITIONS[metric]
    parts = [base, "Metric definition:\n" + escape_braces(definition)]

    if metric != "fluency" and safe_article:
        parts.append("Source article (reference):\n" + escape_braces(safe_article))

    return "\n\n".join(parts)


def print_match_summary(prefix: str, m) -> None:
    if not m:
        print(f"{prefix}: (no data)")
        return
    if getattr(m, "sum_all", None):
        print(
            f"{prefix}  Spearman ρ (Sum over All): {m.sum_all.spearman_rho:.3f} (p={m.sum_all.p_value:.3f})"
        )
    if getattr(m, "poe_gaussian", None):
        print(
            f"{prefix}  Spearman ρ (PoE-Gaussian): {m.poe_gaussian.spearman_rho:.3f} (p={m.poe_gaussian.p_value:.3f})"
        )
    if getattr(m, "lse", None):
        print(
            f"{prefix}  LSE ρ (Least Square Error): {m.lse.spearman_rho:.3f} (p={m.lse.p_value:.3f})"
        )
    if getattr(m, "positionalBias", None) is not None:
        print(f"{prefix}  Positional bias (ΣA − ΣB): {m.positionalBias:.3f}")


# -------------------------
# Main (argparse)
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pairwise comparisons on a single SummEval article/metric folder (with §4.3 definitions)."
    )
    parser.add_argument(
        "--metric_dir",
        type=str,
        required=True,
        help="Path to one metric folder (e.g., cnn/<article_id>/coherence).",
    )
    parser.add_argument(
        "--round_root",
        type=str,
        required=True,
        help="Output root directory; results saved per article+metric.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help=(
            "If set, number of directed comparisons to run (efficient mode). "
            "If omitted, run full O(N^2) matrix."
        ),
    )
    args = parser.parse_args()

    metric_dir = Path(args.metric_dir)
    round_root = Path(args.round_root)
    round_root.mkdir(parents=True, exist_ok=True)

    metric = metric_dir.name.lower()
    if metric not in METRIC_CLAIMS:
        raise SystemExit(f"Unknown metric folder: {metric}")

    # Load candidates
    candidates: List[Candidate] = load_candidates_from_dir(metric_dir)
    if not candidates:
        raise SystemExit(f"No candidate JSONs found in: {metric_dir}")

    # Load article text from parent folder
    article_path = metric_dir.parent / "article.txt"
    article_text = (
        article_path.read_text(encoding="utf-8") if article_path.exists() else ""
    )
    safe_article = escape_braces(article_text) if article_text else ""

    # Context + pair reference
    if metric == "fluency":
        committee_context_template = build_committee_context(metric, None)
        pair_reference = None
    else:
        committee_context_template = build_committee_context(metric, safe_article)
        pair_reference = article_text  # raw text for the agents; not brace-escaped

    metric_round_dir = round_root / metric
    metric_round_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Creating round for metric={metric} at: {metric_round_dir}")
    rnd = Round.create(
        metric_round_dir,
        candidates,
        committee_context_template=committee_context_template,
        member_name=MEMBER_NAME,
        claim_template=METRIC_CLAIMS[metric],
        budget=args.budget,
    )

    # Build pairs
    pairs = ensure_pairs(
        rnd,
        reference=pair_reference,
        K=args.budget,
    )
    rnd.pairs = pairs
    print(f"[info] Pair artifacts ready: {len(pairs)}")

    # Compute + Save
    rnd.analysis = compute_round_analysis(rnd, pairs)
    saved_path = rnd.save()
    print(f"[done] Round saved: {saved_path}")

    if rnd.analysis:
        print_match_summary("Initial:", rnd.analysis.InitialArgumentOnly)
        print_match_summary("Reply:  ", rnd.analysis.ArgumentWithReply)

    print("[all done]")


if __name__ == "__main__":
    main()

"""
python main_summeval.py \
  --metric_dir /train-data1/shaosen/SummEvalDataSet/SummEval/dataFolder/cnn/88c2481234e763c9bbc68d0ab1be1d2375c1349a/coherence \
  --round_root /home/shaosen/LLM/Multi-agent/Saving/SummEval/cnn_7/88c2481234e763c9bbc68d0ab1be1d2375c1349a \
  --budget 48
"""
