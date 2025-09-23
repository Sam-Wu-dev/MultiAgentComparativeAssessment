# main_topicalchat.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from Domain.candidate import Candidate
from Domain.round import Round
from Pipeline.round_analysis import compute_round_analysis
from Pipeline.make_pairs import ensure_pairs

# -------------------------
# Global committee role
# -------------------------
MEMBER_NAME: str = "member"

# -------------------------
# Metric-specific settings (TopicalChat)
# -------------------------
METRIC_CLAIMS = {
    # Dialog-friendly claim templates (A/B are system responses for the same context)
    "natural": "Given the same dialogue context, {A} is more natural than {B}",
    "maintains_context": "Given the same dialogue context, {A} maintains context better than {B}",
    "engaging": "Given the same dialogue context, {A} is more engaging than {B}",
    "uses_knowledge": "Given the same dialogue context, {A} uses knowledge better than {B}",
}

NATURAL_DEFINITION = """
Judge NATURAL purely by linguistic naturalness and human-likeness:
- Fluent, coherent phrasing; appropriate tone and wording.
- Penalize awkward, robotic, or ungrammatical expressions.
- Do NOT consider factual correctness beyond clear linguistic plausibility.
"""

MAINTAINS_CONTEXT_DEFINITION = """
Judge MAINTAINS CONTEXT by how well the response stays consistent with the dialogue history:
- Correctly references entities, intents, and prior turns.
- Avoids contradictions or ignoring user’s last message.
- Penalize off-topic or context-agnostic replies.
"""

ENGAGING_DEFINITION = """
Judge ENGAGING by how much the response invites further conversation:
- Interesting, specific, and responsive; asks good follow-ups when appropriate.
- Avoids bland, generic filler; shows curiosity or personality while staying appropriate.
"""

USES_KNOWLEDGE_DEFINITION = """
Judge USES KNOWLEDGE by whether the response appropriately leverages relevant facts:
- Brings in accurate, on-topic knowledge tied to the user’s interest.
- Avoids hallucinations or irrelevant trivia.
- When knowledge is unnecessary, do not penalize concise, correct responses.
"""

METRIC_TITLES = {
    "natural": "TopicalChat Naturalness Committee",
    "maintains_context": "TopicalChat Context-Consistency Committee",
    "engaging": "TopicalChat Engagement Committee",
    "uses_knowledge": "TopicalChat Knowledge-Use Committee",
}


def _metric_definition(metric: str) -> str:
    return {
        "natural": NATURAL_DEFINITION,
        "maintains_context": MAINTAINS_CONTEXT_DEFINITION,
        "engaging": ENGAGING_DEFINITION,
        "uses_knowledge": USES_KNOWLEDGE_DEFINITION,
    }[metric]


def build_committee_context(metric: str, safe_dialogue: str | None) -> str:
    """
    Build the committee context for a TopicalChat metric.
    We include the dialogue reference for metrics that depend on context
    (maintains_context, uses_knowledge, engaging), but not for pure naturalness.
    """
    title = METRIC_TITLES[metric]
    definition = _metric_definition(metric)
    base = f"""
    We are in the {title}.
    We must decide which of two candidate responses better satisfies the metric: {metric}.
    We are deciding between {{A}} and {{B}}.

    Authoritative metric definition:
    {definition}
    Only evaluate the candidates on the given metric.
    """.strip()

    needs_context = metric in {"maintains_context", "uses_knowledge", "engaging"}
    if needs_context and safe_dialogue:
        return f"""{base}

    Conversation context (reference):
    {safe_dialogue}
    """.rstrip()
    else:
        return base


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pairwise comparisons on a single TopicalChat context/metric folder."
    )
    parser.add_argument(
        "--metric_dir",
        type=str,
        required=True,
        help="Path to one metric folder (e.g., topicalchat/<ctx_id>/engaging).",
    )
    parser.add_argument(
        "--round_root",
        type=str,
        required=True,
        help="Output root directory; results saved per context+metric.",
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

    # metric is the last folder name
    metric = metric_dir.name.lower()
    if metric not in METRIC_CLAIMS:
        raise SystemExit(f"Unknown metric folder: {metric}")

    # load candidates (MX.json files)
    candidates: List[Candidate] = load_candidates_from_dir(metric_dir)
    if not candidates:
        raise SystemExit(f"No candidate JSONs found in: {metric_dir}")

    # load conversation context from parent folder (context.txt)
    ctx_path = metric_dir.parent / "context.txt"
    ctx_text = ctx_path.read_text(encoding="utf-8") if ctx_path.exists() else ""
    safe_ctx = ctx_text.replace("{", "{{").replace("}", "}}")

    # context template + whether to pass reference into pair
    needs_context = metric in {"maintains_context", "uses_knowledge", "engaging"}
    if needs_context:
        committee_context_template = build_committee_context(metric, safe_ctx)
        pair_reference = ctx_text
    else:
        committee_context_template = build_committee_context(metric, None)
        pair_reference = None

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

    # Build pairs (merged function handles full vs efficient via K)
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
python main_topicalchat.py \
  --metric_dir /train-data1/shaosen/USR-TopicalChat/dataFolder/topicalchat"/ctx_0000/engaging \
  --round_root /home/shaosen/LLM/Multi-agent/Saving/TopicalChat/ctx_0000
"""
