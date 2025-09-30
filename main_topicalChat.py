# main_topicalChat.py  (argparse version, with definitions + fact-before-dialogue + speaker labels)
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

from Domain.candidate import Candidate
from Domain.round import Round
from Pipeline.round_analysis import compute_round_analysis
from Pipeline.make_pairs import ensure_pairs

# -------------------------
# Global committee role
# -------------------------
MEMBER_NAME: str = "member"

# -------------------------
# Metric-specific settings (TopicalChat; aligned with G-Eval/UniEval)
# -------------------------
METRIC_CLAIMS = {
    "naturalness":  "Response {A} is more natural than response {B}",
    "coherence":    "Given the same dialogue context, response {A} is more coherent than response {B}",
    "engagingness": "Given the same dialogue context and facts, response {A} is more engaging than response {B}",
    "groundedness": "Given the same facts, response {A} is more grounded in knowledge than response {B}",
}

METRIC_TITLES = {
    "naturalness":  "TopicalChat Naturalness Committee",
    "coherence":    "TopicalChat Coherence Committee",
    "engagingness": "TopicalChat Engagingness Committee",
    "groundedness": "TopicalChat Groundedness Committee",
}

# Legacy directory names → canonical metric keys
LEGACY_TO_CANON = {
    "natural": "naturalness",
    "maintains_context": "coherence",
    "engaging": "engagingness",
    "uses_knowledge": "groundedness",
}

# -------------------------
# Metric Definitions (concise; injected into committee_context)
# -------------------------
METRIC_DEFINITIONS = {
    "naturalness": (
        "Judge NATURALNESS purely by linguistic human-likeness:\n"
        "- Fluent, grammatical, idiomatic, appropriate tone.\n"
        "- Do NOT judge factuality or context usage."
    ),
    "coherence": (
        "Judge COHERENCE by fit to dialogue history:\n"
        "- Stays on topic; references prior turns correctly; no contradictions."
    ),
    "engagingness": (
        "Judge ENGAGINGNESS by how much the reply invites further conversation:\n"
        "- Specific, interesting, responsive; good follow-ups when appropriate.\n"
        "- Facts may enhance engagement if relevant."
    ),
    "groundedness": (
        "Judge GROUNDEDNESS by support from provided facts:\n"
        "- Claims are anchored in the facts; avoid hallucinations.\n"
        "- Style is irrelevant except where it affects evidential correctness."
    ),
}

# -------------------------
# Helpers
# -------------------------
def escape_braces(s: str) -> str:
    """Make curly braces literal for f-strings / prompt templates."""
    return s.replace("{", "{{").replace("}", "}}")

def label_speakers(dialogue_block: str,
                   a_tag: str = "Speaker A",
                   b_tag: str = "Speaker B") -> str:
    """
    Alternate-label each non-empty line in a dialogue block as Speaker A/B.
    """
    if not dialogue_block:
        return ""
    lines = [ln.strip() for ln in dialogue_block.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln]
    out = []
    a_turn = True
    for ln in lines:
        tag = a_tag if a_turn else b_tag
        out.append(f"{tag}: {ln}")
        a_turn = not a_turn
    return "\n".join(out)

def parse_context_file(ctx_path: Path) -> Tuple[str, Optional[str]]:
    """
    Parse a context.txt of the form:

    Context:
    <multi-line dialogue history>

    Fact:
    <multi-line fact or '_nofact'>

    Returns (dialogue_history, fact_or_None).
    """
    if not ctx_path.exists():
        return ("", None)

    raw = ctx_path.read_text(encoding="utf-8")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    ctx_anchor = "Context:"
    fact_anchor = "Fact:"

    ctx_start = text.find(ctx_anchor)
    fact_start = text.find(fact_anchor)

    dialogue = ""
    fact = None

    if ctx_start != -1 and fact_start != -1:
        dialogue = text[ctx_start + len(ctx_anchor):fact_start].strip()
        fact_block = text[fact_start + len(fact_anchor):].strip()
        fact = None if fact_block.lower().startswith("_nofact") else fact_block
    elif ctx_start != -1:
        dialogue = text[ctx_start + len(ctx_anchor):].strip()
    elif fact_start != -1:
        fact_block = text[fact_start + len(fact_anchor):].strip()
        fact = None if fact_block.lower().startswith("_nofact") else fact_block

    return (dialogue, fact)

def build_committee_context(metric: str,
                            fact_block: Optional[str],
                            dialogue_block: Optional[str]) -> str:
    """
    Build committee prompt with definition + references.
    Order of references: Facts first, then Dialogue (if present).
    """
    title = METRIC_TITLES[metric]
    base = f"""
    We are in the {title}.
    We must decide which of two candidate responses better satisfies the metric: {metric}.
    We are deciding between {{A}} and {{B}}.
    Only evaluate the candidates on the given metric.
    """.strip()

    definition = METRIC_DEFINITIONS[metric]
    parts = [base, "Metric definition:\n" + escape_braces(definition)]

    # References: FACTS first, then DIALOGUE
    if fact_block:
        parts.append("Knowledge facts (reference):\n" + escape_braces(fact_block))
    if dialogue_block:
        parts.append("Conversation context (reference):\n" + escape_braces(dialogue_block))

    return "\n\n".join(parts)

def compose_pair_reference(metric: str,
                           dialogue_labeled: Optional[str],
                           fact: Optional[str]) -> Optional[str]:
    """
    UniEval-style inputs (with labeled dialogue), with FACT before DIALOGUE when both are present.
      - naturalness: None
      - coherence: dialogue
      - engagingness: fact (+ dialogue if provided)
      - groundedness: fact
    """
    if metric == "naturalness":
        return None

    if metric == "coherence":
        return ("Conversation context (reference):\n" + escape_braces(dialogue_labeled)) if dialogue_labeled else ""

    if metric == "engagingness":
        chunks = []
        if fact:
            chunks.append("Knowledge facts (reference):\n" + escape_braces(fact))
        if dialogue_labeled:
            chunks.append("Conversation context (reference):\n" + escape_braces(dialogue_labeled))
        return "\n\n".join(chunks) if chunks else ""

    if metric == "groundedness":
        return ("Knowledge facts (reference):\n" + escape_braces(fact)) if fact else ""

    return None

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

# -------------------------
# Main (argparse)
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pairwise comparisons on a single TopicalChat context/metric folder (UniEval-aligned)."
    )
    parser.add_argument(
        "--metric_dir",
        type=str,
        required=True,
        help="Path to one metric folder (e.g., topicalchat/<ctx_id>/coherence or legacy uses_knowledge).",
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

    # metric = last folder name; map legacy -> canonical
    raw_metric = metric_dir.name.lower()
    metric = LEGACY_TO_CANON.get(raw_metric, raw_metric)

    if metric not in METRIC_CLAIMS:
        valid = ", ".join(METRIC_CLAIMS.keys())
        legacy = ", ".join(sorted(LEGACY_TO_CANON.keys()))
        raise SystemExit(
            f"Unknown metric folder: {metric}. Use one of canonical [{valid}] or legacy [{legacy}]."
        )

    # load candidates (MX.json files)
    candidates: List[Candidate] = load_candidates_from_dir(metric_dir)
    if not candidates:
        raise SystemExit(f"No candidate JSONs found in: {metric_dir}")

    # load conversation context from parent folder (context.txt)
    ctx_path = metric_dir.parent / "context.txt"
    dialogue_raw, fact_raw = parse_context_file(ctx_path)
    dialogue_labeled = label_speakers(dialogue_raw) if dialogue_raw else ""

    # Compose pair reference (FACT first where both exist)
    pair_reference = compose_pair_reference(
        metric,
        dialogue_labeled=dialogue_labeled if dialogue_labeled else None,
        fact=fact_raw if fact_raw else None,
    )

    # Build committee context (with metric definition; FACT before DIALOGUE)
    committee_context_template = build_committee_context(
        metric,
        fact_block=fact_raw if fact_raw else None,
        dialogue_block=dialogue_labeled if dialogue_labeled else None,
    )

    metric_round_dir = round_root / metric
    metric_round_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Creating round for metric={metric} (from '{raw_metric}') at: {metric_round_dir}")

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
python main_topicalChat.py \
  --metric_dir /train-data1/shaosen/USR-TopicalChat/dataFolder/topicalchat/ctx_0000/engaging \
  --round_root /home/shaosen/LLM/Multi-agent/Saving/TopicalChat/ctx_0000
"""
