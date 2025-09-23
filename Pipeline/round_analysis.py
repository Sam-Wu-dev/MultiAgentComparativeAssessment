# Pipeline/round_analysis.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from Domain.pair import Pair
from Domain.round import Round
from Domain.round_analysis import (
    RoundAnalysis,
    RankCorrelationResult,
    CandidateRankResult,
    MatchResult,
    CellScore,
)
from Pipeline.poe import (
    poe_gaussian_scores,
)
import math


# -----------------------------
# Spearman helpers
# -----------------------------


def _spearman_rho_with_p(
    gt_ranks: Dict[str, int], model_scores: Dict[str, float]
) -> Tuple[float, float, pd.DataFrame]:
    common = sorted(set(gt_ranks.keys()) & set(model_scores.keys()))
    if len(common) < 3:
        return (
            float("nan"),
            float("nan"),
            pd.DataFrame(columns=["name", "gt_rank", "model_score", "llm_rank"]),
        )

    df = pd.DataFrame(
        {
            "name": common,
            "gt_rank": [gt_ranks[n] for n in common],  # already ranks (1=best)
            "model_score": [model_scores[n] for n in common],  # higher=better
        }
    )

    # Rank model scores once with average-tie handling; higher score → smaller/better rank
    df["llm_rank"] = (-df["model_score"]).rank(method="average").astype(float)

    # Spearman = Pearson on (once-)ranked vectors; gt_rank is already a rank → don't rank again
    x = df["gt_rank"].astype(float).values
    y = df["llm_rank"].astype(float).values
    rho, p = spearmanr(x, y, nan_policy="omit")

    rho = float(rho) if np.isfinite(rho) else float("nan")
    p = float(p) if np.isfinite(p) else float("nan")
    return rho, p, df


def _rank_result_from_scores(
    gt_ranks: Dict[str, int],
    gt_scores: Dict[str, float],
    model_scores: Dict[str, float],
) -> RankCorrelationResult:
    rho, pval, df = _spearman_rho_with_p(gt_ranks, model_scores)

    rank_list: Dict[str, CandidateRankResult] = {}
    for _, r in df.iterrows():
        name = r["name"]
        rank_list[name] = CandidateRankResult(
            gt_rank=int(r["gt_rank"]),
            gt_score=float(gt_scores.get(name, np.nan)),
            llm_rank=int(round(float(r["llm_rank"]))),
            model_score=float(model_scores[name]),
        )

    return RankCorrelationResult(
        spearman_rho=rho,
        p_value=pval,
        rank_list=rank_list,
    )


# -----------------------------
# Table builder
# -----------------------------


def _extract_pair_scores(p: Pair, mode: str) -> Optional[Tuple[float, float]]:
    """Return (scoreA, scoreB) for a pair under the requested mode."""
    if not p.analysis:
        return None
    if mode == "initial":
        sA = getattr(p.analysis.reasonForA, "sumPersuasiveness", None)
        sB = getattr(p.analysis.reasonForB, "sumPersuasiveness", None)
    elif mode == "reply":
        sA = getattr(p.analysis.reasonForA, "sumAdvantage", None)
        sB = getattr(p.analysis.reasonForB, "sumAdvantage", None)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if sA is None or sB is None:
        return None
    return float(sA), float(sB)


def _build_match_table_and_bias(
    round_obj: Round, pairs: List[Pair], mode: str
) -> Tuple[List[List[Optional[CellScore]]], float]:
    """
    Build the full N×N table with each cell storing CellScore(scoreA, scoreB, gap).
    Also compute positionalBias = sum(scoreA) - sum(scoreB) across all filled cells.
    """
    names = [c.name for c in round_obj.candidates]
    idx = {n: i for i, n in enumerate(names)}
    N = len(names)
    table: List[List[Optional[CellScore]]] = [[None] * N for _ in range(N)]

    sum_scoreA = 0.0
    sum_scoreB = 0.0

    for p in pairs:
        s = _extract_pair_scores(p, mode)
        if s is None:
            continue
        sA, sB = s
        i = idx.get(p.nameA)
        j = idx.get(p.nameB)
        if i is None or j is None or i == j:
            continue
        cell = CellScore(scoreA=sA, scoreB=sB, gap=sA - sB)
        table[i][j] = cell
        sum_scoreA += sA
        sum_scoreB += sB

    positional_bias = sum_scoreA - sum_scoreB
    return table, positional_bias


def position_sums_from_table(
    table: List[List[Optional[CellScore]]],
    names: List[str],
):
    """
    Returns:
      - sum_all: Σ over all matches a candidate appears in (gap aggregation)
      - counts:  {"as_A": count_as_A, "as_B": count_as_B, "all": ...}
    (We keep only sum_all for downstream Spearman.)
    """
    sum_as_A = {n: 0.0 for n in names}
    sum_as_B = {n: 0.0 for n in names}
    cnt_as_A = {n: 0 for n in names}
    cnt_as_B = {n: 0 for n in names}
    N = len(names)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            cell = table[i][j]
            if cell is None:
                continue
            gap = float(cell.gap)
            # Row candidate is A
            sum_as_A[names[i]] += gap
            cnt_as_A[names[i]] += 1
            # Column candidate is B
            sum_as_B[names[j]] -= gap
            cnt_as_B[names[j]] += 1

    sum_all = {n: sum_as_A[n] + sum_as_B[n] for n in names}
    counts = {
        n: {"as_A": cnt_as_A[n], "as_B": cnt_as_B[n], "all": cnt_as_A[n] + cnt_as_B[n]}
        for n in names
    }
    return sum_all, counts


# -----------------------------
# LSE from table (NEW)
# -----------------------------
def _lse_scores_from_table(
    table: List[List[Optional[CellScore]]],
    ridge: float = 1e-9,
) -> List[float]:
    """
    Build X, y from pairwise gaps (C_ij = S_i - S_j), anchor S_0 = 0, and solve LS.
    Returns s_hat as a list of length N.
    """
    N = len(table)
    # collect observed (i,j,gap)
    obs: List[Tuple[int, int, float]] = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            cell = table[i][j]
            if cell is None:
                continue
            obs.append((i, j, float(cell.gap)))

    if not obs:
        return [0.0] * N

    K = len(obs)
    # X = [e0^T; W], y = [0; c], where row k of W is e_i - e_j
    X = np.zeros((K + 1, N), dtype=float)
    X[0, 0] = 1.0  # anchor S_0 = 0
    y = np.zeros((K + 1,), dtype=float)

    for k, (i, j, c) in enumerate(obs, start=1):
        X[k, i] = 1.0
        X[k, j] = -1.0
        y[k] = c

    A = X.T @ X
    if ridge and ridge > 0:
        A = A + ridge * np.eye(N)

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)

    s_hat = A_inv @ (X.T @ y)
    return s_hat.tolist()


# -----------------------------
# PoE from table (built-in)
# -----------------------------


def _poe_scores_from_table(table: List[List[Optional[CellScore]]]) -> List[float]:
    comps: List[Tuple[int, int, float]] = []
    # Map matrix indices to contiguous [0..N-1]
    N = len(table)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            cell = table[i][j]
            if cell is None:
                continue
            # logistic P(i>j)
            p_ij = 1.0 / (1.0 + math.exp(cell.scoreB - cell.scoreA))
            comps.append((i, j, float(p_ij)))
    if not comps:
        return [0.0] * N
    s_hat = poe_gaussian_scores(N, comps)
    return s_hat.tolist()


# -----------------------------
# Public entry
# -----------------------------


def compute_round_analysis(round_obj: Round, pairs: List[Pair]) -> RoundAnalysis:
    """
    Build RoundAnalysis for both modes:
      • InitialArgumentOnly (sumPersuasiveness)
      • ArgumentWithReply   (sumAdvantage)

    For each mode we now produce:
      - full match_table with per-cell (scoreA, scoreB, gap)
      - sum_all Spearman vs GT (dense ranks from total per-candidate gap)
      - poe_gaussian Spearman vs GT (scores from PoE over pairwise probabilities)
      - lse Spearman vs GT (scores from LS over pairwise gaps)          <── NEW
      - positionalBias = Σ(scoreA) - Σ(scoreB) across all filled cells
    """
    gt_ranks: Dict[str, int] = {
        c.name: c.gt_rank for c in round_obj.candidates if c.gt_rank is not None
    }
    gt_scores: Dict[str, float] = {
        c.name: c.gt_score for c in round_obj.candidates if c.gt_score is not None
    }
    names = [c.name for c in round_obj.candidates]

    def _one_mode(mode: str) -> MatchResult:
        table, pos_bias = _build_match_table_and_bias(round_obj, pairs, mode)

        # sum_all
        sumAll, _counts = position_sums_from_table(table, names)
        rc_sumAll = _rank_result_from_scores(gt_ranks, gt_scores, sumAll)

        # PoE aggregator (existing)
        poe_scores = _poe_scores_from_table(table)
        poe_dict = {n: float(poe_scores[i]) for i, n in enumerate(names)}
        rc_poe = _rank_result_from_scores(gt_ranks, gt_scores, poe_dict)

        # LSE aggregator (NEW)
        lse_scores = _lse_scores_from_table(table)
        lse_dict = {n: float(lse_scores[i]) for i, n in enumerate(names)}
        rc_lse = _rank_result_from_scores(gt_ranks, gt_scores, lse_dict)

        return MatchResult(
            sum_all=rc_sumAll,
            poe_gaussian=rc_poe,
            lse=rc_lse,  # <── NEW
            positionalBias=pos_bias,
            match_table=table,
        )

    init_res = _one_mode("initial")
    reply_res = _one_mode("reply")

    return RoundAnalysis(
        InitialArgumentOnly=init_res,
        ArgumentWithReply=reply_res,
    )
