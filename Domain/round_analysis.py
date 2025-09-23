# pairwise_multi_agent_ranker/Domain/round_analysis.py

from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CandidateRankResult(BaseModel):
    gt_rank: int
    gt_score: float
    llm_rank: int
    model_score: float


class RankCorrelationResult(BaseModel):
    spearman_rho: float
    p_value: float
    rank_list: Dict[str, CandidateRankResult] = Field(default_factory=dict)


class CellScore(BaseModel):
    """One directional match (row i = A, col j = B)."""

    scoreA: float
    scoreB: float
    gap: float  # scoreA - scoreB


class MatchResult(BaseModel):
    """Full-table order analysis + aggregators."""

    sum_all: RankCorrelationResult
    poe_gaussian: Optional[RankCorrelationResult] = None
    lse: Optional[RankCorrelationResult] = None  # <── NEW
    positionalBias: Optional[float] = (
        None  # sum(scoreA) - sum(scoreB) over all filled cells
    )
    match_table: List[List[Optional[CellScore]]] = Field(default_factory=list)


class RoundAnalysis(BaseModel):
    InitialArgumentOnly: Optional[MatchResult] = None
    ArgumentWithReply: Optional[MatchResult] = None
