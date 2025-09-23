# Domain/pair_analysis.py
from __future__ import annotations
from typing import List
from pydantic import BaseModel
from Domain.artifact import PairArtifacts, CRITScore


class ReasonAnalysis(BaseModel):
    reasonIndex: int
    persuasiveness: float
    counterReasonsPersuasiveness: List[float]
    counterCRITScore: float
    advantage: float


class PairAnalysisBlock(BaseModel):
    reasons: List[ReasonAnalysis]
    sumPersuasiveness: float
    sumAdvantage: float


class PairAnalysis(BaseModel):
    reasonForA: PairAnalysisBlock
    reasonForB: PairAnalysisBlock


def build_analysis(artifacts: PairArtifacts) -> PairAnalysis:
    """
    Compute PairAnalysis directly from PairArtifacts
    without converting to dict.
    """

    def _make_block(
        initial_crits: List[CRITScore], counter_crits_list: List[List[CRITScore]]
    ) -> PairAnalysisBlock:
        reason_res: List[ReasonAnalysis] = []
        adv_sum = 0.0
        pers_sum = 0.0

        for idx, crit in enumerate(initial_crits):
            v = float(crit.validity.score)
            rlt = float(crit.reliability.score)
            pers = round(v * rlt, 4)
            pers_sum += pers

            cnt_scores: List[float] = []
            if idx < len(counter_crits_list):
                for cr in counter_crits_list[idx]:
                    cnt_scores.append(
                        float(cr.validity.score) * float(cr.reliability.score)
                    )

            avg_cnt = round(sum(cnt_scores) / len(cnt_scores), 4) if cnt_scores else 0.0
            adv = round(pers - avg_cnt, 4)
            adv_sum += adv

            reason_res.append(
                ReasonAnalysis(
                    reasonIndex=idx,
                    persuasiveness=pers,
                    counterReasonsPersuasiveness=cnt_scores,
                    counterCRITScore=avg_cnt,
                    advantage=adv,
                )
            )

        return PairAnalysisBlock(
            reasons=reason_res,
            sumPersuasiveness=round(pers_sum, 4),
            sumAdvantage=round(adv_sum, 4),
        )

    return PairAnalysis(
        reasonForA=_make_block(
            artifacts.CRIT_scores_init_A.CRITScores,
            [crit.CRITScores for crit in artifacts.CRIT_scores_QA_A],
        ),
        reasonForB=_make_block(
            artifacts.CRIT_scores_init_B.CRITScores,
            [crit.CRITScores for crit in artifacts.CRIT_scores_QA_B],
        ),
    )
