# Domain/artifact.py

from typing import List, Optional
from pydantic import BaseModel


class ReasonList(BaseModel):
    reasons: List[str]


class Question(BaseModel):
    original_reason: str
    question: str


class QuestionList(BaseModel):
    questions: List[Question]


class Score(BaseModel):
    score: float
    explanation: str


class CRITScore(BaseModel):
    validity: Score
    reliability: Score


class CRITScores(BaseModel):
    CRITScores: List[CRITScore]


def _is_none_reasons(reasons: List[str]) -> bool:
    if not reasons:
        return False
    return all((r or "").strip().lower() == "none" for r in reasons)


def _reasons_crits_consistent(reasons: List[str], crits: List[CRITScore]) -> bool:
    if len(reasons) == len(crits):
        return True
    if _is_none_reasons(reasons) and len(crits) == 0:
        return True
    return False


class PairArtifacts(BaseModel):
    Init_ArgumentA: Optional[str] = None
    Init_ArgumentB: Optional[str] = None

    reason_list_init_A: ReasonList = ReasonList(reasons=[])
    reason_list_init_B: ReasonList = ReasonList(reasons=[])

    CRIT_scores_init_A: CRITScores = CRITScores(CRITScores=[])
    CRIT_scores_init_B: CRITScores = CRITScores(CRITScores=[])

    moderator_question_for_B: QuestionList = QuestionList(questions=[])  # A→B
    moderator_question_for_A: QuestionList = QuestionList(questions=[])  # B→A

    answer_list_A: List[str] = []
    answer_list_B: List[str] = []

    reason_list_QA_A: List[ReasonList] = []
    reason_list_QA_B: List[ReasonList] = []
    CRIT_scores_QA_A: List[CRITScores] = []
    CRIT_scores_QA_B: List[CRITScores] = []

    def is_complete(self) -> bool:
        ok = True

        # 1) Arguments present
        if not (self.Init_ArgumentA and self.Init_ArgumentA.strip()):
            print("❌ Missing Init_ArgumentA")
            ok = False
        if not (self.Init_ArgumentB and self.Init_ArgumentB.strip()):
            print("❌ Missing Init_ArgumentB")
            ok = False

        # 2) Initial reasons ↔ CRIT consistency
        if not _reasons_crits_consistent(
            self.reason_list_init_A.reasons, self.CRIT_scores_init_A.CRITScores
        ):
            print(
                f"❌ A: {len(self.reason_list_init_A.reasons)} reasons "
                f"vs {len(self.CRIT_scores_init_A.CRITScores)} CRIT scores"
            )
            ok = False
        if not _reasons_crits_consistent(
            self.reason_list_init_B.reasons, self.CRIT_scores_init_B.CRITScores
        ):
            print(
                f"❌ B: {len(self.reason_list_init_B.reasons)} reasons "
                f"vs {len(self.CRIT_scores_init_B.CRITScores)} CRIT scores"
            )
            ok = False

        # 3) Question/answer counts
        if len(self.answer_list_A) != len(self.moderator_question_for_A.questions):
            print(
                f"❌ A: {len(self.moderator_question_for_A.questions)} questions "
                f"vs {len(self.answer_list_A)} answers"
            )
            ok = False
        if len(self.answer_list_B) != len(self.moderator_question_for_B.questions):
            print(
                f"❌ B: {len(self.moderator_question_for_B.questions)} questions "
                f"vs {len(self.answer_list_B)} answers"
            )
            ok = False

        # 4) QA lengths per side
        if not (
            len(self.reason_list_QA_A)
            == len(self.answer_list_A)
            == len(self.CRIT_scores_QA_A)
        ):
            print(
                "❌ A QA misaligned: "
                f"reasons={len(self.reason_list_QA_A)}, "
                f"answers={len(self.answer_list_A)}, "
                f"CRIT={len(self.CRIT_scores_QA_A)}"
            )
            ok = False
        if not (
            len(self.reason_list_QA_B)
            == len(self.answer_list_B)
            == len(self.CRIT_scores_QA_B)
        ):
            print(
                "❌ B QA misaligned: "
                f"reasons={len(self.reason_list_QA_B)}, "
                f"answers={len(self.answer_list_B)}, "
                f"CRIT={len(self.CRIT_scores_QA_B)}"
            )
            ok = False

        # 5) QA internal consistency
        for idx, (rl, cs) in enumerate(
            zip(self.reason_list_QA_A, self.CRIT_scores_QA_A)
        ):
            if not _reasons_crits_consistent(rl.reasons, cs.CRITScores):
                print(
                    f"❌ A.QA[{idx}] mismatch: {len(rl.reasons)} reasons vs {len(cs.CRITScores)} CRIT"
                )
                ok = False
        for idx, (rl, cs) in enumerate(
            zip(self.reason_list_QA_B, self.CRIT_scores_QA_B)
        ):
            if not _reasons_crits_consistent(rl.reasons, cs.CRITScores):
                print(
                    f"❌ B.QA[{idx}] mismatch: {len(rl.reasons)} reasons vs {len(cs.CRITScores)} CRIT"
                )
                ok = False

        if ok:
            print("✅ PairArtifacts is complete")
        return ok
