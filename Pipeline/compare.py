# Domain/compare.py

from __future__ import annotations
from typing import List, Dict, Any

from Domain.candidate import Candidate
from Domain.artifact import (
    PairArtifacts,
    ReasonList,
    QuestionList,
    CRITScores,
)
from Shared.workflow import run_agent_with_input_state_sync
from Agents.Committee import (
    AdvocateA_Init,
    AdvocateB_Init,
    ReasonExtractorA,
    ReasonExtractorB,
    make_validation_agent,
    make_qa_reason_extractor,  # ← use question-conditioned extractor for QA
    Moderator_A,
    Moderator_B,
    Responder_A,
    Responder_B,
)

# --- small helpers (paper-friendly, no heuristics) -----------------


def _reasons_from(obj: Any) -> List[str]:
    if not isinstance(obj, dict):
        return []
    r = obj.get("reasons", [])
    return [str(x).strip() for x in (r or [])]


def _has_valid_reasons(obj: Any) -> bool:
    r = _reasons_from(obj)
    return any(x and x.lower() != "none" for x in r)


def _default_claims(nameA: str, nameB: str) -> tuple[str, str]:
    return (
        f"For the same source article, {nameA} is more fluent and grammatically natural than {nameB}",
        f"For the same source article, {nameB} is more fluent and grammatically natural than {nameA}",
    )


def _get_question_text(qlist: QuestionList, idx: int) -> str:
    """Safe fetch of the idx-th question text from a QuestionList."""
    try:
        q = qlist.questions[idx]
        return getattr(q, "question", "") or ""
    except Exception:
        return ""


# --- main pipeline --------------------------------------------------


def run_compare_pipeline(
    candA: Candidate,
    candB: Candidate,
    *,
    committee_context: str,
    member_name: str,
    claimA: str | None,
    claimB: str | None,
) -> PairArtifacts:

    if claimA is None or claimB is None:
        claimA, claimB = _default_claims(candA.name, candB.name)

    base = {
        "committee_context": committee_context,
        "member_name": member_name,
        "candidateA": candA.name,
        "candidateB": candB.name,
        "docA": candA.document or "",
        "docB": candB.document or "",
        "claimA": claimA,
        "claimB": claimB,
    }

    artifacts = PairArtifacts()

    # ----------------- SIDE A: Advocate -> Reasons (position-locked) -> CRIT (if any)
    _, sA1 = run_agent_with_input_state_sync(AdvocateA_Init, dict(base))
    artifacts.Init_ArgumentA = sA1.get("ArgumentA", "")

    _, sA2 = run_agent_with_input_state_sync(ReasonExtractorA, {**base, **sA1})
    reasonsA = sA2.get("reason_list_A", {}) or {}
    artifacts.reason_list_init_A = ReasonList(reasons=_reasons_from(reasonsA))

    if _has_valid_reasons(reasonsA):
        evalA = make_validation_agent(
            "reason_list_A", "CRIT_scores_list_A", "claimA"
        )
        _, sA3 = run_agent_with_input_state_sync(evalA, {**base, **sA1, **sA2})
        artifacts.CRIT_scores_init_A = CRITScores(
            CRITScores=sA3.get("CRIT_scores_list_A", {}).get("CRITScores", []) or []
        )
    else:
        artifacts.CRIT_scores_init_A = CRITScores(CRITScores=[])

    # ----------------- SIDE B: Advocate -> Reasons (position-locked) -> CRIT (if any)
    _, sB1 = run_agent_with_input_state_sync(AdvocateB_Init, dict(base))
    artifacts.Init_ArgumentB = sB1.get("ArgumentB", "")

    _, sB2 = run_agent_with_input_state_sync(ReasonExtractorB, {**base, **sB1})
    reasonsB = sB2.get("reason_list_B", {}) or {}
    artifacts.reason_list_init_B = ReasonList(reasons=_reasons_from(reasonsB))

    if _has_valid_reasons(reasonsB):
        evalB = make_validation_agent(
            "reason_list_B", "CRIT_scores_list_B", "claimB"
        )
        _, sB3 = run_agent_with_input_state_sync(evalB, {**base, **sB1, **sB2})
        artifacts.CRIT_scores_init_B = CRITScores(
            CRITScores=sB3.get("CRIT_scores_list_B", {}).get("CRITScores", []) or []
        )
    else:
        artifacts.CRIT_scores_init_B = CRITScores(CRITScores=[])

    # ----------------- Moderator (only if that side has reasons)
    comb = {**base, **sA1, **sA2, **sB1, **sB2}

    if _has_valid_reasons(reasonsA):
        _, mA = run_agent_with_input_state_sync(Moderator_A, comb)
        artifacts.moderator_question_for_B = QuestionList(
            questions=mA.get("moderator_question_for_B", {}).get("questions", []) or []
        )
    else:
        artifacts.moderator_question_for_B = QuestionList(questions=[])

    if _has_valid_reasons(reasonsB):
        _, mB = run_agent_with_input_state_sync(Moderator_B, comb)
        artifacts.moderator_question_for_A = QuestionList(
            questions=mB.get("moderator_question_for_A", {}).get("questions", []) or []
        )
    else:
        artifacts.moderator_question_for_A = QuestionList(questions=[])

    # ----------------- Responders (only if there are questions)
    for q in artifacts.moderator_question_for_A.questions:
        question_text = getattr(q, "question", "") or ""
        if not question_text.strip():
            continue
        _, outA = run_agent_with_input_state_sync(
            Responder_A, {**base, "question": question_text}
        )
        artifacts.answer_list_A.append(outA.get("answerA", ""))

    for q in artifacts.moderator_question_for_B.questions:
        question_text = getattr(q, "question", "") or ""
        if not question_text.strip():
            continue
        _, outB = run_agent_with_input_state_sync(
            Responder_B, {**base, "question": question_text}
        )
        artifacts.answer_list_B.append(outB.get("answerB", ""))

    # ----------------- QA: question-conditioned reason extraction (stance-locked) + CRIT

    # A’s answers (pair each answer with its originating question)
    QA_Reason_A = make_qa_reason_extractor(
        output_key="qa_reason_list_A",
        answer_key="answer",
        claim_key="claimA",
        position_key="candidateA",
        question_key="question",
    )
    QA_Valid_A = make_validation_agent(
        reason_list_key="qa_reason_list_A",
        output_key="qa_CRITScores_A",
        claim_key="claimA"
    )

    for idx, ans in enumerate(artifacts.answer_list_A):
        if not (ans or "").strip():
            continue
        q_text = _get_question_text(artifacts.moderator_question_for_A, idx)
        _, rA = run_agent_with_input_state_sync(
            QA_Reason_A, {**base, "answer": ans, "question": q_text}
        )
        rlA = rA.get("qa_reason_list_A", {}) or {}
        artifacts.reason_list_QA_A.append(ReasonList(reasons=_reasons_from(rlA)))

        if _has_valid_reasons(rlA):
            _, cA = run_agent_with_input_state_sync(
                QA_Valid_A, {**base, "qa_reason_list_A": rlA}
            )
            artifacts.CRIT_scores_QA_A.append(
                CRITScores(
                    CRITScores=cA.get("qa_CRITScores_A", {}).get("CRITScores", []) or []
                )
            )
        else:
            artifacts.CRIT_scores_QA_A.append(CRITScores(CRITScores=[]))

    # B’s answers (pair each answer with its originating question)
    QA_Reason_B = make_qa_reason_extractor(
        output_key="qa_reason_list_B",
        answer_key="answer",
        claim_key="claimB",
        position_key="candidateB",
        question_key="question",
    )
    QA_Valid_B = make_validation_agent(
        reason_list_key="qa_reason_list_B",
        output_key="qa_CRITScores_B",
        claim_key="claimB"
    )

    for idx, ans in enumerate(artifacts.answer_list_B):
        if not (ans or "").strip():
            continue
        q_text = _get_question_text(artifacts.moderator_question_for_B, idx)
        _, rB = run_agent_with_input_state_sync(
            QA_Reason_B, {**base, "answer": ans, "question": q_text}
        )
        rlB = rB.get("qa_reason_list_B", {}) or {}
        artifacts.reason_list_QA_B.append(ReasonList(reasons=_reasons_from(rlB)))

        if _has_valid_reasons(rlB):
            _, cB = run_agent_with_input_state_sync(
                QA_Valid_B, {**base, "qa_reason_list_B": rlB}
            )
            artifacts.CRIT_scores_QA_B.append(
                CRITScores(
                    CRITScores=cB.get("qa_CRITScores_B", {}).get("CRITScores", []) or []
                )
            )
        else:
            artifacts.CRIT_scores_QA_B.append(CRITScores(CRITScores=[]))

    return artifacts
