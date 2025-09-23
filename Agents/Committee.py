# pairwise_multi_agent_ranker/Agents/Committee.py

from google.adk.agents import LlmAgent
from typing import List
from pydantic import BaseModel
from Shared import MODEL20
from Shared.prompt import JSONFORMAT, LANGUAGE, NAME_REFERENCE


# ── Output Schemas (unchanged) ──
class ReasonList(BaseModel):
    reasons: list[str]


class Question(BaseModel):
    original_reason: str
    question: str


class QuestionList(BaseModel):
    questions: list[Question]


class Score(BaseModel):
    score: float
    explanation: str


class CRITScore(BaseModel):
    validity: Score
    reliability: Score


class CRITScores(BaseModel):
    CRITScores: List[CRITScore]


# ── 1) Initial Advocates ─────────────────────────────────────────────
def make_initial_advocate(output_key: str, claim_key: str, position_key: str):
    return LlmAgent(
        name="Advocate",  # neutral name for both sides
        model=MODEL20,
        instruction=f"""
        {{committee_context}}

        Your role: You are a {{member_name}} in the committee.
        Your task: You must argue for {{{position_key}}} on the claim: "{{{claim_key}}}".

        Candidates:
        • {{candidateA}}: {{docA}}
        • {{candidateB}}: {{docB}}

        Give the committee your opinion on why "{{{claim_key}}}".

        {LANGUAGE}
        {NAME_REFERENCE}
        """,
        output_key=output_key,
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


AdvocateA_Init = make_initial_advocate(
    output_key="ArgumentA", claim_key="claimA", position_key="candidateA"
)
AdvocateB_Init = make_initial_advocate(
    output_key="ArgumentB", claim_key="claimB", position_key="candidateB"
)


# ── 2) Argument → Reason List (position-locked) ─────────────────────
def make_reason_extractor(
    output_key: str, argument_key: str, claim_key: str, position_key: str
):
    return LlmAgent(
        name="ReasonExtractor",
        model=MODEL20,
        instruction=f"""
        {{committee_context}}

        Your role: You are an assistant to the committee.
        Your task: From the member's argument, extract the reasons that:
        1. stand in the position of {{{position_key}}}, and
        2. support the claim: "{{{claim_key}}}"
        
        If there is no such reason, output exactly:
        {{"reasons": ["None"]}}

        Extraction Guidelines:
        - Distinctness: Each reason must present a unique idea, independent from the others, with no duplication or overlap.
        - Self-containment: Each reason must be fully understandable on its own, without relying on references or prior context.
        - Evidence-bound: Each reason must explicitly include concrete evidence grounded in the provided materials.
        
        Argument text to analyze:
        {{{argument_key}}}

        Output strictly as JSON:
        {{
        "reasons": [
            "<reason1>",
            "<reason2>"
        ]
        }}

        {JSONFORMAT}
        {LANGUAGE}
        {NAME_REFERENCE}
        """,
        output_key=output_key,
        output_schema=ReasonList,
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


ReasonExtractorA = make_reason_extractor(
    output_key="reason_list_A",
    argument_key="ArgumentA",
    claim_key="claimA",
    position_key="candidateA",
)
ReasonExtractorB = make_reason_extractor(
    output_key="reason_list_B",
    argument_key="ArgumentB",
    claim_key="claimB",
    position_key="candidateB",
)


def make_validation_agent(
    reason_list_key: str, output_key: str, claim_key: str, position_key: str
):
    return LlmAgent(
        name="Evaluator",
        model=MODEL20,
        instruction=f"""
        {{committee_context}}

        Your role: You are an external reviewer to the committee.
        Your task: Evaluate the reliability and validity of reasons provided by a committee member.

        Candidates:
        • {{candidateA}}: {{docA}}
        • {{candidateB}}: {{docB}}

        Reasons to evaluate (JSON):
        {{{reason_list_key}}}

        For each reason:
        1) Validity — Logical coherence supporting the claim: "{{{claim_key}}}".  
        2) Reliability — Grounding in the provided documents.

        Scores must be floats in [0, 1], rounded to two decimals, with concise explanations.
        Each reason must have exactly one (validity, reliability) pair.

        JSON output format:
        {{
          "CRITScores": [
            {{
              "validity":   {{ "score": <float>, "explanation": "..." }},
              "reliability":{{ "score": <float>, "explanation": "..." }}
            }},
            ...
          ]
        }}

        {JSONFORMAT}
        {LANGUAGE}
        """,
        output_key=output_key,
        output_schema=CRITScores,
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


Evaluator_A = make_validation_agent(
    reason_list_key="reason_list_A",
    output_key="CRIT_scores_list_A",
    claim_key="claimA",
    position_key="candidateA",
)
Evaluator_B = make_validation_agent(
    reason_list_key="reason_list_B",
    output_key="CRIT_scores_list_B",
    claim_key="claimB",
    position_key="candidateB",
)


# ── 4) Comparative Q&A (Moderator) ─────────────────────────────────
def make_moderator(output_key: str, claim_key: str, reason_list_key: str):
    return LlmAgent(
        name="Moderator",
        model=MODEL20,
        instruction=f"""
        {{committee_context}}

        Your role: You are the moderator for the committee.
        Your task: Elicit diverse, decision-useful opinions by asking **one** concise, comparative, evidence-seeking question per reason.
        
        Allowed evidence (HARD RULES):
        - Use only the provided materials: committee context and the two candidates’ materials.
        - Do NOT request or assume external data.
        
        Candidates:
        • {{candidateA}}: {{docA}}
        • {{candidateB}}: {{docB}}
        
        A member claims that "{{{claim_key}}}" and has provided reasons:
        {{{reason_list_key}}}
        
        QUESTION TYPE SELECTION — pick **one** type per reason:
        1) Comparative question  
        WHEN: The reason already compares A vs B on a specific point.
        ASK: Prompt the other member to compare the two candidates on that same point, providing evidence.

        2) Refute question  
        WHEN: The reason explicitly attacks the other candidate.  
        ASK: Invite the other member to defend the other candidate on that point, citing evidence.
        
        3) Balancing question 
        WHEN: The reason presents evidence that one candidate demonstrates a specific strength or impact.
        ASK: Reference that evidence and ask the other member whether the other candidate also demonstrates comparable evidence of the same strength or impact.
        
        Critical constraints (self-contained):
        - The other members will only see the question (without the original reason), so your question must be self-contained.
        
        Output JSON only:
        {{
          "questions": [
            {{
              "original_reason": "<copy the reason text verbatim>",
              "question": "<your question>"
            }},
            ...
          ]
        }}

        {JSONFORMAT}
        {LANGUAGE}
        {NAME_REFERENCE}
        """,
        output_key=output_key,
        output_schema=QuestionList,
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


Moderator_A = make_moderator(
    output_key="moderator_question_for_B",
    claim_key="claimA",
    reason_list_key="reason_list_A",
)
Moderator_B = make_moderator(
    output_key="moderator_question_for_A",
    claim_key="claimB",
    reason_list_key="reason_list_B",
)


# ── 5) Responders ──────────────────────────────────────────────────
def make_responder(
    name: str,
    claim_key: str,
    question_key: str,
    answer_key: str,
    position_key: str,
):
    return LlmAgent(
        name=name,
        model=MODEL20,
        instruction=f"""
        {{committee_context}}

        Your role: You are a {{member_name}} in the committee.
        Your task: Reply to the moderator’s question while supporting {{{position_key}}} on the claim: "{{{claim_key}}}".

        Candidate documents:
        • {{candidateA}}: {{docA}}
        • {{candidateB}}: {{docB}}

        Moderator's question:
        {{{question_key}}}

        Now, reply in support of {{{position_key}}} while answering the question.
        {LANGUAGE}
        {NAME_REFERENCE}
        """,
        output_key=answer_key,
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


# Wire (A answers for A; B answers for B)
Responder_A = make_responder(
    name="Responder",
    claim_key="claimA",
    question_key="question",
    answer_key="answerA",
    position_key="candidateA",
)
Responder_B = make_responder(
    name="Responder",
    claim_key="claimB",
    question_key="question",
    answer_key="answerB",
    position_key="candidateB",
)


def make_qa_reason_extractor(
    output_key: str,
    answer_key: str,  # e.g., "answer"
    claim_key: str,  # e.g., "claimA"
    position_key: str,  # "candidateA" or "candidateB"
    question_key: str,  # e.g., "question"
):
    return LlmAgent(
        name="QAReasonExtractor",
        model=MODEL20,
        instruction=f"""
        {{committee_context}}

        Your role: You are an assistant to the committee.
        Your task: From the member's argument, extract the reasons that:
        1. directly answering the moderator's question, and
        2. stand in the position of {{{position_key}}}, and
        3. support the claim: "{{{claim_key}}}"

        If there is no such reason, output exactly:
        {{"reasons": ["None"]}}

        Extraction Guidelines:
        - Distinctness: Each reason must present a unique idea, independent from the others, with no duplication or overlap.
        - Self-containment: Each reason must be fully understandable on its own, without relying on references or prior context.
        - Evidence-bound: Each reason must explicitly include concrete evidence grounded in the provided materials.

        Moderator's question:
        {{{question_key}}}

        Answer to analyze:
        {{{answer_key}}}

        Output strictly as JSON:
        {{
        "reasons": [
            "<reason1>",
            "<reason2>"
        ]
        }}

        {JSONFORMAT}
        {LANGUAGE}
        {NAME_REFERENCE}
        """,
        output_key=output_key,
        output_schema=ReasonList,
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )
