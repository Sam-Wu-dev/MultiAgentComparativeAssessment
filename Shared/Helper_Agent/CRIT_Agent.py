from typing import List, Callable, Optional
from pydantic import BaseModel
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from Shared import MODEL
from Shared.prompt import *
from Shared.callback import *
from google.genai import types
from Shared.Tool.load_candidate_md import load_single_candidate_tool


# === Schema for Reason Generation ===
class ClaimList(BaseModel):
    claims: List[str]


# === Schema for Score Validation ===
class Score(BaseModel):
    score: float
    reason: str


class Validation(BaseModel):
    reason: str
    validity: Score
    reliability: Score


class Result(BaseModel):
    validations: List[Validation]


# === Callback for CRIT Score ===
def compute_crit_score_callback(
    result_key: str = "result", output_key: str = "crit_score"
) -> Callable[[CallbackContext], Optional[types.Content]]:
    def callback(callback_context: CallbackContext) -> Optional[types.Content]:
        try:
            validations = callback_context.state[result_key]["validations"]
            if not validations:
                return None
            total = sum(
                v["validity"]["score"] * v["reliability"]["score"] for v in validations
            )
            score = total / len(validations)
            callback_context.state[output_key] = score
            print(f"[Callback] Computed CRIT score: {score:.2f}")
        except Exception as e:
            print(f"[Callback] Error computing CRIT score: {e}")
        return None

    return callback


QA_Claim_Agent = LlmAgent(
    name="QA_Claim_Agent",
    model=MODEL,
    instruction=f"""
    
    We have these candidates:
    • candidateA: {{candidateA}}
    • candidateB: {{candidateB}}

    Here is the review document for candidateA:
    {{reviewA}}

    Here is the review document for candidateB:
    {{reviewB}}
    
    Here is the question asked by Chair:
    {{question}}
    
    Here is the reply from {{professor}}:
    {{reply}}

    Your task is to extract all the claims made by {{professor}}:

    Respond in this format:
    {{
        "claims": [
            "<claim 1>",
            "<claim 2>",
            ...
        ]
    }}

    {JSONFORMAT}
    {LANGUAGE}
    """,
    output_key="claims",
    output_schema=ClaimList,
    include_contents="none",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

QA_Validation_Agent = LlmAgent(
    name="QA_Validation_Agent",
    model=MODEL,
    instruction=f"""
    
    
    We have these candidates:
    • candidateA: {{candidateA}}
    • candidateB: {{candidateB}}

    Here is the review document for candidateA:
    {{reviewA}}

    Here is the review document for candidateB:
    {{reviewB}}
    
    Here is the question asked by Chair:
    {{question}}
    
    {{professor}} claims that:
    {{claims}}

    For each claim, evaluate:
    1. **Validity**: Logical coherence and sound reasoning answering the question
    2. **Reliability**: Grounding in actual candidate information

    Give each a float score between 0 and 1, rounded to two decimal places, and an explanation.

    Respond in this format:
    {{
        "result": {{
            "validations": [
                {{
                    "reason" : "<copied reason>",
                    "validity": {{ "score": <float>, "reason": "..." }},
                    "reliability": {{ "score": <float>, "reason": "..." }}
                }}
            ]
        }}
    }}

    {JSONFORMAT}
    {LANGUAGE}
    """,
    output_key="claims",
    output_schema=ClaimList,
    include_contents="none",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# === Agent 2: Validate Each Reason ===
ValidationAgent = LlmAgent(
    name="ValidationAgent",
    model=MODEL,
    instruction=f"""
    {DEBATE_CONTEXT}

    You are a member of the admissions committee evaluating argument quality.

    And here are the review information of candidates {{candidate_names}}:
    candidateA: {{candidatesA}}
    candidateB: {{candidatesB}}
    
    You are given:
    - A claim: {{claim}}
    - A list of reasons: {{reasons}}

    For each reason, evaluate:
    1. **Validity**: Logical coherence and sound reasoning supporting the claim
    2. **Reliability**: Grounding in actual candidate information

    Give each a float score between 0 and 1, rounded to two decimal places, and an explanation.

    Respond in this format:
    {{
        "result": {{
            "validations": [
                {{
                    "reason" : "<copied reason>",
                    "validity": {{ "score": <float>, "reason": "..." }},
                    "reliability": {{ "score": <float>, "reason": "..." }}
                }}
            ]
        }}
    }}

    {JSONFORMAT}
    {LANGUAGE}
    """,
    output_key="result",
    output_schema=Result,
    include_contents="none",
    after_agent_callback=compute_crit_score_callback(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# === CRIT Agent as a Sequential Pipeline ===
CRITAgent = SequentialAgent(
    name="CRITAgent",
    sub_agents=[ValidationAgent],
    description="Generate reasons for a claim, validate them, and compute CRIT score.",
)
