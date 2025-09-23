from typing import List
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from Shared.callback import *
from Shared import MODEL
from ..workflow import *


class Score(BaseModel):
    score: float
    reason: str


class Evaluation(BaseModel):
    type: str
    validity: Score
    reliability: Score


class CRIT(BaseModel):
    CRIT: List[Evaluation]


ValidationAgent = LlmAgent(
    name="ValidationAgent",
    model=MODEL,
    instruction="""
        In the Context: {context}
        For each reason (r) in {reasons}:
        1. Evaluate the **validity** of the reasoning behind r ⇒ Ω.
        2. Evaluate the **reliability** of the evidence supporting r ⇒ Ω.
        3. Identify the type of the reason (r), it can be either theory/opinion/statistics/claim.
        Where Ω is the claim: {claim}
        
        Follow the output schema:
        {
          "CRIT": [
            {
              "type": "theory/opinion/statistics/claim",
              "validity": {"score": 0, "reason": "Validity assessment"},
              "reliability": {"score": 0, "reason": "Reliability assessment"}
            },
            ...
          ]
        }
        
        Make sure to:
        - DO NOT use triple backticks like ```json
        - Use a score between 0 and 10 for both validity and reliability.
        - The JSON is strictly valid (no trailing commas).
        - Respond in Traditional Chinese.
        """,
    output_key="CRIT_SCORE",
    output_schema=CRIT,
    include_contents="none",
    description="An agent that evaluates the validity and reliability of reasons in a debate context.",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
