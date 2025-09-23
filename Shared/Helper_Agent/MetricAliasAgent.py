from google.adk.agents import LlmAgent
from google.genai import types
from pydantic import BaseModel
from typing import List, Dict
from Shared import MODEL
from Shared.prompt import *


class AliasMapping(BaseModel):
    canonical: str
    aliases: List[str]


class AliasResult(BaseModel):
    mappings: List[AliasMapping]


MetricAliasAgent = LlmAgent(
    name="MetricAliasAgent",
    model=MODEL,
    instruction=f"""
    {DEBATE_CONTEXT}
    You are given a list of evaluation feature names that appeared in many pairwise debates:
    {{feature_names}}
    Your job is to solve the alias problem by identifying which features are actually the same but have different names:
    For example "資安實戰經驗" and "資安實務經驗" are actually the same feature.
    Follow this exact JSON output format:
    {{
        "mappings": [
            {{
                "canonical": <canonical name>,
                "aliases": [
                    "資安實戰經驗",
                    "資安實務經驗"
                ]
            }},
            ...
        ]
    }}
    {JSONFORMAT}
    {LANGUAGE}
    """,
    output_key="result",
    output_schema=AliasResult,
    include_contents="none",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
