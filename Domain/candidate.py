# Domain/candidate.py
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class Candidate(BaseModel):
    name: str
    document: str
    gt_score: Optional[float] = None
    gt_rank: Optional[int] = None
