from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from Domain.candidate import Candidate
from Domain.pair import Pair
from Domain.round_analysis import RoundAnalysis


class Round(BaseModel):
    """
    A single 'round' of pairwise comparisons.
    """

    folder: Path
    candidates: List[Candidate] = Field(default_factory=list)

    # Round-scoped prompt settings
    committee_context_template: str
    claim_template: str
    member_name: str

    # NEW: planned comparison budget (can be None for full matrix)
    budget: Optional[int] = None

    analysis: Optional[RoundAnalysis] = None
    pairs: Optional[List[Pair]] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    # Use the round's own budget when naming; fallback to actual #pairs
    def round_meta_filename(self) -> str:
        K = self.budget
        if K is None and self.pairs is not None:
            K = len(self.pairs)
        return "round.json" if not K else f"round_K{int(K)}.json"

    @staticmethod
    def pair_filename(nameA: str, nameB: str) -> str:
        safeA = str(nameA).replace("/", "_")
        safeB = str(nameB).replace("/", "_")
        return f"{safeA}_{safeB}.json"

    @classmethod
    def create(
        cls,
        folder: Path,
        candidates: List[Candidate],
        *,
        committee_context_template: str,
        member_name: str,
        claim_template: str,
        budget: Optional[int] = None,
    ) -> "Round":
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        return cls(
            folder=folder,
            candidates=candidates,
            analysis=None,
            pairs=None,
            committee_context_template=committee_context_template,
            member_name=member_name,
            claim_template=claim_template,
            budget=budget,
        )

    def build_claims(self, nameA: str, nameB: str) -> Tuple[str, str]:
        return self.claim_template.format(A=nameA, B=nameB), self.claim_template.format(
            A=nameB, B=nameA
        )

    def build_context(self, nameA: str, nameB: str) -> str:
        return self.committee_context_template.format(A=nameA, B=nameB)

    # --------------- persistence ---------------

    def save(self) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)

        if self.pairs:
            for p in self.pairs:
                p.save_json()

        # Write the main file with budget in the filename when available
        meta_path = self.folder / self.round_meta_filename()
        obj = self.model_dump(by_alias=False, exclude_none=False)
        obj["folder"] = str(self.folder)
        meta_path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Also write a compatibility copy as round.json if we’re budgeted
        if meta_path.name != "round.json":
            compat = self.folder / "round.json"
            compat.write_text(
                json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        return meta_path

    def pair_path(self, pair: Pair) -> Path:
        return self.folder / self.pair_filename(pair.nameA, pair.nameB)
