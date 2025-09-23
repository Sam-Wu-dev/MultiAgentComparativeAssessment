# Domain/pair.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

from Domain.candidate import Candidate
from Domain.artifact import PairArtifacts
from Domain.pair_analysis import PairAnalysis, build_analysis
from Pipeline.compare import run_compare_pipeline


class Pair(BaseModel):
    canA: Candidate
    canB: Candidate

    # Identifiers
    nameA: str
    nameB: str
    path: Path

    # Round-scoped prompt settings captured at pair-time (persisted)
    reference: Optional[str] = None
    member_name: str
    committee_context: str

    # Resolved claims (must be concrete strings, not templates)
    claimA: str
    claimB: str

    # Results
    artifacts: Optional[PairArtifacts] = None
    analysis: Optional[PairAnalysis] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_artifacts(self) -> bool:
        p = Path(self.path)
        if not p.exists():
            print(f"[warn] path not exist.")
            return False
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            art = obj.get("artifacts")
            if not isinstance(art, dict):
                print(f"[warn] artifacts not exist.")
                return False
            self.artifacts = PairArtifacts.model_validate(art)
            is_complete = self.artifacts.is_complete()
            if not is_complete:
                print(f"[warn] artifacts not complete.")
            return is_complete
        except Exception as e:
            print(f"[warn] failed to load artifacts from {p.name}: {e}")
            return False

    def load_analysis(self) -> bool:
        p = Path(self.path)
        if not p.exists():
            return False
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            ana = obj.get("analysis")
            if not isinstance(ana, dict):
                return False
            self.analysis = PairAnalysis.model_validate(ana)
            return self.analysis_complete()
        except Exception:
            print(f"[warn] failed to load analysis from {p.name}")
            return False

    def compare(self) -> bool:
        """Populate artifacts by running the pipeline."""
        self.artifacts = run_compare_pipeline(
            self.canA,
            self.canB,
            committee_context=self.committee_context,
            member_name=self.member_name,
            claimA=self.claimA,
            claimB=self.claimB,
        )
        return self.artifacts and self.artifacts.is_complete()

    def run_analysis(self) -> bool:
        if self.artifacts is None:
            raise ValueError("run_analysis called before compare(): artifacts is None")
        self.analysis = build_analysis(self.artifacts)
        return True

    def save_json(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        obj = self.model_dump(by_alias=False, exclude_none=True)
        obj["path"] = str(self.path)

        self.path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.path
