# Pipeline/make_pairs.py
from __future__ import annotations
from typing import List, Optional

from Domain.round import Round
from Domain.pair import Pair
from Pipeline.selection import next_pair_greedy


def ensure_pairs(
    rnd: Round,
    K: Optional[int] = None,
    reference: Optional[str] = None,
) -> List[Pair]:
    """
    Build pairs and run compare/analysis with retries, persisting progress.

    Behavior:
      - If K is None or K >= n*(n-1): do full directed matrix (all permutations).
      - Else: build exactly K directed pairs using greedy selection with a
        connectivity-guaranteed seeding (star to node 0).
    """
    n = len(rnd.candidates)
    all_budget = n * (n - 1)
    do_full = (K is None) or (K >= all_budget)

    chosen: list[tuple[int, int]] = []

    if do_full:
        # Full directed matrix
        chosen = [(i, j) for i in range(n) for j in range(n) if i != j]
    else:
        # Efficient: star seeding to node 0 (connectivity), then greedy fill
        picked: set[tuple[int, int]] = set()
        for i in range(1, n):
            if len(picked) >= K:
                break
            picked.add((i, 0))
        while len(picked) < K:
            ij = next_pair_greedy(n, picked)
            if ij is None:
                break
            picked.add(ij)
        chosen = list(picked)

    pairs: List[Pair] = []

    for i, j in chosen:
        a = rnd.candidates[i]
        b = rnd.candidates[j]
        pair_path = rnd.folder / rnd.pair_filename(a.name, b.name)

        claimA, claimB = rnd.build_claims(a.name, b.name)
        ctx = rnd.build_context(a.name, b.name)

        p = Pair(
            canA=a,
            canB=b,
            nameA=a.name,
            nameB=b.name,
            path=pair_path,
            reference=reference,
            member_name=rnd.member_name,
            committee_context=ctx,
            claimA=claimA,
            claimB=claimB,
        )
        has_artifacts = p.load_artifacts()
        has_analysis = p.load_analysis()
        if not has_artifacts:
            success = False
            for attempt in range(2):
                try:
                    print(f"[run ] compare {a.name} vs {b.name} (try {attempt+1}) …")
                    ok = p.compare()
                    if not ok:
                        print(f"[warn] artifacts incomplete for {a.name} vs {b.name}")
                    p.save_json()  # persist partial progress
                    success = True
                    break
                except Exception as e:
                    print(
                        f"[error] compare failed {a.name} vs {b.name} try {attempt+1}: {e}"
                    )
            if not success:
                print(
                    f"[warn] giving up on compare for {a.name} vs {b.name} after 2 attempts"
                )
        if not has_analysis or not has_artifacts:
            try:
                if p.run_analysis():
                    p.save_json()
                    print(f"[score] {pair_path.name}")
            except Exception as e:
                print(f"[warn] scoring failed ({a.name} vs {b.name}): {e}")

        pairs.append(p)

    return pairs
