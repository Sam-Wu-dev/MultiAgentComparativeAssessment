# pairwise_multi_agent_ranker/Pipeline/selection.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def build_A_inv(
    n: int, chosen_pairs: List[Tuple[int, int]], ridge: float = 1e-6
) -> np.ndarray:
    """
    A = W_tilde^T W_tilde with one anchor row on s[0].
    Returns (A)^(-1) robustly (ridge + pinv fallback).
    """
    K = len(chosen_pairs)
    W = np.zeros((K, n), dtype=float)
    for k, (i, j) in enumerate(chosen_pairs):
        W[k, i] = 1.0
        W[k, j] = -1.0

    # Anchor s[0] to 0 to remove translation DOF
    W_tilde = np.vstack([np.eye(1, n, 0), W])  # [e0^T; W]
    A = W_tilde.T @ W_tilde  # n x n

    if ridge and ridge > 0:
        A = A + (ridge * np.eye(n))

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)  # safe fallback

    return A_inv


def next_pair_greedy(n: int, chosen: set[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Pick (i,j) not in chosen that maximizes A_ii + A_jj - 2 A_ij.
    """
    A_inv = build_A_inv(n, list(chosen))
    best = None
    best_val = -1e18
    for i in range(n):
        for j in range(n):
            if i == j or (i, j) in chosen:
                continue
            val = A_inv[i, i] + A_inv[j, j] - 2.0 * A_inv[i, j]
            if val > best_val:
                best_val = val
                best = (i, j)
    return best
