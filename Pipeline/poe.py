# Pipeline/poe.py
from __future__ import annotations
from typing import Iterable, Tuple, List
import numpy as np

PairObs = Tuple[
    int, int, float
]  # (i, j, p_ij) meaning "i better than j" with prob p_ij


def poe_gaussian_scores(n: int, comps: Iterable[PairObs]) -> np.ndarray:
    """
    PoE-Gaussian with linear mean and constant variance.
    Inputs:
      n: number of candidates
      comps: iterable of (i, j, p_ij)
    Returns:
      s_hat: np.ndarray shape (n,) predicted scores
    """
    comps = list(comps)
    if len(comps) == 0:
        return np.zeros(n, dtype=float)

    # De-bias: beta = E[p_ij]
    ps = np.array([p for (_i, _j, p) in comps], dtype=float)
    beta = float(ps.mean()) if np.isfinite(ps.mean()) else 0.5

    # Build W and mu (un-anchored)
    K = len(comps)
    W = np.zeros((K, n), dtype=float)
    mu = np.zeros((K,), dtype=float)
    for k, (i, j, p) in enumerate(comps):
        W[k, i] = 1.0
        W[k, j] = -1.0
        mu[k] = p - beta  # linear mean f_mu(p) = (p - beta)

    # Add one "anchor" row to fix the translation degree of freedom: s_0 ~ N(0, sigma^2)
    W_tilde = np.vstack([np.eye(1, n, 0), W])  # first row selects s[0]
    mu_tilde = np.concatenate([[0.0], mu])

    # Closed-form solution: s_hat = (W^T W)^(-1) W^T mu
    A = W_tilde.T @ W_tilde  # n x n
    b = W_tilde.T @ mu_tilde  # n
    # Solve robustly
    s_hat = np.linalg.solve(A, b)
    return s_hat


def rank_from_scores(s: np.ndarray) -> List[int]:
    # 1=best competition ranking (1,1,3,...) on descending scores
    order = np.argsort(-s)  # descending
    ranks = [0] * len(s)
    last_score = None
    last_rank = 0
    for idx, pos in enumerate(order, start=1):
        sc = s[pos]
        if last_score is None or sc < last_score:
            last_rank = idx
            last_score = sc
        ranks[pos] = last_rank
    return ranks
