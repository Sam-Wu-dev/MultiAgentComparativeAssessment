#!/usr/bin/env python3
# aggregate_summeval_spearman_json.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import math


def load_round_json(metric_dir: Path) -> Optional[dict]:
    path = metric_dir / "round.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return None if (isinstance(v, float) and math.isnan(v)) else v
    except Exception:
        return None


# ── extractors: Spearman ρ ────────────────────────────────────────────
def extract_awr_sum_all_spearman(obj: dict) -> Optional[float]:
    try:
        analysis = obj.get("analysis") or {}
        awr = analysis.get("ArgumentWithReply") or {}
        sum_all = awr.get("sum_all") or {}
        return _safe_float(sum_all.get("spearman_rho"))
    except Exception:
        return None


def extract_awr_poe_gaussian_spearman(obj: dict) -> Optional[float]:
    try:
        analysis = obj.get("analysis") or {}
        awr = analysis.get("ArgumentWithReply") or {}
        poe = awr.get("poe_gaussian") or {}
        return _safe_float(poe.get("spearman_rho"))
    except Exception:
        return None


def extract_awr_lse_spearman(obj: dict) -> Optional[float]:
    try:
        analysis = obj.get("analysis") or {}
        awr = analysis.get("ArgumentWithReply") or {}
        lse = awr.get("lse") or {}
        return _safe_float(lse.get("spearman_rho"))
    except Exception:
        return None


# ── extractors: p-values ─────────────────────────────────────────────
def extract_awr_sum_all_pvalue(obj: dict) -> Optional[float]:
    try:
        analysis = obj.get("analysis") or {}
        awr = analysis.get("ArgumentWithReply") or {}
        sum_all = awr.get("sum_all") or {}
        return _safe_float(sum_all.get("p_value"))
    except Exception:
        return None


def extract_awr_poe_gaussian_pvalue(obj: dict) -> Optional[float]:
    try:
        analysis = obj.get("analysis") or {}
        awr = analysis.get("ArgumentWithReply") or {}
        poe = awr.get("poe_gaussian") or {}
        return _safe_float(poe.get("p_value"))
    except Exception:
        return None


def extract_awr_lse_pvalue(obj: dict) -> Optional[float]:
    try:
        analysis = obj.get("analysis") or {}
        awr = analysis.get("ArgumentWithReply") or {}
        lse = awr.get("lse") or {}
        return _safe_float(lse.get("p_value"))
    except Exception:
        return None


def collect_article_metric_rhos(out_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for article_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        article_id = article_dir.name
        for metric_dir in sorted([p for p in article_dir.iterdir() if p.is_dir()]):
            metric = metric_dir.name
            obj = load_round_json(metric_dir)
            if not obj:
                continue

            rho_sum_all = extract_awr_sum_all_spearman(obj)
            rho_poe = extract_awr_poe_gaussian_spearman(obj)
            rho_lse = extract_awr_lse_spearman(obj)

            p_sum_all = extract_awr_sum_all_pvalue(obj)
            p_poe = extract_awr_poe_gaussian_pvalue(obj)
            p_lse = extract_awr_lse_pvalue(obj)

            if all(
                v is None
                for v in (rho_sum_all, rho_poe, rho_lse, p_sum_all, p_poe, p_lse)
            ):
                continue

            rows.append(
                {
                    "article_id": article_id,
                    "metric": metric,
                    "sum_all": (
                        float(rho_sum_all) if rho_sum_all is not None else float("nan")
                    ),
                    "sum_all_p": (
                        float(p_sum_all) if p_sum_all is not None else float("nan")
                    ),
                    "poe_gaussian": (
                        float(rho_poe) if rho_poe is not None else float("nan")
                    ),
                    "poe_gaussian_p": (
                        float(p_poe) if p_poe is not None else float("nan")
                    ),
                    "lse": float(rho_lse) if rho_lse is not None else float("nan"),
                    "lse_p": float(p_lse) if p_lse is not None else float("nan"),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "article_id",
            "metric",
            "sum_all",
            "sum_all_p",
            "poe_gaussian",
            "poe_gaussian_p",
            "lse",
            "lse_p",
        ],
    )


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate SummEval Spearman ρ for ArgumentWithReply "
            "(sum_all, poe_gaussian, lse) and save JSON."
        )
    )
    ap.add_argument(
        "--out_root",
        required=True,
        type=Path,
        help="Root dir with per-article outputs.",
    )
    ap.add_argument(
        "--out_json", required=True, type=Path, help="Path to save results JSON."
    )
    args = ap.parse_args()

    df = collect_article_metric_rhos(args.out_root)
    if df.empty:
        print("[ERROR] No values found.", file=sys.stderr)
        sys.exit(2)

    # Per-article dict
    per_article: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (aid, m), sub in df.groupby(["article_id", "metric"]):
        row = sub.iloc[0]

        metric_entry: Dict[str, Dict[str, float]] = {}

        # sum_all
        sum_block: Dict[str, float] = {}
        if pd.notna(row["sum_all"]):
            sum_block["spearman_rho"] = float(row["sum_all"])
        if pd.notna(row["sum_all_p"]):
            sum_block["p_value"] = float(row["sum_all_p"])
        if sum_block:
            metric_entry["sum_all"] = sum_block

        # poe_gaussian
        poe_block: Dict[str, float] = {}
        if pd.notna(row["poe_gaussian"]):
            poe_block["spearman_rho"] = float(row["poe_gaussian"])
        if pd.notna(row["poe_gaussian_p"]):
            poe_block["p_value"] = float(row["poe_gaussian_p"])
        if poe_block:
            metric_entry["poe_gaussian"] = poe_block

        # lse (NEW)
        lse_block: Dict[str, float] = {}
        if pd.notna(row["lse"]):
            lse_block["spearman_rho"] = float(row["lse"])
        if pd.notna(row["lse_p"]):
            lse_block["p_value"] = float(row["lse_p"])
        if lse_block:
            metric_entry["lse"] = lse_block

        if metric_entry:
            per_article.setdefault(aid, {})[m] = metric_entry

    # Per-metric averages (Spearman ρ)
    per_metric_avg = {
        "sum_all": (
            df.dropna(subset=["sum_all"])
            .groupby("metric")["sum_all"]
            .mean()
            .round(6)
            .to_dict()
        ),
        "poe_gaussian": (
            df.dropna(subset=["poe_gaussian"])
            .groupby("metric")["poe_gaussian"]
            .mean()
            .round(6)
            .to_dict()
        ),
        "lse": (
            df.dropna(subset=["lse"]).groupby("metric")["lse"].mean().round(6).to_dict()
        ),
    }

    # Global averages (Spearman ρ)
    global_avg = {
        "sum_all": (
            float(df["sum_all"].dropna().mean())
            if df["sum_all"].notna().any()
            else None
        ),
        "poe_gaussian": (
            float(df["poe_gaussian"].dropna().mean())
            if df["poe_gaussian"].notna().any()
            else None
        ),
        "lse": (float(df["lse"].dropna().mean()) if df["lse"].notna().any() else None),
    }
    for k in list(global_avg.keys()):
        if global_avg[k] is not None:
            global_avg[k] = round(global_avg[k], 6)

    result = {
        "per_article": per_article,
        "per_metric_avg": per_metric_avg,
        "global_avg": global_avg,
    }

    args.out_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] Saved results to {args.out_json}")


if __name__ == "__main__":
    main()
"""
python aggregate_summeval_spearman_json.py \
  --out_root /home/shaosen/LLM/Multi-agent/Saving/SummEval/cnn_5 \
  --out_json /home/shaosen/LLM/Multi-agent/Saving/SummEval/cnn_5/spearman_summary_K64.json
  
python aggregate_summeval_spearman_json.py \
  --out_root /home/shaosen/LLM/Multi-agent/Saving/SummEval/cnn_6 \
  --out_json /home/shaosen/LLM/Multi-agent/Saving/SummEval/cnn_6/spearman_summary.json
"""
