#!/usr/bin/env python3
"""
Personal fusion weight tuner for rime-neural (unbiased Newton-Raphson MAP shrinkage).

Statistical anti-overfitting design:
  1. `lua/rime_neural.lua` logs 100% of multi-candidate sentence commits (`sample_weight = 1.0`,
     bounded by 5 MiB auto-rotation) to eliminate selection bias. Legacy 10%-sampled Top-1 logs
     remain supported via `--top1-sample-rate 0.10` (Inverse-Probability Weighting).
  2. Fits exact damped Newton-Raphson MAP estimation around `W_PRIOR` with Gaussian prior
     `0.5 * lam * ||w - W_PRIOR||^2` (equivalent to Bayesian linear updating under Gaussian prior).
  3. Selects regularization strength `lam` by 5-fold out-of-fold weighted **log-loss (NLL)**
     (strictly proper scoring rule; far lower variance than 0-1 step accuracy).
  4. Only recommends an update when ALL THREE guards pass:
       - Minimum sample guards are met (default >= 100 matched records, >= 15 corrections, >= 30 accepts)
       - Out-of-fold weighted NLL improvement passes a one-sided paired 95% CI test (z >= 1.645)
         using Kish effective sample size `N_eff = (sum w)^2 / sum(w^2)`
       - Out-of-fold weighted Top-1 accuracy does not drop below `W_PRIOR`.

Usage:
    python scripts/tune-fusion.py
    python scripts/tune-fusion.py --log path/to/feedback.jsonl --min-samples 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
from typing import List, Optional, Tuple

import numpy as np

# General OOD-verified prior weights: [w_logit, w_rank, w_gram, w_char_lp, w_ctx_lp]
W_PRIOR = np.array([1.0, -1.4, 0.05, 0.15, 0.0], dtype=np.float64)
LAMBDA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def default_log_path() -> str:
    appdata = os.environ.get("APPDATA", "")
    primary = pathlib.Path(appdata) / "Rime" / "neural" / "feedback.jsonl"
    if primary.is_file():
        return str(primary)
    legacy = pathlib.Path(appdata) / "Rime" / "zhiyi" / "feedback.jsonl"
    if legacy.is_file():
        return str(legacy)
    return str(primary)


def log_softmax(xs: List[float]) -> np.ndarray:
    arr = np.array(xs, dtype=np.float64)
    m = float(np.max(arr))
    return arr - (m + math.log(float(np.sum(np.exp(arr - m)))))


def load_samples(
    path: pathlib.Path,
    top1_sample_rate: float = 1.0,
) -> Tuple[List[np.ndarray], List[int], np.ndarray]:
    if not path.is_file():
        return [], [], np.array([], dtype=np.float64)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    w_list: List[float] = []
    top1_weight = 1.0 / max(1e-3, top1_sample_rate)

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        cands = rec.get("cands") or []
        logits = rec.get("logits") or []
        grams = rec.get("gram_deltas") or [0.0] * len(cands)
        clps = rec.get("char_lps") or [0.0] * len(cands)
        chosen = rec.get("chosen_text") or ""
        if len(cands) < 2 or len(logits) != len(cands) or chosen not in cands:
            continue
        y = cands.index(chosen)
        lp = log_softmax(logits)
        has_ctx = 1.0 if len((rec.get("context") or "").strip()) >= 2 else 0.0
        X = np.array(
            [
                [lp[i], math.log(i + 1.0), float(grams[i]), float(clps[i]), lp[i] * has_ctx]
                for i in range(len(cands))
            ],
            dtype=np.float64,
        )
        if "sample_weight" in rec and float(rec["sample_weight"]) > 0.0:
            sw = float(rec["sample_weight"])
        else:
            chosen_rank = int(rec.get("chosen_rank", 0))
            sw = top1_weight if chosen_rank == 1 else 1.0
        X_list.append(X)
        y_list.append(y)
        w_list.append(sw)

    return X_list, y_list, np.array(w_list, dtype=np.float64)


def fit_newton_offset_ridge(
    X_list: List[np.ndarray],
    y_list: List[int],
    weights: np.ndarray,
    lam: float,
    w_prior: np.ndarray = W_PRIOR,
    max_iters: int = 15,
) -> np.ndarray:
    """Exact Newton-Raphson solver for weighted listwise NLL + 0.5 * lam * ||w - w_prior||^2."""
    d = len(w_prior)
    w = w_prior.copy()
    tot_w = float(np.sum(weights))
    if tot_w <= 0.0:
        return w
    eye = np.eye(d, dtype=np.float64)

    for _ in range(max_iters):
        grad = lam * (w - w_prior)
        hess = lam * eye
        for X, y, wt in zip(X_list, y_list, weights):
            s = X @ w
            s -= float(np.max(s))
            p = np.exp(s)
            p /= float(np.sum(p))
            c = float(wt) / tot_w
            diff = p.copy()
            diff[y] -= 1.0
            grad += c * (X.T @ diff)
            mu = p @ X
            Xc = X - mu
            hess += c * ((Xc * p[:, None]).T @ Xc)
        step = np.linalg.solve(hess + 1e-8 * eye, grad)
        w -= step
        if float(np.linalg.norm(step)) < 1e-7:
            break
    return w


def eval_nll_vec(X_list: List[np.ndarray], y_list: List[int], w: np.ndarray) -> np.ndarray:
    out = np.empty(len(y_list), dtype=np.float64)
    for i, (X, y) in enumerate(zip(X_list, y_list)):
        s = X @ w
        s -= float(np.max(s))
        out[i] = math.log(float(np.sum(np.exp(s)))) - float(s[y])
    return out


def eval_weighted_acc(
    X_list: List[np.ndarray], y_list: List[int], weights: np.ndarray, w: np.ndarray
) -> float:
    tot_w = float(np.sum(weights))
    if tot_w <= 0.0:
        return 0.0
    hits = sum(
        float(wt)
        for X, y, wt in zip(X_list, y_list, weights)
        if int(np.argmax(X @ w)) == y
    )
    return hits / tot_w * 100.0


def kish_effective_n(weights: np.ndarray) -> float:
    s1 = float(np.sum(weights))
    s2 = float(np.sum(weights * weights))
    return (s1 * s1 / s2) if s2 > 0.0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Unbiased MAP personal fusion weight tuner for rime-neural")
    ap.add_argument(
        "--log",
        default=default_log_path(),
        help="Path to local feedback.jsonl (defaults to %%APPDATA%%\\Rime\\neural\\feedback.jsonl)",
    )
    ap.add_argument("--min-samples", type=int, default=100, help="Minimum total matched records")
    ap.add_argument("--min-corrections", type=int, default=15, help="Minimum correction records")
    ap.add_argument("--min-accepts", type=int, default=30, help="Minimum accepted Top-1 records")
    ap.add_argument(
        "--top1-sample-rate",
        type=float,
        default=1.0,
        help="Sampling rate of Top-1 accepts for legacy logs without sample_weight (default 1.0)",
    )
    args = ap.parse_args()

    log_path = pathlib.Path(args.log)
    X_list, y_list, weights = load_samples(log_path, args.top1_sample_rate)
    n = len(X_list)
    n_accepts = sum(1 for X, y in zip(X_list, y_list) if int(np.argmax(X @ W_PRIOR)) == y)
    n_corrections = n - n_accepts
    n_eff = kish_effective_n(weights)

    print(
        f"Loaded {n} valid records from {log_path} "
        f"(corrections={n_corrections}, accepts={n_accepts}, Kish N_eff={n_eff:.1f})"
    )
    if n < args.min_samples or n_corrections < args.min_corrections or n_accepts < args.min_accepts:
        print(
            f"Safety guard: need >= {args.min_samples} total, >= {args.min_corrections} corrections, "
            f">= {args.min_accepts} Top-1 accepts to prevent overfitting (keeping prior weights)."
        )
        print("Current prior fusion_weights:", [round(float(x), 3) for x in W_PRIOR])
        return

    prior_nll_vec = eval_nll_vec(X_list, y_list, W_PRIOR)
    prior_nll = float(np.sum(weights * prior_nll_vec) / np.sum(weights))
    prior_acc = eval_weighted_acc(X_list, y_list, weights, W_PRIOR)
    print(f"Prior weights: weighted NLL={prior_nll:.4f} | weighted Top-1 acc={prior_acc:.2f}%")

    folds = 5
    best_lam: Optional[float] = None
    best_nll = prior_nll
    best_oof_nll: Optional[np.ndarray] = None
    best_oof_acc = prior_acc

    for lam in LAMBDA_GRID:
        oof_nll = np.zeros(n, dtype=np.float64)
        oof_hit_w = 0.0
        for f in range(folds):
            tr_i = [i for i in range(n) if i % folds != f]
            va_i = [i for i in range(n) if i % folds == f]
            wf = fit_newton_offset_ridge(
                [X_list[i] for i in tr_i], [y_list[i] for i in tr_i], weights[tr_i], lam
            )
            for i in va_i:
                s = X_list[i] @ wf
                s -= float(np.max(s))
                oof_nll[i] = math.log(float(np.sum(np.exp(s)))) - float(s[y_list[i]])
                if int(np.argmax(s)) == y_list[i]:
                    oof_hit_w += float(weights[i])
        wl = float(np.sum(weights * oof_nll) / np.sum(weights))
        wa = oof_hit_w / float(np.sum(weights)) * 100.0
        print(f"  lambda={lam:5.1f} -> 5-fold NLL={wl:.4f} | Top-1 acc={wa:.2f}%")
        if wl < best_nll - 1e-6:
            best_nll = wl
            best_lam = lam
            best_oof_nll = oof_nll
            best_oof_acc = wa

    if best_lam is None or best_oof_nll is None:
        print("\nPrior weights already achieve the lowest 5-fold log-loss; no update needed.")
        return

    diff = prior_nll_vec - best_oof_nll
    mean_d = float(np.sum(weights * diff) / np.sum(weights))
    var_d = float(np.sum(weights * (diff - mean_d) ** 2) / np.sum(weights))
    se_d = math.sqrt(var_d / max(1.0, n_eff))
    z_score = mean_d / se_d if se_d > 1e-12 else 0.0

    print(
        f"\nBest lambda={best_lam}: delta_NLL={mean_d:+.4f} +/- {se_d:.4f} (z={z_score:.2f}, "
        f"acc {prior_acc:.2f}% -> {best_oof_acc:.2f}%)"
    )
    if z_score < 1.645 or best_oof_acc < prior_acc - 1e-6:
        print(
            "Improvement is NOT statistically significant at 95% confidence (or drops Top-1 acc); "
            "keeping prior weights to avoid overfitting."
        )
        return

    w_new = fit_newton_offset_ridge(X_list, y_list, weights, best_lam)
    w_fmt = [round(float(x), 3) for x in w_new]
    print(f"Statistically significant improvement verified (z={z_score:.2f} >= 1.645)!")
    print("Paste into rime_neural/fusion_weights in your schema custom yaml:")
    print(f"  fusion_weights: {w_fmt}")


if __name__ == "__main__":
    main()
