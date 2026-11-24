#!/usr/bin/env python3
"""
Personal fusion weight tuner for rime-neural (unbiased Newton-Raphson MAP shrinkage).

Statistical anti-overfitting design:
  1. `lua/rime_neural.lua` logs 100% of multi-candidate sentence commits as zero-plaintext
     numeric feature records (`v: 2`, `sample_weight = 1.0`, bounded by 5 MiB auto-rotation)
     to eliminate selection bias while never persisting user text. Legacy v1 records remain
     readable, and legacy 10%-sampled Top-1 logs remain supported via `--top1-sample-rate 0.10`.
  2. Fits exact damped Newton-Raphson Bayesian MAP estimation around `W_PRIOR` with Gaussian
     prior precision `lam`:
         J(w) = (1 / sum_w) * sum_i w_i * NLL_i(w) + (lam / (2 * sum_w)) * ||w - W_PRIOR||^2
     so prior strength stays fixed in effective sample count and vanishes asymptotically as N -> inf.
  3. Selects prior precision `lam` by contiguous blocked 5-fold out-of-fold weighted **log-loss
     (NLL)** (strictly proper scoring rule; contiguous folds prevent adjacent-session leakage).
  4. Only recommends an update when ALL THREE guards pass:
       - Minimum sample guards are met (default >= 100 matched records, >= 15 corrections, >= 30 accepts)
       - Out-of-fold weighted NLL improvement passes a one-sided paired 95% CI test (z >= 1.645)
         using the heteroscedasticity-robust Hajek sandwich variance estimator
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
from typing import Any, List, Optional, Tuple

import numpy as np

# General OOD-verified prior weights: [w_logit, w_rank, w_gram, w_char_lp, w_ctx_lp]
W_PRIOR = np.array([1.0, -1.4, 0.05, 0.15, 0.0], dtype=np.float64)
# Prior precision grid (effective prior sample count; scaled by 1 / sum_w in the mean NLL objective)
LAMBDA_GRID = (0.5, 1.5, 5.0, 15.0, 50.0, 150.0, 500.0)


def _registry_rime_user_dir() -> Optional[pathlib.Path]:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Rime\Weasel") as k:
            val, _ = winreg.QueryValueEx(k, "RimeUserDir")
            if isinstance(val, str) and val.strip():
                return pathlib.Path(val.strip())
    except OSError:
        pass
    return None


def default_log_path() -> str:
    roots: List[pathlib.Path] = []
    reg_dir = _registry_rime_user_dir()
    if reg_dir is not None:
        roots.append(reg_dir)
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        roots.append(pathlib.Path(appdata) / "Rime")
    for root in roots:
        for sub in ("neural", "zhiyi"):
            cand = root / sub / "feedback.jsonl"
            if cand.is_file():
                return str(cand)
    fallback_root = roots[0] if roots else pathlib.Path(".")
    return str(fallback_root / "neural" / "feedback.jsonl")


def log_softmax(xs: List[float]) -> np.ndarray:
    arr = np.array(xs, dtype=np.float64)
    m = float(np.max(arr))
    return arr - (m + math.log(float(np.sum(np.exp(arr - m)))))


def _parse_finite_float_list(raw: Any, expected_len: int, allow_empty: bool = False) -> Optional[List[float]]:
    if raw is None or (allow_empty and raw == []):
        return [0.0] * expected_len if allow_empty else None
    if not isinstance(raw, list) or len(raw) != expected_len:
        return None
    out: List[float] = []
    for v in raw:
        if v is None or isinstance(v, bool):
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(fv):
            return None
        out.append(fv)
    return out


def _iter_log_lines(path: pathlib.Path, include_rotated: bool = True) -> List[str]:
    files: List[pathlib.Path] = []
    if include_rotated:
        rotated = pathlib.Path(str(path) + ".1")
        if rotated.is_file():
            files.append(rotated)
    if path.is_file():
        files.append(path)
    lines: List[str] = []
    for f in files:
        lines.extend(f.read_text(encoding="utf-8", errors="ignore").splitlines())
    return lines


def load_samples(
    path: pathlib.Path,
    top1_sample_rate: float = 1.0,
    include_rotated: bool = True,
) -> Tuple[List[np.ndarray], List[int], np.ndarray, dict]:
    summary = {
        "total_lines": 0,
        "matched": 0,
        "p2c_accepted": 0,
        "disagreed": 0,
        "neural_win": 0,
        "rime_revert": 0,
    }
    raw_lines = _iter_log_lines(path, include_rotated=include_rotated)
    if not raw_lines:
        return [], [], np.array([], dtype=np.float64), summary

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    w_list: List[float] = []
    top1_weight = 1.0 / max(1e-3, top1_sample_rate)

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        summary["total_lines"] += 1

        is_v2 = ("chosen_idx" in rec) or (isinstance(rec.get("v"), int) and rec["v"] >= 2)
        if is_v2:
            raw_logits = rec.get("logits")
            if not isinstance(raw_logits, list) or len(raw_logits) < 2:
                continue
            num_cands = len(raw_logits)
            logits = _parse_finite_float_list(raw_logits, num_cands)
            grams = _parse_finite_float_list(rec.get("gram_deltas"), num_cands, allow_empty=True)
            clps = _parse_finite_float_list(rec.get("char_lps"), num_cands, allow_empty=True)
            if logits is None or grams is None or clps is None:
                continue
            try:
                chosen_idx = int(rec.get("chosen_idx", -1))
                shown_top1_idx = int(rec.get("shown_top1_idx", -1))
            except (TypeError, ValueError):
                continue
            p2c_chosen = bool(rec.get("p2c_chosen", False))
            if p2c_chosen:
                summary["p2c_accepted"] += 1
                if shown_top1_idx == -2:
                    summary["disagreed"] += 1
                    summary["neural_win"] += 1
            if chosen_idx < 0 or chosen_idx >= num_cands:
                continue
            summary["matched"] += 1
            if shown_top1_idx != -1 and shown_top1_idx != 0:
                summary["disagreed"] += 1
                if chosen_idx == shown_top1_idx:
                    summary["neural_win"] += 1
                elif chosen_idx == 0:
                    summary["rime_revert"] += 1
            y = chosen_idx
            has_ctx = 1.0 if bool(rec.get("has_ctx", False)) else 0.0
        else:
            cands = rec.get("cands")
            if not isinstance(cands, list) or len(cands) < 2:
                continue
            num_cands = len(cands)
            logits = _parse_finite_float_list(rec.get("logits"), num_cands)
            grams = _parse_finite_float_list(rec.get("gram_deltas"), num_cands, allow_empty=True)
            clps = _parse_finite_float_list(rec.get("char_lps"), num_cands, allow_empty=True)
            if logits is None or grams is None or clps is None:
                continue
            chosen = rec.get("chosen_text") or ""
            shown_top1 = rec.get("shown_top1") or ""
            p2c_cand = rec.get("p2c_cand") or ""
            if p2c_cand and chosen == p2c_cand and chosen not in cands:
                summary["p2c_accepted"] += 1
                if shown_top1 == p2c_cand:
                    summary["disagreed"] += 1
                    summary["neural_win"] += 1
            if chosen not in cands:
                continue
            summary["matched"] += 1
            if shown_top1 and shown_top1 != cands[0]:
                summary["disagreed"] += 1
                if chosen == shown_top1:
                    summary["neural_win"] += 1
                elif chosen == cands[0]:
                    summary["rime_revert"] += 1
            y = cands.index(chosen)
            has_ctx = 1.0 if len((rec.get("context") or "").strip()) >= 2 else 0.0

        lp = log_softmax(logits)
        X = np.array(
            [
                [lp[i], math.log(i + 1.0), grams[i], clps[i], lp[i] * has_ctx]
                for i in range(num_cands)
            ],
            dtype=np.float64,
        )
        sw: Optional[float] = None
        if "sample_weight" in rec and rec["sample_weight"] is not None:
            try:
                cand_sw = float(rec["sample_weight"])
                if math.isfinite(cand_sw) and cand_sw > 0.0:
                    sw = cand_sw
            except (TypeError, ValueError):
                sw = None
        if sw is None:
            try:
                chosen_rank = int(rec.get("chosen_rank", 0))
            except (TypeError, ValueError):
                chosen_rank = 0
            sw = top1_weight if chosen_rank == 1 else 1.0
        X_list.append(X)
        y_list.append(y)
        w_list.append(sw)

    return X_list, y_list, np.array(w_list, dtype=np.float64), summary


def eval_penalized_obj(
    X_list: List[np.ndarray],
    y_list: List[int],
    weights: np.ndarray,
    tot_w: float,
    lam: float,
    w_prior: np.ndarray,
    w: np.ndarray,
) -> float:
    nll = 0.0
    for X, y, wt in zip(X_list, y_list, weights):
        s = X @ w
        s -= float(np.max(s))
        nll += (float(wt) / tot_w) * (math.log(float(np.sum(np.exp(s)))) - float(s[y]))
    delta = w - w_prior
    eff_lam = lam / tot_w
    return nll + 0.5 * eff_lam * float(delta @ delta)


def fit_newton_offset_ridge(
    X_list: List[np.ndarray],
    y_list: List[int],
    weights: np.ndarray,
    lam: float,
    w_prior: np.ndarray = W_PRIOR,
    max_iters: int = 20,
) -> np.ndarray:
    """Damped Newton-Raphson solver with Armijo backtracking line search for Bayesian MAP
    weighted listwise NLL + (lam / (2 * sum_w)) * ||w - w_prior||^2."""
    d = len(w_prior)
    w = w_prior.copy()
    tot_w = float(np.sum(weights))
    if tot_w <= 0.0:
        return w
    eff_lam = lam / tot_w
    eye = np.eye(d, dtype=np.float64)

    for _ in range(max_iters):
        grad = eff_lam * (w - w_prior)
        hess = eff_lam * eye
        obj_cur = 0.5 * eff_lam * float((w - w_prior) @ (w - w_prior))
        for X, y, wt in zip(X_list, y_list, weights):
            s = X @ w
            s -= float(np.max(s))
            exp_s = np.exp(s)
            sum_exp = float(np.sum(exp_s))
            p = exp_s / sum_exp
            c = float(wt) / tot_w
            obj_cur += c * (math.log(sum_exp) - float(s[y]))
            diff = p.copy()
            diff[y] -= 1.0
            grad += c * (X.T @ diff)
            mu = p @ X
            Xc = X - mu
            hess += c * ((Xc * p[:, None]).T @ Xc)
        step = np.linalg.solve(hess + 1e-8 * eye, grad)
        dir_deriv = -float(grad @ step)
        if dir_deriv >= 0.0:
            break
        alpha = 1.0
        accepted = False
        for _ in range(10):
            w_cand = w - alpha * step
            obj_cand = eval_penalized_obj(X_list, y_list, weights, tot_w, lam, w_prior, w_cand)
            if obj_cand <= obj_cur + 1e-4 * alpha * dir_deriv:
                w = w_cand
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        if alpha * float(np.linalg.norm(step)) < 1e-7:
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


def hajek_paired_se(diff: np.ndarray, weights: np.ndarray, mean_d: float) -> float:
    """Heteroscedasticity-robust Hajek sandwich SE for a weighted ratio mean."""
    n = len(diff)
    s1 = float(np.sum(weights))
    if n < 2 or s1 <= 0.0:
        return 0.0
    var_hat = (n / (n - 1.0)) * float(np.sum((weights * (diff - mean_d)) ** 2)) / (s1 * s1)
    return math.sqrt(max(0.0, var_hat))


def main() -> None:
    ap = argparse.ArgumentParser(description="Unbiased MAP personal fusion weight tuner for rime-neural")
    ap.add_argument(
        "--log",
        default=default_log_path(),
        help="Path to local feedback.jsonl (defaults to RimeUserDir/%%APPDATA%%\\Rime\\neural\\feedback.jsonl)",
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
    ap.add_argument(
        "--no-rotated",
        action="store_true",
        help="Do not include rotated feedback.jsonl.1 alongside feedback.jsonl",
    )
    args = ap.parse_args()

    log_path = pathlib.Path(args.log)
    X_list, y_list, weights, summary = load_samples(
        log_path, args.top1_sample_rate, include_rotated=not args.no_rotated
    )
    n = len(X_list)
    n_accepts = sum(1 for X, y in zip(X_list, y_list) if int(np.argmax(X @ W_PRIOR)) == y)
    n_corrections = n - n_accepts
    n_eff = kish_effective_n(weights)

    print(
        f"Loaded {n}/{summary['total_lines']} matched records from {log_path} "
        f"(corrections={n_corrections}, accepts={n_accepts}, Kish N_eff={n_eff:.1f})"
    )
    print(
        f"Rerank breakdown: disagreements={summary['disagreed']} "
        f"(neural_win={summary['neural_win']}, rime_revert={summary['rime_revert']}), "
        f"p2c_accepted={summary['p2c_accepted']}"
    )
    if (
        n < max(5, args.min_samples)
        or float(np.sum(weights)) <= 0.0
        or n_corrections < args.min_corrections
        or n_accepts < args.min_accepts
    ):
        print(
            f"Safety guard: need >= {max(5, args.min_samples)} total, >= {args.min_corrections} corrections, "
            f">= {args.min_accepts} Top-1 accepts to prevent overfitting (keeping prior weights)."
        )
        print("Current prior fusion_weights:", [round(float(x), 3) for x in W_PRIOR])
        return

    tot_w = float(np.sum(weights))
    prior_nll_vec = eval_nll_vec(X_list, y_list, W_PRIOR)
    prior_nll = float(np.sum(weights * prior_nll_vec) / tot_w)
    prior_acc = eval_weighted_acc(X_list, y_list, weights, W_PRIOR)
    print(f"Prior weights: weighted NLL={prior_nll:.4f} | weighted Top-1 acc={prior_acc:.2f}%")

    folds = 5
    fold_bounds = [(f * n // folds, (f + 1) * n // folds) for f in range(folds)]
    best_lam: Optional[float] = None
    best_nll = prior_nll
    best_oof_nll: Optional[np.ndarray] = None
    best_oof_acc = prior_acc

    for lam in LAMBDA_GRID:
        oof_nll = np.zeros(n, dtype=np.float64)
        oof_hit_w = 0.0
        for lo, hi in fold_bounds:
            va_i = list(range(lo, hi))
            tr_i = list(range(0, lo)) + list(range(hi, n))
            wf = fit_newton_offset_ridge(
                [X_list[i] for i in tr_i], [y_list[i] for i in tr_i], weights[tr_i], lam
            )
            for i in va_i:
                s = X_list[i] @ wf
                s -= float(np.max(s))
                oof_nll[i] = math.log(float(np.sum(np.exp(s)))) - float(s[y_list[i]])
                if int(np.argmax(s)) == y_list[i]:
                    oof_hit_w += float(weights[i])
        wl = float(np.sum(weights * oof_nll) / tot_w)
        wa = oof_hit_w / tot_w * 100.0
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
    mean_d = float(np.sum(weights * diff) / tot_w)
    se_d = hajek_paired_se(diff, weights, mean_d)
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
