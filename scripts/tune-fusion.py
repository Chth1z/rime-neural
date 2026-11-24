#!/usr/bin/env python3
"""
Personal fusion weight tuner for rime-neural (unbiased Newton-Raphson MAP shrinkage).

Statistical anti-overfitting design:
  1. `lua/rime_neural.lua` logs eligible multi-candidate sentence commits at full sampling probability as zero-plaintext
     numeric feature records (`v: 2`, `sample_weight = 1.0`, bounded by 5 MiB auto-rotation)
     to avoid top-1 subsampling while keeping user text in memory. Legacy v1 records remain
     readable (and can be converted in-place to zero-plaintext v2 via `--scrub-legacy-inplace`),
     and legacy 10%-sampled Top-1 logs remain supported via `--top1-sample-rate 0.10`.
  2. Fits exact damped Newton-Raphson Bayesian MAP estimation around `W_PRIOR` with Gaussian
     prior precision `lam` scaled by Kish effective sample size `N_eff` (scale-invariant):
         J(w) = (1 / sum_w) * sum_i w_i * NLL_i(w) + (lam / (2 * N_eff)) * ||w - W_PRIOR||^2
     so prior strength stays fixed in effective sample count and vanishes asymptotically as N_eff -> inf.
     Freezes `w_ctx_lp` at `W_PRIOR[4]` unless both `has_ctx=True` and `has_ctx=False` have >= 20 samples
     so collinear features cannot arbitrarily split weight between `w_logit` and `w_ctx_lp`.
  3. Selects prior precision `lam` by contiguous blocked 5-fold out-of-fold weighted **log-loss
     (NLL)**, scoring each fit at the same three-decimal precision used in YAML.
  4. Only recommends an update when ALL FOUR guards pass:
       - Minimum effective sample guards are met (`min(n, Kish N_eff) >= min_samples`,
         `>= 15` corrections, `>= 30` accepts)
       - Out-of-fold weighted NLL improvement passes a one-sided paired normal approximation
         using the Hajek sandwich SE and a Bonferroni adjustment across the precision grid
       - Out-of-fold weighted Top-1 accuracy does not drop below `W_PRIOR`
       - Fitted (and 3-decimal rounded) weights satisfy domain sanity invariants (`w_logit >= 0.1`,
         `w_rank <= 0.0`, `w_logit + w_ctx_lp >= 0.1`) and differ from `W_PRIOR`.

Usage:
    python scripts/tune-fusion.py
    python scripts/tune-fusion.py --log path/to/feedback.jsonl --schema rime_frost_double_pinyin
    python scripts/tune-fusion.py --scrub-legacy-inplace
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
from statistics import NormalDist
import sys
import tempfile
import time
from collections import Counter
from typing import Any, List, Optional, Tuple

import numpy as np

# General OOD-verified prior weights: [w_logit, w_rank, w_gram, w_char_lp, w_ctx_lp]
W_PRIOR = np.array([1.0, -1.4, 0.05, 0.15, 0.0], dtype=np.float64)
# Prior precision grid (effective prior sample count; scaled by 1 / Kish N_eff)
LAMBDA_GRID = (0.5, 1.5, 5.0, 15.0, 50.0, 150.0, 500.0)
MAX_ABS_FEATURE = 1e6
MAX_SAMPLE_WEIGHT = 1e6
MIN_CTX_STRATUM = 20
PLAINTEXT_KEYS = ("raw_input", "context", "cands", "chosen_text", "shown_top1", "p2c_cand")


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


def _parse_strict_int(raw: Any, default: Optional[int] = None) -> Optional[int]:
    if raw is None or isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if math.isfinite(raw) and raw.is_integer():
            return int(raw)
        return default
    if isinstance(raw, str):
        s = raw.strip()
        if s and s.isascii() and (s.isdigit() or (s[0] in "+-" and s[1:].isdigit())):
            try:
                return int(s)
            except ValueError:
                return default
    return default


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
        if not math.isfinite(fv) or abs(fv) > MAX_ABS_FEATURE:
            return None
        out.append(fv)
    return out


def _parse_sample_weight(raw_sw: Any) -> Optional[float]:
    if raw_sw is None or isinstance(raw_sw, bool):
        return None
    try:
        fsw = float(raw_sw)
    except (TypeError, ValueError):
        return None
    if math.isfinite(fsw) and 0.0 < fsw <= MAX_SAMPLE_WEIGHT:
        return fsw
    return None


def _convert_v1_to_v2(rec: dict) -> Optional[dict]:
    """Convert a legacy v1 record (containing raw text fields) into a zero-plaintext v2 record.
    Preserves absent `sample_weight` (omitting the key) so `--top1-sample-rate` remains effective."""
    cands = rec.get("cands")
    if not isinstance(cands, list) or len(cands) < 2:
        return None
    num_cands = len(cands)
    logits = _parse_finite_float_list(rec.get("logits"), num_cands)
    grams = _parse_finite_float_list(rec.get("gram_deltas"), num_cands, allow_empty=True)
    clps = _parse_finite_float_list(rec.get("char_lps"), num_cands, allow_empty=True)
    if logits is None or grams is None or clps is None:
        return None

    chosen = rec.get("chosen_text") if isinstance(rec.get("chosen_text"), str) else ""
    shown_top1 = rec.get("shown_top1") if isinstance(rec.get("shown_top1"), str) else ""
    p2c_cand = rec.get("p2c_cand") if isinstance(rec.get("p2c_cand"), str) else ""
    ctx = rec.get("context") if isinstance(rec.get("context"), str) else ""
    schema = rec.get("schema") if isinstance(rec.get("schema"), str) else ""
    model = rec.get("model") if isinstance(rec.get("model"), str) else ""

    chosen_idx = cands.index(chosen) if (chosen and chosen in cands) else -1
    p2c_shown = bool(p2c_cand)
    p2c_chosen = bool(p2c_cand and chosen == p2c_cand and chosen_idx < 0)
    if p2c_cand and shown_top1 == p2c_cand and shown_top1 not in cands:
        shown_top1_idx = -2
    elif shown_top1 and shown_top1 in cands:
        shown_top1_idx = cands.index(shown_top1)
    else:
        shown_top1_idx = -1

    chosen_rank = _parse_strict_int(rec.get("chosen_rank"), 0) or 0
    ts = _parse_strict_int(rec.get("ts"), 0) or 0
    sw = _parse_sample_weight(rec.get("sample_weight"))

    reverted = (shown_top1_idx > 0 or shown_top1_idx == -2) and chosen_idx == 0
    out: dict[str, Any] = {
        "v": 2,
        "ts": ts,
        "schema": schema,
        "has_ctx": len(ctx.strip()) >= 2,
        "num_cands": num_cands,
        "logits": logits,
        "gram_deltas": grams,
        "char_lps": clps,
        "shown_top1_idx": shown_top1_idx,
        "chosen_idx": chosen_idx,
        "p2c_shown": p2c_shown,
        "p2c_chosen": p2c_chosen,
        "chosen_rank": chosen_rank,
        "reverted": reverted,
    }
    if sw is not None:
        out["sample_weight"] = sw
    if model:
        out["model"] = model
    if isinstance(rec.get("has_gram"), bool):
        out["has_gram"] = rec["has_gram"]
    return out


def _atomic_replace_with_retry(src: str, dst: pathlib.Path, attempts: int = 5) -> None:
    last_err: Optional[OSError] = None
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_err = exc
            if i + 1 < attempts:
                time.sleep(0.04 * (i + 1))
    if last_err is not None:
        raise last_err


def scrub_legacy_log_files(path: pathlib.Path, include_rotated: bool = True) -> Tuple[int, int]:
    """Atomically rewrite legacy v1 plaintext records into zero-plaintext v2 numeric records."""
    targets: List[pathlib.Path] = []
    if include_rotated:
        rot = pathlib.Path(str(path) + ".1")
        if rot.is_symlink():
            raise OSError(f"Refusing to scrub symlink: {rot}")
        if rot.is_file():
            targets.append(rot)
    if path.is_symlink():
        raise OSError(f"Refusing to scrub symlink: {path}")
    if path.is_file():
        targets.append(path)

    converted_total = 0
    kept_total = 0
    for target in targets:
        out_lines: List[str] = []
        changed = False
        for raw_line in target.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw_line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                changed = True
                continue
            if not isinstance(rec, dict):
                changed = True
                continue
            has_plaintext = any(k in rec for k in PLAINTEXT_KEYS)
            if has_plaintext:
                if "chosen_idx" in rec and isinstance(rec.get("logits"), list):
                    # Numeric outcomes are authoritative even when a stray cands array is present.
                    v2 = {k: v for k, v in rec.items() if k not in PLAINTEXT_KEYS}
                    v2["v"] = 2
                else:
                    v2 = _convert_v1_to_v2(rec)
                if v2 is not None:
                    out_lines.append(json.dumps(v2, ensure_ascii=False, separators=(",", ":")))
                    converted_total += 1
                changed = True
            else:
                out_lines.append(s)
                kept_total += 1
        if changed:
            fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".fb_scrub_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                    if out_lines:
                        fh.write("\n".join(out_lines) + "\n")
                _atomic_replace_with_retry(tmp_name, target)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
    return converted_total, kept_total


def _iter_log_lines(path: pathlib.Path, include_rotated: bool = True) -> Tuple[List[str], int]:
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
    return lines, len(files)


def load_samples(
    path: pathlib.Path,
    top1_sample_rate: float = 1.0,
    include_rotated: bool = True,
    schema_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    has_gram_filter: Optional[bool] = None,
) -> Tuple[List[np.ndarray], List[int], np.ndarray, dict]:
    summary: dict[str, Any] = {
        "files_found": 0,
        "total_lines": 0,
        "legacy_v1": 0,
        "matched": 0,
        "p2c_accepted": 0,
        "disagreed": 0,
        "neural_win": 0,
        "rime_revert": 0,
        "with_ctx": 0,
        "without_ctx": 0,
        "schemas": Counter(),
        "models": Counter(),
        "has_gram": Counter(),
    }
    raw_lines, files_found = _iter_log_lines(path, include_rotated=include_rotated)
    summary["files_found"] = files_found
    if not raw_lines:
        return [], [], np.array([], dtype=np.float64), summary

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    w_list: List[float] = []
    top1_weight = 1.0 / max(1e-3, min(1.0, top1_sample_rate))

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

        rec_schema = rec.get("schema") if isinstance(rec.get("schema"), str) else ""
        rec_model = rec.get("model") if isinstance(rec.get("model"), str) else ""
        rec_has_gram = rec.get("has_gram") if isinstance(rec.get("has_gram"), bool) else None
        if schema_filter and rec_schema != schema_filter:
            continue
        if model_filter and rec_model != model_filter:
            continue
        if has_gram_filter is not None and rec_has_gram is not has_gram_filter:
            continue

        is_v2 = ("chosen_idx" in rec) or (
            isinstance(rec.get("v"), int) and not isinstance(rec.get("v"), bool) and rec["v"] >= 2
        )
        if not is_v2:
            summary["legacy_v1"] += 1
            converted = _convert_v1_to_v2(rec)
            if converted is None:
                continue
            v2 = converted
        else:
            raw_logits = rec.get("logits")
            if not isinstance(raw_logits, list) or len(raw_logits) < 2:
                continue
            num_cands = len(raw_logits)
            logits = _parse_finite_float_list(raw_logits, num_cands)
            grams = _parse_finite_float_list(rec.get("gram_deltas"), num_cands, allow_empty=True)
            clps = _parse_finite_float_list(rec.get("char_lps"), num_cands, allow_empty=True)
            if logits is None or grams is None or clps is None:
                continue
            chosen_idx = _parse_strict_int(rec.get("chosen_idx"), None)
            shown_top1_idx = _parse_strict_int(rec.get("shown_top1_idx"), -1)
            if chosen_idx is None or shown_top1_idx is None:
                continue
            if shown_top1_idx < -2 or shown_top1_idx >= num_cands:
                shown_top1_idx = -1
            v2 = {
                "logits": logits,
                "gram_deltas": grams,
                "char_lps": clps,
                "num_cands": num_cands,
                "chosen_idx": chosen_idx,
                "shown_top1_idx": shown_top1_idx,
                "p2c_chosen": bool(rec.get("p2c_chosen", False)),
                "has_ctx": bool(rec.get("has_ctx", False)),
                "sample_weight": rec.get("sample_weight"),
                "chosen_rank": rec.get("chosen_rank", 0),
            }

        num_cands = v2["num_cands"]
        logits = v2["logits"]
        grams = v2["gram_deltas"]
        clps = v2["char_lps"]
        chosen_idx = v2["chosen_idx"]
        shown_top1_idx = v2["shown_top1_idx"]
        p2c_chosen = bool(v2["p2c_chosen"]) and (chosen_idx < 0)

        if p2c_chosen:
            summary["p2c_accepted"] += 1
            if shown_top1_idx == -2:
                summary["disagreed"] += 1
                summary["neural_win"] += 1
        elif chosen_idx < 0 and shown_top1_idx == -2:
            # P2C was shown as top-1 (disagreeing with Rime #1), and user chose outside Top-K
            summary["disagreed"] += 1

        if chosen_idx < 0 or chosen_idx >= num_cands:
            continue

        summary["matched"] += 1
        if rec_schema:
            summary["schemas"][rec_schema] += 1
        if rec_model:
            summary["models"][rec_model] += 1
        if rec_has_gram is not None:
            summary["has_gram"][rec_has_gram] += 1

        if shown_top1_idx != -1 and shown_top1_idx != 0:
            summary["disagreed"] += 1
            if chosen_idx == shown_top1_idx:
                summary["neural_win"] += 1
            elif chosen_idx == 0:
                summary["rime_revert"] += 1

        y = chosen_idx
        has_ctx_bool = bool(v2["has_ctx"])
        if has_ctx_bool:
            summary["with_ctx"] += 1
        else:
            summary["without_ctx"] += 1
        has_ctx = 1.0 if has_ctx_bool else 0.0
        lp = log_softmax(logits)
        X = np.array(
            [
                [lp[i], math.log(i + 1.0), grams[i], clps[i], lp[i] * has_ctx]
                for i in range(num_cands)
            ],
            dtype=np.float64,
        )

        sw = _parse_sample_weight(v2.get("sample_weight"))
        if sw is None:
            chosen_rank = _parse_strict_int(v2.get("chosen_rank"), 0) or 0
            sw = top1_weight if chosen_rank == 1 else 1.0

        X_list.append(X)
        y_list.append(y)
        w_list.append(sw)

    return X_list, y_list, np.array(w_list, dtype=np.float64), summary


def kish_effective_n(weights: np.ndarray) -> float:
    s1 = float(np.sum(weights))
    s2 = float(np.sum(weights * weights))
    return (s1 * s1 / s2) if (math.isfinite(s1) and math.isfinite(s2) and s2 > 0.0) else 0.0


def eval_penalized_obj(
    X_list: List[np.ndarray],
    y_list: List[int],
    weights: np.ndarray,
    tot_w: float,
    eff_lam: float,
    w_prior: np.ndarray,
    w: np.ndarray,
) -> float:
    nll = 0.0
    for X, y, wt in zip(X_list, y_list, weights):
        s = X @ w
        s -= float(np.max(s))
        nll += (float(wt) / tot_w) * (math.log(float(np.sum(np.exp(s)))) - float(s[y]))
    delta = w - w_prior
    return nll + 0.5 * eff_lam * float(delta @ delta)


def fit_newton_offset_ridge(
    X_list: List[np.ndarray],
    y_list: List[int],
    weights: np.ndarray,
    lam: float,
    w_prior: np.ndarray = W_PRIOR,
    max_iters: int = 20,
    freeze_ctx_lp: bool = False,
) -> np.ndarray:
    """Damped Newton-Raphson solver with Armijo backtracking line search for scale-invariant
    Bayesian MAP weighted listwise NLL + (lam / (2 * Kish_N_eff)) * ||w - w_prior||^2."""
    d = len(w_prior)
    w = w_prior.copy()
    tot_w = float(np.sum(weights))
    n_eff = kish_effective_n(weights)
    if not math.isfinite(tot_w) or tot_w <= 0.0 or not math.isfinite(n_eff) or n_eff <= 0.0:
        return w
    eff_lam = lam / n_eff
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
            if not math.isfinite(sum_exp) or sum_exp <= 0.0:
                return w
            p = exp_s / sum_exp
            c = float(wt) / tot_w
            obj_cur += c * (math.log(sum_exp) - float(s[y]))
            diff = p.copy()
            diff[y] -= 1.0
            grad += c * (X.T @ diff)
            mu = p @ X
            Xc = X - mu
            hess += c * ((Xc * p[:, None]).T @ Xc)
        if freeze_ctx_lp and d > 4:
            grad[4] = 0.0
            hess[4, :] = 0.0
            hess[:, 4] = 0.0
            hess[4, 4] = 1.0
        if not (np.isfinite(grad).all() and np.isfinite(hess).all()):
            break
        try:
            step = np.linalg.solve(hess + 1e-8 * eye, grad)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(step).all():
            break
        dir_deriv = -float(grad @ step)
        if not math.isfinite(dir_deriv) or dir_deriv >= 0.0:
            break
        alpha = 1.0
        accepted = False
        for _ in range(10):
            w_cand = w - alpha * step
            obj_cand = eval_penalized_obj(X_list, y_list, weights, tot_w, eff_lam, w_prior, w_cand)
            if math.isfinite(obj_cand) and obj_cand <= obj_cur + 1e-4 * alpha * dir_deriv:
                w = w_cand
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        if alpha * float(np.linalg.norm(step)) < 1e-7:
            break
    return w


def weights_pass_domain_sanity(w: np.ndarray) -> Tuple[bool, str]:
    """Reject pathological weight vectors caused by selection-biased logs (e.g. inverted rank prior)."""
    if not np.isfinite(w).all():
        return False, "non-finite weight value"
    w_logit, w_rank, _w_gram, _w_char_lp, w_ctx_lp = [float(x) for x in w]
    if w_logit < 0.1:
        return False, f"w_logit={w_logit:.3f} < 0.1 (neural logit weight must stay positive)"
    if w_rank > 0.0:
        return False, f"w_rank={w_rank:+.3f} > 0 (positive rank weight would invert Rime rank order)"
    if (w_logit + w_ctx_lp) < 0.1:
        return False, f"w_logit + w_ctx_lp={w_logit + w_ctx_lp:.3f} < 0.1"
    return True, ""


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
    if not math.isfinite(tot_w) or tot_w <= 0.0:
        return 0.0
    hits = sum(
        float(wt)
        for X, y, wt in zip(X_list, y_list, weights)
        if int(np.argmax(X @ w)) == y
    )
    return hits / tot_w * 100.0


def hajek_paired_se(diff: np.ndarray, weights: np.ndarray, mean_d: float) -> float:
    """Heteroscedasticity-robust Hajek sandwich SE for a weighted ratio mean."""
    n = len(diff)
    s1 = float(np.sum(weights))
    if n < 2 or not math.isfinite(s1) or s1 <= 0.0:
        return 0.0
    var_hat = (n / (n - 1.0)) * float(np.sum((weights * (diff - mean_d)) ** 2)) / (s1 * s1)
    return math.sqrt(max(0.0, var_hat))


def paired_z_score(mean_d: float, se_d: float) -> float:
    if se_d > 1e-12:
        return mean_d / se_d
    if mean_d > 1e-12:
        return math.inf
    return 0.0


def format_weights(w: np.ndarray) -> List[float]:
    out: List[float] = []
    for x in w:
        rv = round(float(x), 3)
        out.append(0.0 if rv == 0.0 else rv)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Unbiased MAP personal fusion weight tuner for rime-neural")
    ap.add_argument(
        "--log",
        help="Path to local feedback.jsonl (defaults to RimeUserDir/%%APPDATA%%\\Rime\\neural\\feedback.jsonl)",
    )
    ap.add_argument("--min-samples", type=int, default=100, help="Minimum matched records & Kish N_eff")
    ap.add_argument("--min-corrections", type=int, default=15, help="Minimum prior-miss records")
    ap.add_argument("--min-accepts", type=int, default=30, help="Minimum prior-hit records")
    ap.add_argument(
        "--top1-sample-rate",
        type=float,
        default=1.0,
        help="Sampling rate in (0, 1] of Top-1 accepts for legacy logs without sample_weight (default 1.0)",
    )
    ap.add_argument("--schema", default="", help="Filter feedback records to a single Rime schema_id")
    ap.add_argument("--model", default="", help="Filter feedback records to a single ONNX model filename")
    ap.add_argument(
        "--has-gram",
        choices=("any", "true", "false"),
        default="any",
        help="Filter records by whether a .gram language model was active (default: any)",
    )
    ap.add_argument(
        "--no-rotated",
        action="store_true",
        help="Do not include rotated feedback.jsonl.1 alongside feedback.jsonl",
    )
    ap.add_argument(
        "--scrub-legacy-inplace",
        action="store_true",
        help="Atomically convert legacy v1 plaintext records in feedback.jsonl(.1) to zero-plaintext v2 and exit",
    )
    args = ap.parse_args()

    if not (0.0 < args.top1_sample_rate <= 1.0):
        print("Error: --top1-sample-rate must be in (0, 1].", file=sys.stderr)
        sys.exit(2)
    if args.min_corrections < 0 or args.min_accepts < 0:
        print("Error: --min-corrections and --min-accepts must be >= 0.", file=sys.stderr)
        sys.exit(2)

    log_path = pathlib.Path(args.log or default_log_path())
    if args.scrub_legacy_inplace:
        has_primary = log_path.is_file() or log_path.is_symlink()
        has_rot = (not args.no_rotated) and (
            pathlib.Path(str(log_path) + ".1").is_file()
            or pathlib.Path(str(log_path) + ".1").is_symlink()
        )
        if not has_primary and not has_rot:
            print(f"No feedback log found at {log_path}.")
            return
        converted, kept = scrub_legacy_log_files(log_path, include_rotated=not args.no_rotated)
        print(f"Scrubbed {log_path}: converted {converted} legacy v1 record(s) to zero-plaintext v2 (kept {kept} v2).")
        return

    gram_filter: Optional[bool] = None
    if args.has_gram == "true":
        gram_filter = True
    elif args.has_gram == "false":
        gram_filter = False

    X_list, y_list, weights, summary = load_samples(
        log_path,
        args.top1_sample_rate,
        include_rotated=not args.no_rotated,
        schema_filter=args.schema.strip() or None,
        model_filter=args.model.strip() or None,
        has_gram_filter=gram_filter,
    )
    if summary["files_found"] == 0:
        print(f"No feedback log file found at {log_path} (enable rime_neural/log_feedback: true first).")
        print("Current prior fusion_weights:", format_weights(W_PRIOR))
        return

    n = len(X_list)
    n_prior_hits = sum(1 for X, y in zip(X_list, y_list) if int(np.argmax(X @ W_PRIOR)) == y)
    n_prior_misses = n - n_prior_hits
    n_eff = kish_effective_n(weights)

    print(
        f"Loaded {n}/{summary['total_lines']} matched records from {log_path} "
        f"(prior_misses={n_prior_misses}, prior_hits={n_prior_hits}, Kish N_eff={n_eff:.1f})"
    )
    print(
        f"Rerank breakdown: disagreements={summary['disagreed']} "
        f"(neural_win={summary['neural_win']}, rime_revert={summary['rime_revert']}), "
        f"p2c_accepted={summary['p2c_accepted']}"
    )
    if summary["legacy_v1"] > 0:
        print(
            f"Note: {summary['legacy_v1']} legacy v1 record(s) contain plaintext fields; "
            "run with --scrub-legacy-inplace to convert them to zero-plaintext v2."
        )
    if len(summary["schemas"]) > 1 and not args.schema:
        print(f"Note: multiple schemas pooled ({dict(summary['schemas'])}); pass --schema <id> to tune one.")
    if len(summary["models"]) > 1 and not args.model:
        print(f"Note: multiple models pooled ({dict(summary['models'])}); pass --model <file> to isolate one.")
    if len(summary["has_gram"]) > 1 and gram_filter is None:
        print(
            f"Note: mixed .gram states pooled ({dict(summary['has_gram'])}); "
            "pass --has-gram true|false to isolate one."
        )

    min_req = max(5, args.min_samples)
    tot_w = float(np.sum(weights))
    if (
        n < min_req
        or n_eff < min_req
        or not math.isfinite(tot_w)
        or tot_w <= 0.0
        or n_prior_misses < args.min_corrections
        or n_prior_hits < args.min_accepts
    ):
        print(
            f"Safety guard: need >= {min_req} total & Kish N_eff, >= {args.min_corrections} prior misses, "
            f">= {args.min_accepts} prior hits to prevent overfitting (keeping prior weights)."
        )
        print("Current prior fusion_weights:", format_weights(W_PRIOR))
        return

    freeze_ctx = summary["with_ctx"] < MIN_CTX_STRATUM or summary["without_ctx"] < MIN_CTX_STRATUM
    if freeze_ctx:
        print(
            f"Note: freezing w_ctx_lp at {W_PRIOR[4]:.3f} "
            f"(with_ctx={summary['with_ctx']}, without_ctx={summary['without_ctx']}, need >= {MIN_CTX_STRATUM} each)."
        )

    prior_nll_vec = eval_nll_vec(X_list, y_list, W_PRIOR)
    prior_nll = float(np.sum(weights * prior_nll_vec) / tot_w)
    prior_acc = eval_weighted_acc(X_list, y_list, weights, W_PRIOR)
    print(f"Prior weights: weighted NLL={prior_nll:.4f} | weighted Top-1 acc={prior_acc:.2f}%")

    folds = 5
    train_folds = []
    for f in range(folds):
        lo, hi = f * n // folds, (f + 1) * n // folds
        train_x = X_list[:lo] + X_list[hi:]
        # With >= 2 candidates the log-softmax column cannot be identically zero.
        n_ctx = sum(bool(X[:, 4].any()) for X in train_x)
        freeze_fold = min(n_ctx, len(train_x) - n_ctx) < MIN_CTX_STRATUM
        train_y = y_list[:lo] + y_list[hi:]
        train_weights = np.concatenate((weights[:lo], weights[hi:]))
        train_folds.append((lo, hi, train_x, train_y, train_weights, freeze_fold))
    z_critical = NormalDist().inv_cdf(1.0 - 0.05 / len(LAMBDA_GRID))
    best_raw: Optional[dict[str, Any]] = None
    best_valid: Optional[dict[str, Any]] = None

    for lam in LAMBDA_GRID:
        oof_nll = np.zeros(n, dtype=np.float64)
        oof_hit_w = 0.0
        for lo, hi, train_x, train_y, train_weights, freeze_fold in train_folds:
            wf = fit_newton_offset_ridge(
                train_x, train_y, train_weights, lam, freeze_ctx_lp=freeze_fold,
            )
            wf = np.array(format_weights(wf))
            for i in range(lo, hi):
                s = X_list[i] @ wf
                s -= float(np.max(s))
                oof_nll[i] = math.log(float(np.sum(np.exp(s)))) - float(s[y_list[i]])
                if int(np.argmax(s)) == y_list[i]:
                    oof_hit_w += float(weights[i])
        wl = float(np.sum(weights * oof_nll) / tot_w)
        wa = oof_hit_w / tot_w * 100.0
        print(f"  lambda={lam:5.1f} -> 5-fold NLL={wl:.4f} | Top-1 acc={wa:.2f}%")

        if wl < prior_nll - 1e-6:
            diff = prior_nll_vec - oof_nll
            mean_d = float(np.sum(weights * diff) / tot_w)
            se_d = hajek_paired_se(diff, weights, mean_d)
            z_score = paired_z_score(mean_d, se_d)
            cand_info = {
                "lam": lam,
                "nll": wl,
                "acc": wa,
                "mean_d": mean_d,
                "se_d": se_d,
                "z": z_score,
            }
            if best_raw is None or wl < best_raw["nll"] - 1e-9:
                best_raw = cand_info
            if z_score >= z_critical and wa >= prior_acc - 1e-6:
                w_cand = fit_newton_offset_ridge(
                    X_list, y_list, weights, lam, freeze_ctx_lp=freeze_ctx
                )
                w_fmt_cand = format_weights(w_cand)
                sane_raw, _ = weights_pass_domain_sanity(w_cand)
                sane_fmt, _ = weights_pass_domain_sanity(np.array(w_fmt_cand, dtype=np.float64))
                if sane_raw and sane_fmt and w_fmt_cand != format_weights(W_PRIOR) and (
                    best_valid is None or wl < best_valid["nll"] - 1e-9
                ):
                    best_valid = {**cand_info, "w_fmt": w_fmt_cand}

    if best_raw is None:
        print("\nPrior weights already achieve the lowest 5-fold log-loss; no update needed.")
        return

    if best_valid is None:
        print(
            f"\nBest lambda={best_raw['lam']}: delta_NLL={best_raw['mean_d']:+.4f} +/- {best_raw['se_d']:.4f} "
            f"(z={best_raw['z']:.2f}, acc {prior_acc:.2f}% -> {best_raw['acc']:.2f}%)"
        )
        print("No update passed the grid-adjusted NLL, Top-1 accuracy and weight guards; keeping prior weights.")
        return

    w_fmt = best_valid["w_fmt"]
    print(
        f"\nBest lambda={best_valid['lam']}: delta_NLL={best_valid['mean_d']:+.4f} +/- {best_valid['se_d']:.4f} "
        f"(z={best_valid['z']:.2f}, acc {prior_acc:.2f}% -> {best_valid['acc']:.2f}%)"
    )
    print(f"Passed the grid-adjusted one-sided NLL test (z={best_valid['z']:.2f} >= {z_critical:.3f}).")
    print("Paste into rime_neural/fusion_weights in your schema custom yaml:")
    print(f"  fusion_weights: {w_fmt}")


if __name__ == "__main__":
    main()
