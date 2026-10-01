"""Zero-shot option-scoring baseline on MMLU with calibration metrics.

Scores the MMLU validation split (used only to fit a post-hoc temperature) and
the test split, then writes metrics, a reliability diagram and per-item
predictions to the output directory.

    uv run python scripts/mmlu_baseline.py --model Qwen/Qwen3.5-4B --out results/mmlu_zeroshot_qwen3.5-4b
    uv run python scripts/mmlu_baseline.py --model Qwen/Qwen3.5-0.8B --device cpu --limit 40 --out results/smoke
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer

from llmlab import calibration as cal
from llmlab.mcq import build_prompt, score_prompts
from llmlab.plots import reliability_figure

K = 4


def load_split(split: str, limit: int | None, seed: int):
    ds = load_dataset("cais/mmlu", "all", split=split)
    if limit and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
    return ds


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--format", choices=["auto", "chat", "plain"], default="auto",
                    help="auto: plain for *-Base models, chat otherwise")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="subsample each split (smoke tests)")
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--check-batch", type=int, default=32,
                    help="re-score this many test items one at a time and compare with batched scores")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chat = {"chat": True, "plain": False}.get(args.format, not args.model.endswith("-Base"))
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).to(args.device).eval()

    splits = {}
    for name in ("validation", "test"):
        ds = load_split(name, args.limit, args.seed)
        prompts = [build_prompt(tok, q, c, s, chat) for q, c, s in zip(ds["question"], ds["choices"], ds["subject"])]
        logits, mass = score_prompts(model, tok, prompts, K, args.batch_size, args.device, desc=name)
        splits[name] = {"logits": logits, "mass": mass, "labels": np.array(ds["answer"]),
                        "subjects": np.array(ds["subject"]), "prompts": prompts}

    # batching sanity check: left padding must not change the scores
    n_chk = min(args.check_batch, len(splits["test"]["prompts"]))
    single = np.concatenate([score_prompts(model, tok, [p], K, 1, args.device, desc="check")[0]
                             for p in splits["test"]["prompts"][:n_chk]]) if n_chk else np.zeros((0, K))
    batched_p = cal.softmax_t(splits["test"]["logits"][:n_chk], 1.0) if n_chk else single
    batch_check = {"n": n_chk, "max_abs_prob_diff": float(np.abs(cal.softmax_t(single, 1.0) - batched_p).max()) if n_chk else None,
                   "argmax_agreement": float((single.argmax(1) == splits["test"]["logits"][:n_chk].argmax(1)).mean()) if n_chk else None}

    val, test = splits["validation"], splits["test"]
    temperature = cal.fit_temperature(val["logits"], val["labels"])
    p_raw = cal.softmax_t(test["logits"], 1.0)
    p_ts = cal.softmax_t(test["logits"], temperature)
    y = test["labels"]

    def report(p):
        m = cal.all_metrics(p, y, args.n_bins)
        m["ci95"] = {name: cal.bootstrap_ci(p, y, fn, args.n_boot, args.seed) for name, fn in {
            "accuracy": cal.accuracy, "brier": cal.brier, "nll": cal.nll,
            "ece_width": lambda a, b: cal.ece(a, b, args.n_bins, "width"),
            "ece_mass": lambda a, b: cal.ece(a, b, args.n_bins, "mass"),
        }.items()}
        return m

    pred_freq = np.bincount(p_raw.argmax(1), minlength=K) / len(y)
    true_freq = np.bincount(y, minlength=K) / len(y)
    metrics = {
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "prompt_format": "chat (thinking disabled, assistant prefilled 'Answer:')" if chat else "plain completion",
        "benchmark": "cais/mmlu all; temperature fit on validation, metrics on test",
        "n_validation": int(len(val["labels"])), "n_test": int(len(y)),
        "binning": f"top-label, {args.n_bins} bins, equal-width on [0,1] and equal-mass (confidence quantiles)",
        "brier_definition": "multiclass sum_k (p_k - y_k)^2, range [0,2]; brier_top = binary Brier of top-label confidence",
        "temperature": temperature,
        "test_raw": report(p_raw),
        "test_temperature_scaled": report(p_ts),
        "letter_mass": {"test_mean": float(test["mass"].mean()), "test_min": float(test["mass"].min()),
                        "test_frac_below_0.5": float((test["mass"] < 0.5).mean())},
        "position_bias": {"predicted_freq_ABCD": pred_freq.round(4).tolist(), "true_freq_ABCD": true_freq.round(4).tolist()},
        "batch_check": batch_check,
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "device": args.device,
                "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
                "dtype": str(dtype)},
        "commit": git_commit(),
        "limit": args.limit,
        "seconds": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "commit.txt").write_text(metrics["commit"] + "\n")
    np.savez_compressed(out / "predictions.npz",
                        **{f"{s}_{k}": v for s in splits for k, v in splits[s].items() if k != "prompts"},
                        temperature=temperature)
    with open(out / "example_prompts.txt", "w") as f:
        f.write("\n\n======\n\n".join(test["prompts"][:2]))
    reliability_figure([(f"{args.model} raw (T=1)", p_raw), (f"temperature-scaled (T={temperature:.2f})", p_ts)],
                       y, out / "reliability.png", args.n_bins)

    r, s = metrics["test_raw"], metrics["test_temperature_scaled"]
    print(json.dumps({"acc": r["accuracy"], "ece_width": r["ece_width"], "ece_mass": r["ece_mass"],
                      "brier": r["brier"], "nll": r["nll"], "auroc": r["auroc"], "T": temperature,
                      "ts_ece_width": s["ece_width"], "ts_brier": s["brier"],
                      "letter_mass": metrics["letter_mass"]["test_mean"], "batch_check": batch_check,
                      "seconds": metrics["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
