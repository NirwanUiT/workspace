"""Separate bf16 rounding from left-padding effects in batched option scoring.

Scores the same MMLU test items one at a time and in left-padded batches, in
bf16 and in fp32. If fp32 batched matches fp32 single, padding is handled
correctly and the bf16 differences are numerical noise.

    uv run python scripts/check_batching.py --model Qwen/Qwen3.5-4B --n 64 --out results/mmlu_zeroshot_qwen3.5-4b/batch_check.json
"""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from llmlab.calibration import softmax_t
from llmlab.mcq import build_prompt, score_prompts
from mmlu_baseline import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_split("test", None, 0).select(range(args.n))
    chat = not args.model.endswith("-Base")
    prompts = [build_prompt(tok, q, c, s, chat) for q, c, s in zip(ds["question"], ds["choices"], ds["subject"])]

    probs = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype).to("cuda").eval()
        single = np.concatenate([score_prompts(model, tok, [p], 4, 1, "cuda", desc=f"{name} single")[0] for p in prompts])
        batched = score_prompts(model, tok, prompts, 4, args.batch_size, "cuda", desc=f"{name} batched")[0]
        probs[f"{name}_single"], probs[f"{name}_batched"] = softmax_t(single, 1.0), softmax_t(batched, 1.0)
        del model
        torch.cuda.empty_cache()

    def cmp(a, b):
        d = np.abs(probs[a] - probs[b])
        return {"max_abs_prob_diff": float(d.max()), "mean_abs_prob_diff": float(d.mean()),
                "argmax_agreement": float((probs[a].argmax(1) == probs[b].argmax(1)).mean())}

    result = {"model": args.model, "n": args.n, "batch_size": args.batch_size,
              "padding_effect_fp32 (fp32 single vs fp32 batched)": cmp("fp32_single", "fp32_batched"),
              "padding_plus_rounding_bf16 (bf16 single vs bf16 batched)": cmp("bf16_single", "bf16_batched"),
              "precision_effect (bf16 single vs fp32 single)": cmp("bf16_single", "fp32_single")}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
