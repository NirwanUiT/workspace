# llm-lab

Experiments in building, fine-tuning and calibrating LLMs. The current thread is a
Jev-style "System One" decision model: an open LLM that returns calibrated
probabilities over a fixed set of options instead of free text. The background
reading is in [reports/](reports/) and [research_notes/](research_notes/).

## Setup

Python 3.12 environment managed by [uv](https://docs.astral.sh/uv/), with torch
built for CUDA 12.8 (matches the vs-c1/vs-c2 driver). The environment lives in
`.venv/` here, on local disk; never on `/mnt`, which is mounted noexec.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once, installs to ~/.local/bin
cd ~/workspace/llm-lab
uv sync                                           # creates .venv from uv.lock
uv run pytest -q                                  # calibration metric tests (CPU)
```

## Layout

| Path | What |
|---|---|
| `src/llmlab/calibration.py` | Accuracy, NLL, Brier, top-label ECE (equal-width and equal-mass bins), AUROC, bootstrap CIs, temperature scaling |
| `src/llmlab/mcq.py` | Zero-shot option scoring: one forward pass, softmax over the answer-letter logits |
| `src/llmlab/plots.py` | Reliability diagrams |
| `scripts/mmlu_baseline.py` | MMLU baseline: scores validation and test, fits a temperature on validation, reports test metrics |
| `scripts/run_mmlu_gpu.sh` | Runs the baseline for Qwen3.5-4B and Qwen3.5-4B-Base on one free GPU, with activity-log lines |
| `scripts/check_batching.py` | Separates bf16 rounding from left-padding effects (fp32 vs bf16, single vs batched) |
| `results/<run>/` | `metrics.json`, `reliability.png`, `commit.txt`, `example_prompts.txt` (committed); `predictions.npz` (per-item, not committed) |

## Running the MMLU baseline

GPUs are shared. Take one only through the cluster helper, log the run, and use tmux:

```bash
export CUDA_VISIBLE_DEVICES=$(bash /mnt/nas1/nba055-2/idea_1/Nanograph/cluster/freegpu.sh 20) || exit 1
LOG=~/workspace/llm-lab/logs/mmlu_qwen3.5-4b.log
echo "$(date -Is) START claude-code llm-mmlu-baseline host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES log=$LOG" \
  >> /mnt/nas1/nba055-2/idea_1/Nanograph/cluster/activity.log
uv run python scripts/mmlu_baseline.py --model Qwen/Qwen3.5-4B --out results/mmlu_zeroshot_qwen3.5-4b > $LOG 2>&1
echo "$(date -Is) END claude-code llm-mmlu-baseline host=$(hostname) exit=$?" \
  >> /mnt/nas1/nba055-2/idea_1/Nanograph/cluster/activity.log
```

CPU smoke test (small model, 40 items per split, a few minutes):

```bash
CUDA_VISIBLE_DEVICES= uv run python scripts/mmlu_baseline.py --model Qwen/Qwen3.5-0.8B \
  --device cpu --limit 40 --batch-size 8 --check-batch 8 --out results/smoke_cpu
```

## Method notes

- **Scoring.** The prompt ends where the answer letter goes. For chat models it uses the
  chat template with thinking disabled and the assistant turn pre-filled with `Answer:`.
  For `*-Base` models it uses the plain MMLU completion format. Each option's logit is the
  logsumexp of the `X` and ` X` token log-probs. `letter_mass` in `metrics.json` reports how
  much of the full next-token distribution lands on the letter tokens.
- **Calibration.** Temperature is fitted on MMLU `validation` (1,531 items) by minimising
  NLL; all reported metrics are on `test` (14,042 items). ECE is top-label with 15 bins,
  reported for both equal-width and equal-mass binning, with 95% bootstrap CIs.
- **Batch check.** The script re-scores a few test items one at a time and reports the
  largest probability difference against the batched (left-padded) scores.

## Results so far

MMLU test (14,042 items), zero-shot option scoring, 15 bins, 95% bootstrap CIs in each `metrics.json`.

| Model | Acc | ECE raw (width / mass) | Brier raw | AUROC | T (fit on val) | ECE after T (width / mass) |
|---|---|---|---|---|---|---|
| Qwen3.5-4B (chat, thinking off) | 0.739 | 0.088 / 0.088 | 0.366 | 0.835 | 1.62 | 0.010 / 0.012 |
| Qwen3.5-4B-Base (plain) | 0.730 | 0.011 / 0.012 | 0.367 | 0.827 | 1.08 | 0.012 / 0.016 |

Post-training makes the model overconfident while leaving accuracy and ranking (AUROC) about
the same; one temperature fitted on 1,531 validation items removes it. In fp32, left-padded
batches match single-item scoring (max prob diff 0.001); bf16 adds up to ~0.05 per-item noise.

## Data rules

WP4 data may be used here, read from where it lives in `~/workspace/WP4`. Never copy raw
WP4 data into this repo, and commit only aggregate results derived from it.
