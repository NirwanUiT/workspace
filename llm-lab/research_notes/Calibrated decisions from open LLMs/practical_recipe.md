# Practical recipe: calibrated LLM-based decision model on one A100 40GB (as of 2026-10-01)

## Which current open-weight models fit a single A100 40GB as bases?

### Takeaway
Qwen is the strongest-documented family for this budget: the Qwen3.5 small dense models (0.8B/2B/4B/9B, Mar 2026) are the most practical fine-tuning bases on 40GB. The 27B dense models (Qwen3.6-27B, Qwen3.8-27B) and the 35B-A3B MoE fit for inference (FP8/4-bit) but not comfortably for training. Qwen3.6 has no small sizes, and the only open "Qwen3.8" models are 27B and a 2.4T MoE. Gemma 4 (Apache 2.0) is the main alternative: E2B/E4B/12B fit for fine-tuning, while 26B-A4B/31B are inference-only on 40GB.

### Cited Findings
- **Qwen3.5 small series**: 0.8B, 2B, 4B and 9B dense models, released on HF Hub and ModelScope on 2026-03-02 under Apache-2.0. Qwen3.5 also has 27B dense, 35B-A3B MoE, 122B-A10B and 397B-A17B (Feb 2026). — [QwenLM/Qwen3.8 GitHub](https://github.com/QwenLM/Qwen3.8); [Qwen on X](https://x.com/Alibaba_Qwen/status/2028460046510965160)
- The Qwen3.5 small models are post-trained dense models with a vision encoder (text and image input). They support thinking and non-thinking modes, a native context of 262,144 tokens, 201 languages, and Apache 2.0. — [The Rundown (secondary)](https://www.therundown.ai/tools/qwen3-5-small); [Qwen on X](https://x.com/Alibaba_Qwen/status/2028460046510965160)
- **Qwen3.6**: the open weights are only Qwen3.6-35B-A3B (MoE, 35B total / 3B active, 2026-04-16) and Qwen3.6-27B (dense, 2026-04-22), plus FP8 variants. The HF collection has no 0.6B–14B Qwen3.6 models. — [HF Qwen3.6 collection](https://huggingface.co/collections/Qwen/qwen36); [QwenLM/Qwen3.8 GitHub](https://github.com/QwenLM/Qwen3.8)
- **Qwen3.6-27B model card**:
  - Apache 2.0, with a vision encoder.
  - Hybrid architecture: "16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))".
  - Native context of 262,144 tokens, extensible to about 1,010,000 with YaRN.
  - Thinking mode is on by default (`<think>...</think>`) and is turned off with `enable_thinking: False`.
  - Recommended serving: vLLM ≥0.19.0 or SGLang ≥0.5.10. — [HF model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- **Qwen3.8**:
  - Open-weight releases are Qwen3.8-2.4T-A95B (2026-08-12) and Qwen3.8-27B (2026-08-14), both Apache-2.0 with 262,144-token context. — [QwenLM/Qwen3.8 GitHub](https://github.com/QwenLM/Qwen3.8)
  - Qwen3.8-Max was previewed on 2026-07-19 and released via API on 2026-08-03. — [Yotta Labs (secondary)](https://www.yottalabs.ai/post/qwen-3-8-max-release-date-specs-how-to-access-2026); [Qwen blog](https://qwen.ai/blog?id=qwen3.8)
  - Secondary sources describe a "Qwen3.8-Flash-Next" (125B MoE, 6B active) scheduled for 2026-08-26. I did not confirm this against a primary source. — [BuildFastWithAI (secondary)](https://www.buildfastwithai.com/blogs/qwen3-8-flash-next-preview)
- **Gemma 4**:
  - Released 2026-03-31 in E2B, E4B, 26B-A4B (MoE) and 31B sizes.
  - Multi-token-prediction (MTP) variants followed on 2026-04-16, and Gemma 4 12B Unified (text, audio and image input, 256K context) on 2026-06-03. — [Gemma releases page](https://ai.google.dev/gemma/docs/releases)
  - E2B has about 2.3B effective parameters and E4B about 4.5B, both with 128K context. 31B has about 30.7B parameters and 256K context.
  - The licence is Apache 2.0, a change from the earlier Gemma licence. — [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/); [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4); the parameter counts and licence are as summarised by search results and secondary reviews ([webscraft](https://webscraft.org/blog/gemma-4-povniy-oglyad-rozmiri-litsenziya-porivnyannya-z-gemma-3?lang=en)).

### Inferences
- The "Qwen 3.6 / Qwen3.8 small sizes (0.6B–14B)" in the brief do not exist as far as primary sources show. 0.6B/1.7B/4B/8B/14B were Qwen3 sizes (2025). The current small Qwen line is **Qwen3.5-0.8B/2B/4B/9B**. Use Qwen3.5-4B or -9B as the main fine-tuning base, Qwen3.5-0.8B/2B for fast ablations, and Qwen3.6-27B or Qwen3.8-27B for zero/few-shot scoring only.
- Every Qwen3.5+ model is a vision-language checkpoint with a hybrid Gated-DeltaNet architecture. Generic recipes written for pure text decoders (e.g. LoRA target modules, `AutoModelForSequenceClassification` heads) may need adjusting. Check that `AutoModelForSequenceClassification` supports each architecture before committing.
- Turn thinking mode off for calibration scoring (`enable_thinking=False`). Then the label-token probability is read straight after the prompt and is not conditioned on a sampled reasoning trace.
- Gemma 4 E4B and 12B are a good second family for robustness checks across model families. Both are Apache 2.0 like Qwen, so licensing does not limit publication.

### Gaps
- I did not verify the current Llama, Phi or SmolLM releases (sizes, licences, dates) against primary sources in this pass. The writer should not state their versions without checking.
- I did not confirm Qwen3.8-27B architecture details or whether a Qwen3.8 small series is planned.
- Gemma 4 effective parameter counts and context lengths come from search summaries of the model card and blog, not a direct fetch of the card.

## Tooling: transformers/PEFT/TRL/Unsloth/bitsandbytes/vLLM, versions and Python/CUDA requirements

### Takeaway
Python 3.8 cannot run the current stack. TRL 1.x needs Python ≥3.10, vLLM supports 3.10–3.13, and Qwen3.5 needs transformers v5. Create a fresh `uv` or conda environment with Python 3.11 or 3.12. Check the server's NVIDIA driver before choosing the CUDA wheel: current vLLM wheels target CUDA 12.9 or 13.0.

### Cited Findings
- **TRL**:
  - Latest version is 1.14.1 (2026-09-29), requires Python ≥3.10, Apache-2.0.
  - Trainers include SFTTrainer, GRPOTrainer, DPOTrainer, KTOTrainer and RewardTrainer. — [PyPI trl](https://pypi.org/project/trl/)
  - TRL v1.0 (around April 2026) marked the move to a stable API with a unified CLI and config system. — [MarkTechPost (secondary)](https://www.marktechpost.com/2026/04/01/hugging-face-releases-trl-v1-0-a-unified-post-training-stack-for-sft-reward-modeling-dpo-and-grpo-workflows/)
- **GRPOTrainer custom rewards**: GRPOTrainer accepts custom Python reward functions (sync or async) via `reward_funcs`, as one function or a list; async rewards run concurrently. It can use vLLM for rollouts. This is the hook for a Brier or log-score reward on a verbalised probability. — [TRL GRPO docs](https://huggingface.co/docs/trl/grpo_trainer); [TRL GitHub doc](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md)
- **vLLM**:
  - vLLM 0.30.0 requires Python ≥3.10, <3.15.
  - Default binaries are built with CUDA 12.9, with alternative builds for 12.8 and 13.0. PyPI ships CUDA 13.0 wheels for v0.30.0. — [vLLM GPU install docs](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/); [PyPI vllm](https://pypi.org/project/vllm/)
- **Qwen serving**: the Qwen3.6-27B card asks for vLLM ≥0.19.0 or SGLang ≥0.5.10. — [HF model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- **Unsloth with Qwen3.5**:
  - "Please use `transformers v5` for Qwen3.5. Older versions will not work."
  - Unsloth GRPO works for Qwen3.5 only "if you disable fast vLLM inference and use Unsloth inference instead" (`fast_inference=False`). — [Unsloth Qwen3.5 fine-tune guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune.md)
- **MAPIE v1**: the API was redesigned into per-technique classes (e.g. `SplitConformalRegressor`, `CrossConformalClassifier`) that work with any scikit-learn-compatible estimator. The current docs are at 1.2.0. — [MAPIE v1 release notes](https://mapie.readthedocs.io/en/stable/v1_release_notes.html)
- **netcal**: a Python 3 library for measuring and mitigating miscalibration (e.g. ECE, reliability diagrams, temperature/histogram/isotonic/beta scaling). — [PyPI netcal](https://pypi.org/project/netcal/)

### Inferences
- Suggested environment:
  - Install user-space `uv`, then run `uv venv -p 3.12`.
  - Install `torch` built for the CUDA version the driver supports (`nvidia-smi` shows the maximum CUDA version).
  - Install `transformers>=5`, `peft`, `trl>=1.14`, `accelerate`, `bitsandbytes`, and optionally `unsloth`.
  - Put **vLLM in a separate venv**. It pins its own torch version, and mixing it with the training environment is a common source of breakage.
- If the server driver is older than what CUDA 12.8+/13.0 wheels need, vLLM pip wheels will fail. In that case use HF transformers batched forward passes for scoring. For classification-style calibration you only need one forward pass per example to read label logits, so vLLM is not essential.
- On a shared server, set `CUDA_VISIBLE_DEVICES` and cap vLLM `gpu_memory_utilization`. Other users' processes make the 40GB budget smaller in practice.

### Gaps
- I did not fetch current exact versions of transformers, PEFT, bitsandbytes or Unsloth, or their Python minimums. transformers v5 almost certainly requires Python ≥3.10, but this was not confirmed.
- I did not check the minimum NVIDIA driver version for the CUDA 12.9/13.0 vLLM wheels against NVIDIA docs.
- I did not confirm that `AutoModelForSequenceClassification` supports Qwen3.5 or Gemma 4 in transformers v5.

## Memory and throughput budgets for LoRA/QLoRA on 40GB; typical hyperparameters

### Takeaway
On 40GB, bf16 LoRA works up to about the 9B class: Unsloth reports 22GB for Qwen3.5-9B. 27B dense bf16 LoRA needs about 56GB and the 35B-A3B MoE about 74GB, so neither fits. Unsloth explicitly advises against QLoRA (4-bit) on Qwen3.5 dense or MoE models. That makes "27B via QLoRA" a poor plan for Qwen3.5-era models; treat 27B as inference-only (FP8/4-bit scoring) on this GPU.

### Cited Findings
- **Unsloth bf16 LoRA VRAM for Qwen3.5**: "0.8B: 3GB • 2B: 5GB • 4B: 10GB • 9B: 22GB • 27B: 56GB". Qwen3.5-35B-A3B bf16 LoRA "works on 74GB VRAM". — [Unsloth Qwen3.5 fine-tune guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune.md)
- "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense, due to higher than normal quantization differences." — [Unsloth Qwen3.5 fine-tune guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune.md)
- **Unsloth example hyperparameters**: r=16, alpha=16, per-device batch size 1 with gradient accumulation 4, max sequence length 2048 as a starting point, `adamw_8bit`. Target modules: q/k/v/o_proj and gate/up/down_proj. — [Unsloth Qwen3.5 fine-tune guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune.md)

### Inferences
These are rough estimates, not measured.

- **Model size vs 40GB**:
  - 1.5–2B: bf16 LoRA or even full fine-tuning fits easily.
  - 4B: bf16 LoRA at about 10GB leaves room for batch 8–16 at short sequence lengths.
  - 8–9B: bf16 LoRA at about 22GB; use gradient checkpointing and batch 1–4 with accumulation.
  - 14B-class (Qwen3 legacy 14B, Gemma 4 12B): bf16 LoRA is likely around 30–35GB and tight. QLoRA is feasible for non-Qwen3.5 architectures.
  - 27B: QLoRA weights alone take about 14–16GB, so it could fit in 40GB, but Unsloth's warning applies to Qwen3.5/3.6-style models. Prefer zero/few-shot scoring at 27B.
- **Sequence length**: serialised tabular rows are short (under 512 tokens), so activation memory is small. That makes batch size, not sequence length, the main lever.
- **Hyperparameters for classification-head or label-token LoRA**:
  - r = 8–32 and alpha = r to 2r.
  - Learning rate around 1e-4 to 2e-4 for LoRA.
  - 1–3 epochs.
  - Keep a held-out calibration split for temperature scaling after fine-tuning.
  
  These are general community defaults, not taken from a fetched source except r=16/alpha=16 above.
- **Throughput**: I have no measured tokens/s. For a scoring-only workload (one forward pass, reading next-token logits for "A/B/C"), even 9B models score thousands of short prompts per minute on an A100 with batched HF inference. This is an inference, not a benchmark.

### Gaps
- I found no primary-source measured VRAM numbers for Gemma 4, or for QLoRA at 14B/27B on current architectures.
- I found no measured A100 throughput figures for training or scoring.

## Datasets and libraries to measure calibration of decision outputs

### Takeaway
Use multiple-choice QA sets, where per-option label-token probabilities give a clean probability vector, plus a real tabular task. Measure ECE (with a fixed binning scheme, reported alongside the classwise variant), Brier score, NLL, and reliability diagrams. Add conformal prediction sets (MAPIE or a hand-rolled split-conformal procedure) for coverage guarantees.

### Cited Findings
- MAPIE v1 provides split- and cross-conformal classifiers (`CrossConformalClassifier` etc.) for any scikit-learn-compatible estimator. To apply it to an LLM, wrap the LLM's probability output in a sklearn-style `predict_proba` estimator. — [MAPIE v1 release notes](https://mapie.readthedocs.io/en/stable/v1_release_notes.html); [MAPIE paper](https://arxiv.org/pdf/2207.12274)
- netcal measures miscalibration and provides post-hoc calibration methods. — [PyPI netcal](https://pypi.org/project/netcal/)

### Inferences
- **Benchmarks**:
  - MMLU-Pro (10 options, harder and less contaminated than MMLU).
  - ARC-Challenge and CommonsenseQA (multiple-choice).
  - BoolQ (binary).
  - TriviaQA (open-ended; needs a verbalised-confidence or P(True) protocol rather than option logits).
  
  These are standard; I did not re-fetch their HF pages in this pass.
- **Metrics stack**: `torchmetrics.classification.MulticlassCalibrationError` (ECE/MCE), `sklearn.metrics.brier_score_loss` / `log_loss`, and netcal for reliability diagrams and temperature/isotonic scaling. Fit temperature scaling on a held-out split and compare before vs after.
- For a decision task, report at minimum:
  - accuracy / macro-F1
  - NLL and multiclass Brier
  - top-label ECE (15 bins) and classwise ECE
  - a reliability diagram
  - conformal set size and coverage at α = 0.1
- Compare three ways of getting a probability from the LLM:
  - (a) label-token logits, which are the best calibrated after temperature scaling;
  - (b) verbalised probability, which can be trained with a GRPO Brier-score reward;
  - (c) a `SequenceClassification` head plus LoRA.

### Gaps
- I did not verify current torchmetrics or netcal versions, or the MMLU-Pro/BoolQ dataset IDs on the HF Hub.
- I did not find 2026 papers on LLM calibration benchmarks in this pass.

## Application: sperm motility (WHO 2021 categories) from kinematic features as an LLM decision task; honest baselines

### Takeaway
The literature consistently shows that LLM-on-serialised-tabular helps mainly in the extreme few-shot regime. Once tens to hundreds of labelled rows exist, gradient-boosted trees and tabular foundation models (TabPFN-3) match or beat it at far lower compute. For 8 numeric CASA features with plentiful per-track labels, a calibrated GBDT or TabPFN is the honest baseline the LLM must beat. LLM value is likelier from:
- combining free-text notes or metadata;
- few-shot transfer to new labs or protocols;
- natural-language explanations.

### Cited Findings
- **TabLLM** (Hegselmann et al., AISTATS 2023) serialises rows to text (9 serialisation formats studied) and fine-tunes T0. It beats XGBoost and TabPFN in the very-few-shot regime. — [TabLLM PMLR](https://proceedings.mlr.press/v206/hegselmann23a/hegselmann23a.pdf)
- Revisiting GBDTs for few-shot tabular learning: TabPFN and TabLLM perform similarly on average (about 1% in favour of TabPFN). With more labelled data, gradient-boosted trees overtake LLM-based prediction at much lower compute. — [arXiv 2411.04324](https://arxiv.org/html/2411.04324v1)
- Serialisation choice matters for few-shot LLM tabular classification. — [arXiv 2312.12464](https://arxiv.org/pdf/2312.12464)
- **TabPFN-3** (May 2026) reportedly:
  - "beats 8-hour-tuned gradient-boosted-tree baselines on datasets up to 1M training rows and 200 features";
  - handles 1M rows on one H100;
  - has a TabPFN-3-Plus variant that beats AutoGluon 1.5 extreme at about 10× speed.
  
  These are vendor/author claims. — [arXiv 2605.13986](https://arxiv.org/abs/2605.13986)
- Rezaee (arXiv, Aug/Sep 2026) reviewed 186 papers across nine modalities including tabular data. Language-mediated models are competitive in "extreme few-shot prediction", but "where direct tests exist, apparent parity is traceable to information asymmetry, benchmark contamination, or computation performed outside the language model." — [arXiv 2608.28980](https://arxiv.org/abs/2608.28980)
- LLMTabBench (May 2026) evaluates LLMs on binary tabular classification from zero to few shots. I saw only the title and did not read the content. — [arXiv 2605.24417](https://arxiv.org/pdf/2605.24417)
- **Sperm-motility ML literature**:
  - SVMs on CASA kinematic features classify progressive, intermediate and hyperactivated tracks with up to about 89% accuracy.
  - An ensemble of InceptionTime models on raw (x,y) trajectories classifies progressive vs hyperactivated tracks.
  - An open-source pipeline goes from phase-contrast video to WHO 2021 motility percentages and natural-language reports.
  
  — [Refined trajectory smoothing & DL classification, PMC13016988 / PubMed 41633817](https://pubmed.ncbi.nlm.nih.gov/41633817/); [Unraveling sperm kinematic heterogeneity with ML, Asian J Androl (PMC13065317)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13065317/); [Modular DL pipeline for sperm motility, NMI](https://journals.uio.no/NMI/article/view/13361). Details come from search snippets; the full texts were blocked by captcha.
- Gaussian-mixture "motility fingerprints" suggest that kinematics vary continuously rather than falling into crisp classes. — [bioRxiv 2025.11.20.689493](https://www.biorxiv.org/content/10.1101/2025.11.20.689493.full.pdf)

### Inferences
- **Label-leakage risk**: if the progressive/non-progressive/immotile labels come from CASA rules (thresholds on VSL/VAP/STR), any model just relearns the threshold rule. Accuracy near 100% would then be meaningless. Use expert-annotated labels (e.g. from video review) or say explicitly that the task is "rule emulation".
- **Honest baseline ladder**:
  1. The CASA/WHO rule itself.
  2. Multinomial logistic regression.
  3. LightGBM/XGBoost, then temperature- or isotonic-calibrated.
  4. TabPFN (v2 or 3, which runs fine on an A100 for small data).
  5. The LLM: zero-shot, few-shot, then LoRA on label-token logits.
  
  Report all of them on the same patient-grouped splits (GroupKFold by donor/sample, to avoid track-level leakage), with NLL, Brier, ECE and conformal coverage.
- **Serialisation template**, e.g. "VCL: 85.2 µm/s; VSL: 40.1 µm/s; VAP: 52.3 µm/s; LIN: 0.47; STR: 0.77; WOB: 0.61; ALH: 3.4 µm; BCF: 11.2 Hz. Category? A) progressive B) non-progressive C) immotile".
  - Read P(A/B/C) from the next-token logits with thinking disabled.
  - Round numbers consistently; tokenising many decimals adds noise.
- **Where the LLM might add value**:
  - (a) Few-shot adaptation to a new lab, CASA device or frame rate with only a few labelled tracks.
  - (b) Fusing free-text clinical or lab notes and metadata (temperature, medium, time since collection) that tabular models cannot use directly.
  - (c) Producing explanations. Check these against the feature attributions of the tabular model, since they may be unfaithful.
  - (d) A sample-level report that aggregates per-track probabilities into WHO percentages with uncertainty.
- An expected-result framing worth preregistering: "the LLM does not beat calibrated GBDT/TabPFN on numeric features alone, but may close the gap in few-shot or multimodal settings".

### Gaps
- I could not fetch the WHO 2021 (6th edition) manual text. The writer should verify the exact category definitions. My understanding is that WHO 2021 defines progressive / non-progressive / immotile by visual assessment, and the 6th edition reinstated the rapid/slow split within progressive. The writer should also check whether any kinematic thresholds are prescribed.
- Full texts of the sperm-motility papers were blocked (captcha). Dataset names (e.g. VISEM-Tracking), label provenance and exact metrics remain unverified.
- I found no published study applying LLMs specifically to sperm-motility classification. That may be a novelty point, or simply something I missed.
