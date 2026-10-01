# Calibration and Uncertainty in Large Language Models: State of Research (2023–2026)

Orientation for a CV/ML reader: an LLM answers by generating tokens; "confidence" can be read out in three main ways. (1) **Token-probability / logit-based**: the softmax probability the model assigned to the answer tokens (e.g. the probability of the letter "B" in a multiple-choice question, or the product/mean of per-token probabilities for a free-text answer); this is the direct analogue of max-softmax confidence in an image classifier. (2) **Verbalized**: the model is prompted to *write* a number ("Confidence: 80%"); this is just more generated text, not an internal probability. (3) **Sampling-consistency**: sample K answers at temperature > 0 and use agreement (e.g. the fraction matching the majority answer, or "semantic entropy" over meaning-clusters of answers) as confidence; analogous to MC-dropout/ensemble disagreement. "Base"/"pre-trained" model = trained only on next-token prediction; "post-trained"/"aligned" = further instruction tuning (SFT), RLHF/DPO preference tuning, and/or RL with verifiable rewards (RLVR, reward = answer is checkably correct, used for "reasoning" models like o1/o3, DeepSeek-R1, Qwen3).

## 1. Verbalized vs token-probability vs sampling-consistency confidence: which is better calibrated, and where?

### Takeaway
There is no universal winner, and recent work (2025–2026) shows the ranking itself is highly sensitive to the measurement protocol. Rough consensus: for **base models on closed-form tasks**, token probabilities are well calibrated; for **RLHF/chat models**, raw token probabilities are over-sharpened and verbalized confidence is often better *calibrated* but poor at *discrimination* (AUROC near chance, confidences clustered at 80–100%); **sampling consistency** (incl. semantic entropy) is the most robust signal across tasks and especially for long-form generation, at K× inference cost; **trained/fine-tuned confidence estimators** beat all prompting-only methods.

### Cited Findings
- Foundational (2022): Kadavath et al., "Language Models (Mostly) Know What They Know" (Anthropic) — large pre-trained models are well calibrated on multiple-choice and true/false questions when presented in an appropriate format; models can be asked to evaluate P(True) of their own sampled answers, and trained to predict P(IK) ("I know"); the RLHF policy is miscalibrated but this can largely be fixed by temperature adjustment. — [arXiv 2207.05221](https://arxiv.org/abs/2207.05221)
- Foundational (2023): Tian et al., "Just Ask for Calibration" (EMNLP 2023) — for RLHF models (ChatGPT, GPT-4, Claude), verbalized confidences are typically *better calibrated* than the model's conditional token probabilities, giving ~50% relative reduction in ECE on TriviaQA, SciQ, TruthfulQA. — [arXiv 2305.14975](https://arxiv.org/abs/2305.14975)
- Xiong et al., "Can LLMs Express Their Uncertainty?" (ICLR 2024) — systematic framework of prompting × sampling × aggregation for black-box confidence; LLMs are overconfident when verbalizing (possibly mimicking human patterns); consistency across multiple sampled responses and better aggregation mitigate overconfidence; calibration and failure prediction improve with scale; the white-box (logit) vs black-box gap is narrow (AUROC 0.522 vs 0.605); no single technique consistently wins and all struggle on knowledge-intensive/specialist tasks. — [arXiv 2306.13063](https://arxiv.org/abs/2306.13063)
- OpenAI SimpleQA (2024) measured two black-box calibration signals: (a) stated confidence — all models fall below y=x (overconfident); larger models more calibrated (o1-preview > o1-mini, GPT-4o > GPT-4o-mini); (b) answer frequency over 100 repeated samples — accuracy rises with frequency, and o1-preview's frequency ≈ accuracy (best calibrated). I.e. sampling consistency was better calibrated than stated confidence. — [Wei et al. 2024, arXiv 2411.04368](https://arxiv.org/abs/2411.04368); [OpenAI blog](https://openai.com/index/introducing-simpleqa/)
- Semantic entropy (Farquhar et al., *Nature* 2024): cluster sampled answers by bidirectional entailment and compute entropy over meanings; detects confabulations better than naive token-entropy baselines across QA datasets and models. — [Nature 630, 2024](https://www.nature.com/articles/s41586-024-07421-0)
- Kapoor et al. (NeurIPS 2024), "LLMs Must Be Taught to Know What They Don't Know" — prompting alone is insufficient for good calibration; fine-tuning ("calibration tuning") on ~1,000 graded correct/incorrect examples improves ECE and AUROC over standard baselines and generalizes; gains plateau after ~5,000 examples. — [arXiv 2406.08391](https://arxiv.org/abs/2406.08391); [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2024/file/9c20f16b05f5e5e70fa07e2a4364b80e-Paper-Conference.pdf)
- 2026 protocol-sensitivity study ("Same Answer, Different Confidence", Llama-3-8B, Mistral-7B, Qwen2.5-7B × TriviaQA, SciQ, TruthfulQA, MMLU = 12 settings): whether verbalized beats token likelihood depends on the conditioning context. Changing only where/how the answer is scored (bare query vs a "Guess:" slot) flipped the ECE winner in 4/12 settings (mean gap shift 0.254) and the AUROC winner in 9/12 (mean shift 0.158). Verbalized beats the bare-turn token score in 10/12 settings by AUROC but loses to the Guess-slot token score in 11/12. Choice of ECE estimator changed 0/12 winners. Verbalized scores separated correct from plausible-wrong answers by only +0.031. Recommends reporting answer provenance, scored string, token readout, full conditioning text. — [arXiv 2605.27752](https://arxiv.org/html/2605.27752)
- 2026 ACL Findings paper on evaluation pitfalls: verbalization-based confidences are extremely sparse (few distinct values, e.g. 80/90/95), hurting ranking metrics; token logprobs remain stable under proper evaluation; combining token probabilities with verbalization reduces sparsity. — [ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1671.pdf)
- Long-form scientific QA benchmark (2026; 685k responses, up to 20 LLMs, 7 datasets): only *semantic consistency of the final answer* yields well-calibrated outputs; verbalized and token-aggregation sequence-level methods "fail systematically"; instruction tuning causes token-probability polarization (collapsed confidence distributions). — [arXiv 2602.00279](https://arxiv.org/abs/2602.00279)
- LM-Polygraph benchmark (Vashurin et al., TACL 2025): unified benchmark of many UQ methods (information-based, sampling/consistency, density-based, verbalized/reflexive) over 11 text-generation tasks, with confidence-normalization methods; evaluates with Prediction-Rejection Ratio (PRR). — [arXiv 2406.15627](https://arxiv.org/abs/2406.15627)
- Calibrating verbalized probabilities post hoc is an active line (e.g. 2024 work on calibrating verbalized probability distributions over classes). — [arXiv 2410.06707](https://arxiv.org/pdf/2410.06707)

### Inferences
- For an open-weights model where logits are available, the practical ordering suggested by the literature is: trained probe/calibration-tuned head ≥ sampling consistency (semantic entropy, majority-vote frequency) ≥ well-formatted token probability (with temperature scaling on a held-out set) ≳ verbalized; verbalized is mainly attractive when only API text is available.
- The ECE-vs-AUROC split matters: verbalized confidence can look "calibrated" on ECE simply because average confidence ≈ average accuracy, while having little ability to rank which answers are wrong (AUROC ~0.5–0.6).
- Reported comparisons are not portable across papers unless the prompt template, answer-scoring position, and token aggregation are identical.

### Gaps
- No single large head-to-head table across all three families on current frontier (2026) models was found; numbers come from different models/datasets and are not directly comparable.
- Precise per-dataset ECE values for Tian et al. and Xiong et al. were not extracted in this session (only summary statistics).

## 2. Does RLHF / instruction / preference tuning harm calibration? What about RLVR for reasoning models?

### Takeaway
Strong and repeated evidence that preference-based post-training (RLHF/PPO, also SFT) degrades the calibration of *token probabilities* (over-sharpening toward 0/1), canonically shown in the GPT-4 report (MMLU ECE 0.007 → 0.074). RLHF also induces verbalized overconfidence, traced partly to reward models favouring confident-sounding answers. RLVR with binary correctness rewards raises accuracy but further drives overconfidence; adding a proper-scoring-rule (Brier) term to the reward (RLCR and related 2025–2026 methods) restores calibration without accuracy loss. Separately, *long chain-of-thought* reasoning appears to improve verbalized calibration relative to non-reasoning models, though reasoning models remain overconfident on hard knowledge tasks.

### Cited Findings
- GPT-4 Technical Report (2023, foundational): on a subset of MMLU, the pre-trained GPT-4 is highly calibrated (ECE 0.007); after post-training (PPO/RLHF) ECE rises to 0.074 (reliability diagram, Fig. 8). — [arXiv 2303.08774](https://arxiv.org/abs/2303.08774); [PDF](https://cdn.openai.com/papers/gpt-4.pdf)
- Kadavath et al. 2022 (foundational): RLHF policies are miscalibrated, largely fixable by temperature rescaling. — [arXiv 2207.05221](https://arxiv.org/abs/2207.05221)
- Zhu et al. 2023, "On the Calibration of Large Language Models and Alignment" (EMNLP Findings 2023): track calibration through pre-training and alignment stages; alignment (instruction tuning, RLHF) worsens calibration. — [arXiv 2311.13240](https://arxiv.org/abs/2311.13240)
- Leng et al. (2024/ICLR 2025), "Taming Overconfidence in LLMs: Reward Calibration in RLHF": RLHF-trained LLMs are verbally overconfident with sharpened output probability; reward models used in PPO are biased to give high scores to high-confidence responses regardless of quality. Proposed PPO-M (confidence-aware reward model training) and PPO-C (reward adjustment via moving average) reduce calibration error with comparable task performance (Llama3-8B, Mistral-7B; six datasets). — [arXiv 2410.09724](https://arxiv.org/abs/2410.09724)
- 2026 position paper "Calibration as a First-Class Criterion in LLM Evaluation": synthesizes evidence that instruction tuning and RLHF hurt calibration even when accuracy improves; models are more confident in an answer presented as their own output than when the same answer comes from the user; none of nine major model families (GPT, Claude, Gemini, DeepSeek, Llama, Qwen, Gemma, OLMo) report calibration metrics in releases. — [arXiv 2609.26489](https://arxiv.org/html/2609.26489v1)
- Long-form benchmark (2026): instruction tuning causes systematic token-probability polarization; reasoning models diverge—some reproduce polarization, others mitigate it—clustering by provider (training pipeline matters). — [arXiv 2602.00279](https://arxiv.org/abs/2602.00279)
- Damani et al. 2025, RLCR ("Beyond Binary Rewards"): binary-reward RL degrades calibration and increases hallucination. Concrete numbers (Qwen-based model trained to output answer + verbalized confidence):
  - HotpotQA: Base acc 39.7%, AUROC 0.54, Brier 0.53, ECE 0.53 → RLVR acc 63.0%, AUROC 0.50, Brier 0.37, ECE 0.37 → RLCR acc 62.1%, AUROC 0.69, Brier 0.21, ECE 0.03.
  - Math: Base acc 56.1%, ECE 0.39 → RLVR acc 72.9%, AUROC 0.47, ECE 0.26 → RLCR acc 72.7%, AUROC 0.67, Brier 0.17, ECE 0.10.
  - OOD average: RLVR Brier/ECE 0.46/0.46 (worse than base) vs RLCR 0.21/0.21.
  - Theory: reward = correctness + bounded proper scoring rule yields both accurate and calibrated optima. — [arXiv 2507.16806](https://arxiv.org/html/2507.16806)
- Other 2026 RLVR-calibration methods report that RLVR pushes confidence toward saturation (≈1.0) because rollouts lack "uncertain but correct" examples; e.g. DCPO reports AIME 2025 ECE reduced from 0.435 (Qwen3-8B) to 0.128 (−71.6%). — [arXiv 2603.09117 (via aggregator)](https://pith.science/paper/2603.09117); see also [CAPO, arXiv 2604.12632](https://arxiv.org/pdf/2604.12632), [CREDO, arXiv 2609.34857](https://arxiv.org/html/2609.34857) (numbers from search snippets; not independently verified)
- Yoon et al. (NeurIPS 2025), "Reasoning Models Better Express Their Confidence": in 33 of 36 settings (6 reasoning models × 6 datasets), reasoning models have better calibrated verbalized confidence than non-reasoning counterparts; calibration improves as the CoT unfolds; removing "slow thinking" behaviours (backtracking, exploring alternatives) degrades it; non-reasoning models improve when guided to slow-think in-context. — [arXiv 2505.14489](https://arxiv.org/abs/2505.14489)
- Mei et al. (EACL Findings 2026), "Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know?": reasoning models (o3-mini, DeepSeek-R1, Claude 3.7 Sonnet, Gemini) are systematically overconfident, often stating 85–100% even when wrong on SimpleQA and GPQA; ECE strongly negatively correlated with accuracy (r = −0.96 to −1.00 for most models); deeper reasoning helps calibration only while it raises accuracy—once accuracy saturates, more reasoning increases ECE; introspection prompts improve o3-mini and R1 but worsen Claude 3.7 Sonnet. — [arXiv 2506.18183](https://arxiv.org/abs/2506.18183); [ACL Anthology](https://aclanthology.org/2026.findings-eacl.178/)

### Inferences
- Two distinct effects are often conflated: (a) RLHF/RLVR destroys the calibration of *token probabilities* (distribution sharpening — a property of the training objective, analogous to over-trained classifiers), and (b) the *verbalized* confidence of a reasoning model can still be reasonably informative, especially when the model reasons. Yoon et al. and Mei et al. are compatible: reasoning helps relative to non-reasoning, but absolute overconfidence remains on hard/knowledge-limited tasks.
- RLVR's effect is "worse calibration unless calibration is in the reward"; the fix (Brier-augmented reward) is cheap and well-motivated by proper scoring rule theory.
- For open models, base-model logits may be a better-calibrated signal than chat-model logits for the same question (cf. 2026 "BaseCal" using base-model signals — [arXiv 2601.03042](https://arxiv.org/pdf/2601.03042), not read in detail).

### Gaps
- No vendor technical report since GPT-4 (2023) that I found publishes a pre- vs post-training calibration plot; the 2026 position paper states none of nine major releases report calibration. Qwen/DeepSeek/Google calibration disclosures were not found.
- Exact base model for RLCR (believed Qwen2.5-7B) not confirmed from the fetched text.
- DCPO/CAPO/CREDO numbers come from search snippets/aggregators and should be verified.

## 3. Calibration metrics for LLMs: ECE pitfalls, Brier, NLL, AUROC, reliability diagrams, risk–coverage

### Takeaway
ECE (top-label, binned) remains the most reported number but is increasingly criticized as bin-sensitive, biased, and confounded by accuracy; recent work recommends pairing it with a **proper scoring rule** (Brier or log-loss/NLL) plus a **discrimination/selective-prediction metric** (AUROC, AUPRC, risk–coverage/AURC, or PRR), plus reliability diagrams, and controlling for accuracy when comparing models.

### Cited Findings
- ECE estimates are bin-sensitive and statistically biased; reported value depends on rarely justified binning choices. Brier score is a proper scoring rule (cannot be gamed without genuine calibration) but requires reducing free-form outputs to binary correctness. AUROC measures *ranking* (separating correct from incorrect) not calibration. — [arXiv 2609.26489](https://arxiv.org/html/2609.26489v1)
- Accuracy confound: global ECE/Brier comparisons across LLMs are confounded by accuracy differences; after accuracy control (ACE framework: instance-, distribution-, candidate-aligned views), many reported calibration advantages weaken and rankings frequently reverse. — [Yang et al. 2026, arXiv 2606.30814](https://arxiv.org/abs/2606.30814)
- Mei et al. find ECE almost perfectly anticorrelated with accuracy (r ≈ −0.96 to −1.00), consistent with the accuracy confound. — [arXiv 2506.18183](https://arxiv.org/abs/2506.18183)
- In the 12-setting protocol study, the choice of ECE estimator changed 0/12 winners while conditioning context changed 4/12 (ECE) and 9/12 (AUROC) — i.e. prompt protocol matters more than estimator. — [arXiv 2605.27752](https://arxiv.org/html/2605.27752)
- RLCR reports accuracy, AUROC, Brier and ECE together; illustrates that RLVR can lower ECE (0.53→0.37) while AUROC drops to chance (0.50) — ECE alone would mislead. — [arXiv 2507.16806](https://arxiv.org/html/2507.16806)
- LM-Polygraph uses Prediction-Rejection Ratio (PRR) — area under a rejection (risk–coverage-style) curve normalized between random and oracle — as the main metric for selective generation. — [arXiv 2406.15627](https://arxiv.org/abs/2406.15627)
- UQ evaluation with AUROC/PRR can be confounded by response-length bias in both the UQ method and the correctness judge (e.g. LLM-as-judge), producing spurious interactions. — [arXiv 2504.13677](https://arxiv.org/pdf/2504.13677)
- Verbalized confidences are sparse (few distinct values), which degrades AUROC and makes binned ECE unstable. — [ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1671.pdf)
- Surveys catalog metrics: Shorinwa et al. survey (ACM Computing Surveys 2025) and Liu et al. KDD 2025 survey summarize ECE, Brier, AUROC, AUPRC, PRR, risk–coverage as standard. — [arXiv 2412.05563](https://arxiv.org/abs/2412.05563); [arXiv 2503.15850](https://arxiv.org/html/2503.15850)

### Inferences
- For a CV researcher: classwise-ECE vs top-label ECE debates from vision (Nixon et al. 2019; Kumar et al. 2019) carry over, but most LLM work uses a *binary* "is this answer correct?" framing, so ECE is effectively top-label/binary; for multiple-choice with full option distributions, classwise ECE or NLL is possible but rarely reported.
- Recommended minimal reporting set (synthesis): accuracy, Brier (or NLL), ECE with stated binning (or a smoothed/debiased estimator), AUROC, a risk–coverage curve/AURC or PRR, a reliability diagram, plus the exact elicitation prompt.

### Gaps
- I did not find an LLM-specific paper that systematically benchmarks smooth-ECE / debiased ECE estimators vs equal-width binning; classwise vs top-label ECE is rarely discussed in LLM papers (gap in literature, not just in search).

## 4. Known failure modes

### Takeaway
LLM confidence fails in predictable ways: systematic overconfidence (worse on harder/long-tail knowledge), strong sensitivity to prompt/format and to where the answer is scored, multiple-choice letter (option-ID) bias and option-order sensitivity, first-token vs text-answer mismatch in instruction-tuned models, self-preference ("my own answer"), and degradation for long-form outputs where token-level signals collapse.

### Cited Findings
- **Overconfidence**: verbalized confidence clusters at high values (Xiong et al. ICLR 2024 — [arXiv 2306.13063](https://arxiv.org/abs/2306.13063)); reasoning models state 85–100% when wrong on SimpleQA/GPQA ([arXiv 2506.18183](https://arxiv.org/abs/2506.18183)); all models below y=x on SimpleQA ([arXiv 2411.04368](https://arxiv.org/abs/2411.04368)).
- **Difficulty / knowledge shift**: ECE grows as accuracy falls (r ≈ −0.96 to −1.00) and extra reasoning on saturated hard sets mostly inflates ECE — [arXiv 2506.18183](https://arxiv.org/abs/2506.18183). All black-box methods struggle on specialist-knowledge tasks — [arXiv 2306.13063](https://arxiv.org/abs/2306.13063).
- **Distribution shift**: RLVR-trained models' calibration on OOD datasets is worse than the base model (Brier/ECE 0.46), while RLCR's Brier-reward transfers OOD (0.21) — [arXiv 2507.16806](https://arxiv.org/html/2507.16806). Calibration-tuned estimators generalize across tasks — [arXiv 2406.08391](https://arxiv.org/abs/2406.08391).
- **Multiple-choice letter (selection) bias**: across 20 LLMs and 3 benchmarks, models prefer specific option IDs (e.g. "A"), primarily due to token bias—a priori probability mass on ID tokens; PriDe debiasing separates the prior (ICLR 2024 spotlight). — [Zheng et al., arXiv 2309.03882](https://arxiv.org/abs/2309.03882)
- **Option order sensitivity** (2023): reordering answer options changes LLM accuracy substantially (gaps reported from ~13% to ~75% across benchmarks). — [Pezeshkpour & Hruschka, arXiv 2308.11483](https://arxiv.org/abs/2308.11483)
- **First-token vs text answer**: for instruction-tuned models, first-token probabilities often disagree with the generated text answer; when mismatch is high (>50%), text answers are more robust to option-order perturbations than first-token probabilities, even vs PriDe (COLM 2024). — [arXiv 2404.08382](https://arxiv.org/abs/2404.08382)
- **Protocol/prompt sensitivity**: same answer, different conditioning context → winner flips; replacing an accepted TriviaQA alias with the canonical reference raised confidence by +0.072 — [arXiv 2605.27752](https://arxiv.org/html/2605.27752).
- **Self-attribution bias**: models are more confident in an answer framed as their own than when the user supplies it — [arXiv 2609.26489](https://arxiv.org/html/2609.26489v1).
- **Introspection non-uniform**: asking to critique own reasoning helps o3-mini/R1, hurts Claude 3.7 Sonnet — [arXiv 2506.18183](https://arxiv.org/abs/2506.18183).
- **Long-form vs short-answer**: sequence-level token aggregation and verbalized confidence fail systematically in long-form scientific QA; only final-answer semantic consistency is calibrated — [arXiv 2602.00279](https://arxiv.org/abs/2602.00279). Linguistic calibration of long-form generations (Band et al., ICML 2024) trains models to express calibrated hedges per claim — [arXiv 2404.00474](https://arxiv.org/abs/2404.00474).
- **Length bias in evaluation**: UQ methods and correctness judges both correlate with response length, distorting AUROC/PRR — [arXiv 2504.13677](https://arxiv.org/pdf/2504.13677).

### Inferences
- For any MCQ-style decision task with open models, letter-token probabilities should be debiased (permute options and average, or PriDe-style prior removal) before being treated as calibrated probabilities.
- Long-form outputs should be decomposed into atomic claims or final answers before applying confidence; whole-sequence likelihood is unreliable.

### Gaps
- Few quantitative studies of calibration under *temporal* or *domain* shift for frontier 2025–2026 models; evidence is mostly OOD datasets within QA.

## 5. Key surveys and benchmark suites (2024–2026)

### Takeaway
The field has consolidated around a few surveys (Shorinwa et al. ACM CSUR 2025; Liu et al. KDD 2025; 2025–2026 follow-ups) and benchmark/tooling suites (LM-Polygraph, SimpleQA's calibration analyses, long-form UQ benchmarks), with 2026 work emphasizing evaluation methodology (protocol sensitivity, accuracy control).

### Cited Findings
- Shorinwa et al., "A Survey on UQ of LLMs: Taxonomy, Open Research Challenges, and Future Directions" (ACM Computing Surveys 2025): taxonomy of token-level, self-verbalized, semantic-similarity, and mechanistic-interpretability UQ; applications; lists benchmarks (TruthfulQA, FEVER, HaluEval, FActScore, ParaRel, AmbigQA, MAQA, LM-Polygraph). — [arXiv 2412.05563](https://arxiv.org/abs/2412.05563); [ACM](https://dl.acm.org/doi/10.1145/3744238)
- Liu et al., "Uncertainty Quantification and Confidence Calibration in LLMs: A Survey" (KDD 2025). — [arXiv 2503.15850](https://arxiv.org/html/2503.15850); [ACM](https://dl.acm.org/doi/10.1145/3711896.3736569)
- "A Survey of Uncertainty Estimation Methods on LLMs" (2025). — [arXiv 2503.00172](https://arxiv.org/pdf/2503.00172)
- "From Passive Metric to Active Signal: The Evolving Role of UQ in LLMs" (2026 survey; UQ used for decision/control, e.g. in RL and agents). — [arXiv 2601.15690](https://arxiv.org/pdf/2601.15690)
- LM-Polygraph benchmark (TACL 2025): 11 tasks, many UQ methods, normalization, PRR metric; code and models public. — [arXiv 2406.15627](https://arxiv.org/abs/2406.15627)
- Ye et al. 2024, "Benchmarking LLMs via Uncertainty Quantification" — uses conformal prediction to add uncertainty to LLM leaderboards (foundational benchmark for conformal LLM evaluation). — [arXiv 2401.12794](https://arxiv.org/abs/2401.12794)
- SimpleQA (OpenAI 2024) — short-form factuality with built-in calibration analyses (stated confidence and answer frequency). — [arXiv 2411.04368](https://arxiv.org/abs/2411.04368)
- Long-form scientific QA UQ benchmark (2026; 20 LLMs, 7 datasets, 685k responses). — [arXiv 2602.00279](https://arxiv.org/abs/2602.00279)
- KalshiBench (2025): epistemic calibration measured against prediction-market questions with unknown-at-training outcomes. — [arXiv 2512.16030](https://arxiv.org/abs/2512.16030)
- AbstentionBench / selective answering and conformal methods are adjacent benchmark lines (not fetched here).

### Inferences
- For a PhD student entering the area, the minimum reading list: Kadavath 2022 → GPT-4 report Fig. 8 → Tian 2023 → Xiong 2024 → Farquhar 2024 (semantic entropy) → Kapoor 2024 → Shorinwa/Liu surveys → LM-Polygraph → RLCR 2025 → Yoon 2025 & Mei 2025/26 → protocol-sensitivity and accuracy-controlled evaluation (2026).

### Gaps
- Did not verify contents of AbstentionBench, KalshiBench, or the 2026 "Passive Metric to Active Signal" survey beyond titles/snippets.
