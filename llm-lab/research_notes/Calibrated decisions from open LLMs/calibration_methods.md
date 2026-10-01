# Methods for Calibrated, Typed Decisions from Open-Weight LLMs

Scope note: notes prepared 2026-10-01. Items marked [bg] are well-known results cited from background knowledge with their canonical URL; they were not re-fetched in this session, so the report writer should treat exact numbers in those items as needing a spot-check. Everything else was confirmed against the cited page in this session. "Single-paper" marks claims that rest on one paper and have not (to my knowledge) been independently replicated.

## Q1. Option/label scoring: next-token logits, sequence log-likelihood, constrained decoding, contextual calibration and label/position-bias fixes

### Takeaway
Reading the logits of the answer tokens (e.g. "A"/"B"/"C"/"D", "Yes"/"No") after one prefill pass is the cheapest way to get a probability distribution over fixed options. The raw distribution is biased by the prompt format, label tokens and option order. Cheap inference-time corrections remove much of that bias: contextual calibration, Batch Calibration, and PriDe-style permutation priors. They do not make an RLHF/instruct model's probabilities well calibrated on their own, so a small labelled calibration set is still needed (Q3).

### Cited Findings
**Mechanics**
- Mechanism: append the options and an answer cue to the prompt and run one forward pass (prefill). Take the logits at the last position, restrict them to the K label tokens, and apply softmax over those K logits. That gives a K-way distribution at the cost of one forward pass. Conformal-MCQA work uses exactly this score ("softmax over option-letter logits") on MMLU — [Kumar et al. 2023, arXiv 2305.18404](https://arxiv.org/abs/2305.18404)
- Alternative: score each option's full text by its summed (or length-normalised) log-likelihood given the context. This needs one pass per option, or one batched pass with a shared prefix, and it is sensitive to option length and surface form. [bg] This is the standard "cloze" scoring in lm-eval-harness-style evaluation.
- Constrained decoding (grammar or JSON mode) forces generated tokens into a valid set. "Let Me Speak Freely?" finds that format restrictions degrade reasoning, and that stricter constraints degrade it more — [Tam et al., EMNLP Industry 2024](https://arxiv.org/pdf/2408.02442). A 2026 follow-up reports that the accuracy cost comes mainly from the format-requesting prompt in open-weight models. Letting the model reason freely and then format in a second pass recovers most of the loss — [The Format Tax, arXiv 2604.03616](https://arxiv.org/html/2604.03616). A further 2026 paper reports that structured output collapses answer diversity across 44 models — [arXiv 2607.18476](https://arxiv.org/pdf/2607.18476) (single-paper; not read in depth).
- For a single-token decision, constrained decoding to the label set is mathematically the same as taking an argmax over the restricted logits. The probabilities you should use are the renormalised restricted logits, not the sampled token (inference).

**Bias and its correction**
- Contextual calibration ("Calibrate Before Use"; [bg]): run a content-free input (e.g. "N/A") through the same prompt to estimate the prompt's label prior p_cf. Then rescale the label probabilities with W = diag(p_cf)^-1, which makes the content-free input predict uniform. This gave large gains for GPT-3 few-shot classification and cut variance across prompt choices — [Zhao et al., ICML 2021, arXiv 2102.09690](https://arxiv.org/abs/2102.09690)
- Batch Calibration (ICLR 2024) estimates the contextual bias by averaging the LLM's label scores over a batch of real unlabelled test inputs, rather than a content-free token. It is zero-shot and inference-only, adds negligible cost, unifies prior calibration methods, and beats them by at least 6% across more than 10 tasks — [Zhou et al., arXiv 2309.17249](https://arxiv.org/abs/2309.17249); [Google Research blog](https://research.google/blog/batch-calibration-rethinking-calibration-for-in-context-learning-and-prompt-engineering/)
- Selection bias in multiple-choice questions: LLMs put a priori probability mass on particular option-ID tokens (A/B/C/D), a "token bias", which is the main intrinsic cause of the bias. PriDe splits the prediction into an intrinsic prediction over option contents and a prior over option IDs. It estimates that prior by permuting option contents on a small number of test samples, then uses it to debias later samples — [Zheng et al., "Large Language Models Are Not Robust Multiple Choice Selectors", ICLR 2024, arXiv 2309.03882](https://awesomepapers.io/llm-papers/papers/2309.03882)
- Training-time fixes for selection bias (2025–2026, each a single paper):
  - Permutation-Aware GRPO enforces consistent predictions across permutations of the options — [arXiv 2603.21016, ACL 2026](https://aclanthology.org/2026.acl-long.1621/)
  - An unsupervised LoRA fine-tune driven by a Permutation Bias Metric — [arXiv 2511.21709](https://www.arxiv.org/pdf/2511.21709)
  - "Strengthened Symbol Binding" — [arXiv 2406.01026](https://arxiv.org/pdf/2406.01026)
- Logits versus verbalised confidence: for RLHF models (GPT-3.5/4, Claude-2), verbalised confidence was often better calibrated than token probabilities on TriviaQA, SciQ and TruthfulQA, often with about 50% relative ECE reduction. Prompting for several candidate answers before stating confidence helped further — [Tian et al., EMNLP 2023, arXiv 2305.14975](https://aclanthology.org/2023.emnlp-main.330/). [bg] The GPT-4 technical report likewise showed that the pre-trained model's MMLU probabilities were well calibrated and that post-training made them worse — [arXiv 2303.08774](https://arxiv.org/abs/2303.08774). [bg] Base models are "mostly" calibrated on multiple-choice when the format is right — [Kadavath et al. 2022, arXiv 2207.05221](https://arxiv.org/abs/2207.05221)

### Inferences
- Logit scoring is the natural "one pass, distribution over K options" primitive. Its known failure modes are (a) prompt and label-token prior bias, (b) option-order bias, and (c) instruct/RLHF overconfidence. Batch Calibration or PriDe fixes (a) and (b) cheaply. Item (c) needs temperature scaling or similar on labelled data.
- Single-token labels avoid length-normalisation problems. Prefer label words that tokenise to one token each and have similar prior frequency, e.g. "Yes"/"No" or capital letters.

### Gaps
- I found no systematic 2025–2026 benchmark comparing first-token logit scoring with full-option log-likelihood on calibration (as opposed to accuracy) for current open instruct models.
- It is unclear whether Batch Calibration's gains carry over to strongly instruction-tuned models (Llama-3/Qwen-3 class). The paper's experiments predate those models.

## Q2. Classification or regression heads on an LLM backbone (last-token hidden state → linear head), trained with LoRA, versus generative fine-tuning

### Takeaway
Adding a linear head on the last-token hidden state and training it jointly with LoRA (Hugging Face `AutoModelForSequenceClassification` with PEFT `task_type=SEQ_CLS`) is a well-supported, cheap and accurate route for fixed-label decisions. It matches or beats BERT and instruction-tuning on single-label tasks with far fewer trainable parameters. Its probabilities are a softmax over K classes by construction, so calibration can be handled with standard tools. Evidence on calibration specifically comparing head-based and generative fine-tuning is thin. Fine-tuned LoRA models are known to be overconfident on small data, and Laplace-LoRA addresses that.

### Cited Findings
- 2025 comparison: an embedding/classification head on the final-token embedding of causal LLMs up to 8B, trained with 4-bit quantisation plus LoRA, matches or beats fine-tuned BERT on single-label patent classification while training 10–30x fewer parameters. Instruction-tuning (prompt → response) was competitive only for multi-label tasks, and only with at least 100M trainable parameters. No calibration results were reported — [Yousefiramandi & Cooney, arXiv 2512.12677](https://arxiv.org/abs/2512.12677) (single-paper)
- Practitioner evidence: for simple tasks, fine-tuning only the last block plus the head can match LoRA. LoRA on deeper layers helps on complex or domain-specific tasks — [rasbt LLMs-from-scratch discussion #921](https://github.com/rasbt/LLMs-from-scratch/discussions/921) (practitioner, not peer-reviewed)
- [bg] Hugging Face `LlamaForSequenceClassification` and similar classes pool the hidden state of the last non-padding token and apply a linear `score` layer. With `num_labels=1` the problem type becomes regression (MSE). PEFT's `TaskType.SEQ_CLS` keeps the head trainable alongside LoRA — [Transformers docs](https://huggingface.co/docs/transformers/model_doc/llama), [PEFT docs](https://huggingface.co/docs/peft)
- [bg] LLM2Vec turns decoder-only LLMs into bidirectional encoders through three steps: enable bidirectional attention, masked next-token prediction, and unsupervised SimCSE. It gives strong embeddings for downstream heads — [BehnamGhader et al., COLM 2024, arXiv 2404.05961](https://arxiv.org/abs/2404.05961)
- Calibration of LoRA fine-tunes: fine-tuned LLMs become overconfident, especially on small datasets. Laplace-LoRA applies a post-hoc Laplace approximation to the LoRA parameters only. Training is unchanged, and it substantially lowers ECE and NLL across tasks while keeping accuracy — [Yang et al., ICLR 2024, arXiv 2308.13111](https://arxiv.org/abs/2308.13111)
- Probe or head versus generation for confidence: Kapoor et al. report that "training through the features of a model is necessary for good performance". A frozen probe is weaker than LoRA-trained uncertainty heads — [Kapoor et al., NeurIPS 2024, arXiv 2406.08391](https://arxiv.org/abs/2406.08391)
- RLCR's baselines included a linear probe and BCE- or Brier-trained classifiers on top of an RLVR model. All improved calibration over raw RLVR, e.g. HotPotQA ECE 0.37 → 0.07 with the BCE classifier, but RLCR matched or exceeded them using one model — [Damani et al., arXiv 2507.16806](https://arxiv.org/html/2507.16806v2)

### Inferences
- The head-based approach is the most direct fit for "one pass → calibrated distribution over predefined options". There are no label-token or position biases, and no tokenisation issues.
- Its cost is that you lose zero-shot flexibility: options are fixed at training time, and adding a label needs retraining.
- Because training uses cross-entropy (a proper scoring rule), the head is calibrated in-distribution up to overfitting. Temperature scaling on a held-out split is still advisable.
- A hybrid is possible: keep the LM head but fine-tune with LoRA on "answer = label token". This is a classification head with tied, pretrained weights, and keeps the label semantics available for zero-shot use.

### Gaps
- I found no head-to-head 2024–2026 study reporting ECE/Brier for a sequence-classification head versus generative (label-token) LoRA fine-tuning on the same tasks. This is an open empirical question worth a pilot experiment.

## Q3. Post-hoc calibration: temperature scaling, Platt/isotonic, Thermometer and adaptive temperature; data needs

### Takeaway
Temperature scaling, a single scalar T fitted by NLL on held-out labelled data, is the established baseline. It preserves argmax accuracy and typically needs only hundreds to a few thousand labelled examples. LLM-specific variants learn the temperature from features, so it can transfer across tasks (Thermometer) or vary per token or input (Adaptive Temperature Scaling). These are useful when a new task has no labels, or when RLHF has distorted calibration.

### Cited Findings
- [bg] Temperature scaling (one parameter, fitted on a validation set) is the standard post-hoc fix for overconfident deep classifiers. It does not change accuracy — [Guo et al., ICML 2017, arXiv 1706.04599](https://arxiv.org/abs/1706.04599). Platt scaling (a logistic fit on the score) and isotonic regression (non-parametric and monotone) are the classical alternatives. Isotonic is more flexible but needs more data and can overfit with small calibration sets; this is general ML knowledge, and I found no LLM-specific study of data requirements.
- Thermometer (ICML 2024) trains an auxiliary recognition network on multiple tasks. The network predicts a dataset-specific temperature from the LLM's features, without labels for the new task. On MMLU with LLaMA-2-Chat 7B, ECE fell from 0.260 uncalibrated to 0.078, close to the 0.062 achieved by temperature scaling with full labels. It reduced calibration error on all 57 MMLU subjects, transferred between BIG-bench and MMLU, preserved accuracy, and added about 0.5% inference time — [Shen et al., arXiv 2403.08819](https://arxiv.org/abs/2403.08819); [ICML PDF](https://sia.mit.edu/wp-content/uploads/2024/12/2024-shen-das-greenewald-sattigeri-wornell-ghosh-icml.pdf)
- Adaptive Temperature Scaling (ATS, EMNLP 2024) predicts a per-token temperature from hidden features and is fitted on a standard SFT dataset. Its loss uses a uniform target when the model is wrong and one-hot cross-entropy when it is right. It improves post-RLHF calibration by 10–50% with no effect on accuracy — [Xie, Chen, Lee, Mitchell, Finn, arXiv 2409.19817](https://arxiv.org/abs/2409.19817); [code](https://github.com/johnathan-xie/adaptive-temperature-scaling)
- Contextual or Batch Calibration (Q1) is post-hoc and label-free, but corrects bias (a shift in the label prior) rather than sharpness (temperature). The two corrections are complementary.
- Tian et al. found that combining prompting strategies with temperature scaling cut RLHF-LM ECE by more than 50% — [arXiv 2305.14975](https://arxiv.org/abs/2305.14975)
- Data needs: Kapoor et al. report that about 1,000 graded examples are enough for a fine-tuned uncertainty estimator to beat baselines — [arXiv 2406.08391](https://arxiv.org/abs/2406.08391). Thermometer needs labelled data only on the source tasks.
- Newer single-paper work, not read in depth:
  - Bilevel optimisation beyond post-hoc temperature scaling — [arXiv 2608.07419](https://arxiv.org/pdf/2608.07419)
  - "Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator", which uses the base model to calibrate the post-trained one — [arXiv 2505.16690](https://arxiv.org/pdf/2505.16690)

### Inferences
- For a fixed decision task with even a few hundred labels, fit a vector or scalar temperature (plus a bias correction) on the K-way logits. This is cheap, robust and accuracy-preserving.
- Use Thermometer- or ATS-style methods when tasks proliferate and each lacks labels.
- Report ECE together with Brier/NLL and reliability diagrams. ECE alone is binning-sensitive.

### Gaps
- I found no rigorous LLM-specific sample-complexity curves for temperature, Platt and isotonic scaling. Practical numbers (hundreds to about 1k) come from Kapoor et al. and general ML practice.

## Q4. Calibration-aware training: proper scoring rules, RL with Brier/log rewards (RLCR and others), calibration-tuning, uncertainty-aware DPO

### Takeaway
There is good evidence that standard RL with binary correctness rewards (RLVR) and RLHF damage calibration, and that adding a bounded proper scoring rule (Brier) to the reward restores it with no accuracy loss. RLCR (MIT, 2025–2026) is the main reference. Calibration-tuning (Kapoor et al., NeurIPS 2024) shows that a small LoRA fine-tune on graded correct and incorrect answers gives generalising uncertainty estimates. "Uncertainty-aware DPO" papers mostly use uncertainty to reweight preference pairs. They are not yet an established calibration method.

### Cited Findings
**RLCR (Reinforcement Learning with Calibration Rewards)** — [Damani, Puri, Slocum, Shenfeld, Choshen, Kim, Andreas, arXiv 2507.16806 (v2 May 2026)](https://arxiv.org/abs/2507.16806); [project page](https://rl-calibration.github.io/); listed as ICLR 2026 by a [paper-notes site](https://en.papernotes.org/ICLR2026/reinforcement_learning/beyond_binary_rewards_training_lms_to_reason_about_their_uncertainty/)
- How it works: the model outputs reasoning, an answer and a numeric confidence q. The reward is R = 1[y≡y*] − (q − 1[y≡y*])², i.e. correctness plus a Brier term.
- Theory: the reward is maximised when q equals the true success probability and the answer is the most likely correct one. This requires a bounded scoring rule. The log score is unbounded and can incentivise wrong answers.
- Results: the main model is Qwen2.5-7B, with OLMo-2 and Qwen3 ablations.

  | Setting | Base | RLVR | RLCR |
  |---|---|---|---|
  | HotPotQA accuracy | 39.7% | 63.0% | 62.1% |
  | HotPotQA ECE | 0.53 | 0.37 | 0.03 |
  | HotPotQA Brier | – | 0.37 | 0.21 |
  | OOD factual ECE | 0.40 | 0.46 | 0.21 |
  | OOD factual accuracy | 53.3% | 53.9% | 56.2% |
  | Math in-domain ECE | – | 0.26 | 0.10 |

  RLVR made out-of-distribution calibration worse than the base model.
- Pitfalls:
  - Out-of-distribution ECE of about 0.21–0.25 is still high.
  - Confidences across mutually exclusive answers can sum to more than 1 out of distribution.
  - An SFT warm-up improves calibration but causes forgetting (out-of-distribution math accuracy 43.8% vs 50.9%).
  - It depends on format tags.
  - Test-time confidence ensembling gives only modest gains.
- Other 2026 single papers in the same vein, not read in depth:
  - Calibration-Aware Policy Optimization — [arXiv 2604.12632](https://arxiv.org/pdf/2604.12632)
  - UCPO, Uncertainty-Aware Policy Optimization — [arXiv 2601.22648](https://arxiv.org/pdf/2601.22648)

**Calibration-tuning** — [Kapoor, Gruver, et al., NeurIPS 2024, arXiv 2406.08391](https://arxiv.org/abs/2406.08391)
- Fine-tuning on a small dataset of graded correct and incorrect answers gives an uncertainty estimate with good generalisation and small overhead. About 1,000 graded examples beat the baselines.
- LoRA (optionally combined with prompting) beats both prompting-only methods and frozen probes. Estimates generalise to new formats and subjects, and even to other models' generations.

**Post-RLHF calibration repair**
- ATS (Q3) targets the miscalibration that RLHF introduces — [arXiv 2409.19817](https://arxiv.org/abs/2409.19817)

**Uncertainty-aware preference optimisation**
- IUPO uses uncertainty-aware iterative DPO for reasoning; it is aimed at accuracy, not calibration — [ACL 2025](https://aclanthology.org/2025.acl-long.1169/)
- TUR-DPO adds a calibrated uncertainty score to the reward, down-weights uncertain pairs, and reports lower ECE and Brier — [arXiv 2605.00224](https://arxiv.org/html/2605.00224) (single-paper)

**Proper scoring rules generally**
- [bg] Cross-entropy (log loss) and Brier are strictly proper. Supervised fine-tuning with cross-entropy on label tokens or a classification head is already proper-scoring-rule training. Miscalibration then comes mainly from overfitting, distribution shift, or later RLHF/RLVR stages (inference from the above).

### Inferences
- For a "System One" decision model that outputs a distribution over fixed labels, you do not need RL with verbalised confidence. Cross-entropy training of a head or label tokens, followed by temperature scaling, already optimises a proper scoring rule.
- RLCR-style rewards matter when the model must reason first and then report a confidence, i.e. a "System Two" setting, or when a policy has already been RL-trained and its logits are no longer calibrated.
- RLCR's own probe and classifier baselines reach ECE 0.07 in-domain. A cheap probe or head on top of the model is a strong alternative to full RL.

### Gaps
- There is little independent replication of RLCR on non-QA classification tasks.
- I found no study comparing RLCR with a simple head plus temperature scaling on multi-class decision tasks.

## Q5. Conformal prediction for LLMs (MCQA/classification sets; conformal language modeling)

### Takeaway
Split conformal prediction wraps any score, such as the softmax over option logits, to produce prediction sets with a finite-sample marginal coverage guarantee P(y ∈ C(x)) ≥ 1−α. It needs only an exchangeable held-out calibration set of a few hundred to a few thousand items, and no retraining. It is established and cheap: one pass per item plus a quantile. Its limits are the exchangeability assumption under topic or prompt shift, and that the guarantee is marginal rather than conditional.

### Cited Findings
- [bg] Reference tutorial: split conformal method, LAC/APS/RAPS scores, coverage theorem, and conditional-coverage caveats — [Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction", arXiv 2107.07511](https://arxiv.org/abs/2107.07511)
- MCQA with LLMs: the method uses softmax over option-letter logits as the score and evaluates LAC and APS on MMLU. Conformal uncertainty (set size) is tightly correlated with accuracy and is useful for selective classification. The paper also studies exchangeability across subjects — [Kumar et al., arXiv 2305.18404 (ICML 2023 workshop)](https://arxiv.org/abs/2305.18404)
- Coverage definition and the attraction for LLMs: no retraining is needed, only calibration and test sets. Black-box variants use frequency over samples (predictive entropy from repeated sampling) as the score — [overview via search; arXiv 2508.05544 "Conformal Sets in MCQA under Black-Box Settings"](https://pith.science/paper/2508.05544)
- Extensions (single papers):
  - Adaptive, prompt-dependent conformal scores that improve conditional coverage — [arXiv 2604.13991](https://arxiv.org/html/2604.13991v1)
  - Paraphrase-robust conformal prediction — [OpenReview](https://openreview.net/forum?id=Uf04r8gDn7)
  - Enhanced conformal prediction for medical MCQA — [arXiv 2503.05505](https://arxiv.org/abs/2503.05505)
- Conformal Language Modeling (ICLR 2024) is for open-ended generation. It calibrates a stopping rule for sampling, adding candidates to a set until at least one admissible response is covered with high probability. A rejection rule removes low-quality or redundant samples. It works with any sampling API — [Quach et al., arXiv 2306.10193](https://arxiv.org/abs/2306.10193)
- Related: ConU, conformal uncertainty for LLM generation — [EMNLP Findings 2024](https://aclanthology.org/2024.findings-emnlp.404.pdf). A survey — [Campos et al., arXiv 2405.01976](https://arxiv.org/pdf/2405.01976). A library — [TorchCP, arXiv 2402.12683](https://arxiv.org/pdf/2402.12683)

### Inferences
- For a fixed-option decision model, conformal prediction is a cheap layer on top of whatever score Q1–Q3 produce. Better-calibrated scores give smaller sets, but coverage holds regardless.
- In practice it fits a decision policy of "act if |C(x)| = 1, defer or escalate otherwise".
- Class-conditional (Mondrian) conformal prediction is advisable when classes are imbalanced.

### Gaps
- There is limited evidence on how quickly coverage breaks under realistic prompt or domain drift for LLM classifiers. The paraphrase-robust and adaptive papers are early.

## Q6. Evaluating several independent questions over one shared context in one pass (shared prefix / KV-cache reuse, multi-head outputs)

### Takeaway
There are three engineering routes, all supported by sources.
1. Prefix (KV) caching: encode the context once, then run one short suffix per question.
2. Packing several questions into one sequence with custom attention masks and virtual position IDs, so that they cannot see each other (IPPD, Parallel Prompting).
3. Multiple classification heads on one pooled representation (inference only; no dedicated paper found).

For single-token decisions, approach 1 is nearly free beyond the context prefill. Approach 2 gives up to about 7x throughput when outputs span multiple tokens.

### Cited Findings
- Intra-Prompt Parallel Decoding (IPPD, Amazon/McGill/Mila, 2026) stacks the context and several questions in one prompt. Each token gets a virtual position ID, as if processed alone. Attention masks enforce causality, context isolation and question separation, so there is no cross-question leakage. Results: up to 7x effective throughput versus standard batching, 91–100% exact-match agreement across six models, and it usually beats prefix caching with PagedAttention. Limitations: all questions must be known in advance and be independent, and the gain is smaller for single-token answers and very long contexts — [Glavas et al., arXiv 2609.05707](https://arxiv.org/html/2609.05707)
- Parallel Prompting does a single pass over the shared context and decodes all continuations in parallel with matrix–matrix operations. It avoids cross-query interference and supports dynamic KV-cache management — [OpenReview T5KBO4IeM2](https://openreview.net/forum?id=T5KBO4IeM2)
- "Divide, Cache, Conquer" (2025) recasts multi-label classification as independent yes/no questions per dimension (24 affect dimensions), with prefix caching for efficiency. Small models (Gemma3-1B, PLLuM-8B, and others) are distilled from DeepSeek-V3 annotations — [Langner et al., arXiv 2511.03830](https://arxiv.org/abs/2511.03830)
- Practitioner note on whether to put several independent questions in one request or split them into many — [DEV Community](https://dev.to/eyanpen/multiple-independent-questions-batch-into-one-request-or-split-into-many-an-analysis-of-llm-1h6m) (not peer-reviewed). There is also KV-cache-sharing infrastructure work, but much of it is about systems or security, e.g. [NDSS 2025 on prompt leakage via KV sharing](https://www.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf)

### Inferences
- With prefix caching (vLLM or SGLang automatic prefix caching), each additional yes/no or K-way question costs only the suffix tokens plus one decode position. Reading logits at the last suffix token needs no decoding at all.
- With a sequence-classification head, one forward pass gives one representation. For several questions you can either:
  - use multi-task heads on the same pooled state, if the questions are fixed at training time, or
  - use question-specific suffixes with prefix caching, each followed by a head.
- The IPPD-style masking trick also works for scoring: put Q1…Qn after the context, each attending only to the context and itself, and read the label logits at each question's last token in one forward pass.
- Pitfall: naive packing (several questions visible to each other in one prompt, with joint answers) introduces cross-question interference and order effects. The masking work exists to avoid this.

### Gaps
- I found no paper evaluating calibration, as opposed to accuracy and throughput, under packed or masked multi-question evaluation, and no paper on multi-head LLM classifiers for independent questions over a shared context.

## Q7. Which approaches best fit a "System One decision model" (fast, one pass, calibrated probabilities over predefined options)?

### Takeaway
The best-supported pipeline has four layers:
1. Fine-tune with LoRA using cross-entropy, either with a classification head on the last-token state or on the label tokens of the LM head.
2. Fit temperature (and bias) scaling on a few hundred to about 1k held-out labelled examples.
3. Optionally add a conformal layer for coverage-guaranteed abstention.
4. Use prefix caching or masked packing to answer several questions per context.

Zero-shot logit scoring with Batch Calibration or PriDe is the no-training fallback. RL with calibration rewards (RLCR) suits reasoning-first "System Two" settings better.

### Cited Findings
- The head-based LoRA classifier is accurate and parameter-efficient for single-label decisions — [arXiv 2512.12677](https://arxiv.org/abs/2512.12677)
- Laplace-LoRA improves the calibration of LoRA fine-tunes post hoc — [arXiv 2308.13111](https://arxiv.org/abs/2308.13111)
- About 1k graded examples are enough to train good uncertainty estimators — [arXiv 2406.08391](https://arxiv.org/abs/2406.08391)
- Temperature-style methods (Thermometer, ATS) fix LLM miscalibration without accuracy loss — [arXiv 2403.08819](https://arxiv.org/abs/2403.08819), [arXiv 2409.19817](https://arxiv.org/abs/2409.19817)
- Label-free bias correction for zero-shot logit scoring — [Batch Calibration](https://arxiv.org/abs/2309.17249), [PriDe](https://awesomepapers.io/llm-papers/papers/2309.03882)
- Conformal sets from option logits — [arXiv 2305.18404](https://arxiv.org/abs/2305.18404)
- Shared-context multi-question efficiency — [IPPD](https://arxiv.org/html/2609.05707), [Divide-Cache-Conquer](https://arxiv.org/abs/2511.03830)
- RLVR or binary-reward RL harms calibration, and RLCR repairs it. RLCR still needs generated reasoning plus a verbalised confidence, which is multi-token and slower — [arXiv 2507.16806](https://arxiv.org/html/2507.16806v2)
- Constrained or structured generation can cost accuracy when reasoning is suppressed — [arXiv 2408.02442](https://arxiv.org/pdf/2408.02442)

### Inferences

**Comparison matrix** (my synthesis)

| Approach | Passes | Training | Calibration quality | Main pitfalls |
|---|---|---|---|---|
| Zero-shot label-logit scoring | 1 (prefill) | none | poor for instruct/RLHF models | prompt, label and position bias |
| + Batch Calibration / PriDe | 1 (+ small permutation set for PriDe) | none | bias corrected, sharpness not | — |
| + temperature scaling | 1 | ~100s–1k labels | good in-distribution, preserves accuracy | — |
| LoRA + classification head (or label-token SFT) | 1 | 1k–10k+ labels | proper-loss trained, good with temperature scaling, Laplace-LoRA for small data | fixed label set; overfitting overconfidence |
| Calibration-tuned uncertainty head | 1–2 | ~1k graded | generalises across tasks | needs a second head or prompt |
| RLCR / verbalised confidence | many tokens (decode) | RL | best when reasoning is needed | slow; OOD ECE still ~0.2 |
| Conformal wrapper | +0 | calibration set | guaranteed marginal coverage | exchangeability; set size depends on score quality |

- Rule of thumb for an ML/CV researcher: treat the LLM as a backbone, exactly like fine-tuning a ViT with a linear head. Calibrate with temperature scaling as in CV. The LLM-specific issues are (i) prompt, label-token and position biases if you use the LM head, and (ii) RLHF-induced overconfidence if you start from an instruct model. Starting from a base model, or using a separate head, sidesteps (ii).

### Gaps
- No single paper evaluates the full "System One" pipeline (head + temperature scaling + conformal + shared-prefix multi-question) end to end on calibration metrics.
- It is unresolved whether base or instruct backbones give better-calibrated heads after LoRA fine-tuning. The GPT-4 report and Tian et al. suggest RLHF harms logit calibration, but I found no head-based study confirming this for open models.
