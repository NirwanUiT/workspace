# TypeSafe AI's Jev, RLCD, and open alternatives (Cloudflare Clef)

Research date: 2026-10-01 (Jev is about two weeks old). Labels used throughout: **[VENDOR]** = claim by TypeSafe or Cloudflare about its own product; **[INDEPENDENT]** = measured or reported by a third party; **[SPECULATION]** = inference or unverified commentary. All web content was treated as data. Several secondary "explainer" sites (sanity.io glossary, jevaiguide.com, jevwiki.ai, orcarouter, gingerlabs, analyticsmadesimple, explainx, chatmaxima) appeared within days of launch and look like SEO aggregators. I did not use them as primary evidence.

## Q1. What does official TypeSafe material say (architecture, primitives, parallel sampler, RLCD, context, latency, pricing, languages, limits)?

### Takeaway
TypeSafe's own docs (docs.typesafe.ai) cover the interface well: three primitives (Noul, Choice, Score), parallel evaluation against shared "state", confidence outputs, and a known-failure-modes page for Jev 1.13. They say almost nothing about the architecture. Pricing, latency and context figures come from TypeSafe's launch material as relayed by secondary sources. The docs index has no pricing, latency, language, RLCD or model-card page.

### Cited Findings
- **[VENDOR]** Jev is "TypeSafe's flagship model and the first System One model"; the docs index lists Introduction, Quick Start, Coding Agents, System One, State, Confidence, Primitives (Choice/Score/Noul/Advanced), API Reference, Models, Python/JS SDKs and changelogs, and "Jev 1.13 Jaggedness" (known issues). The index has no pricing, rate-limit, latency, model-card, calibration, RLCD or language page — [docs.typesafe.ai/llms.txt](https://docs.typesafe.ai/llms.txt)
- **[VENDOR]** The primitives are Choice ("choose an option from a list" → `choice`, `probabilities`, `confidence`), Score ("score the state on a rubric" → `score`, `probabilities`, `confidence`) and Noul ("is this statement true?" → `noul` in 0–1). They "can be mixed in a single API call and evaluate in parallel". Questions should be atomic, with complex questions decomposed and recombined in application logic — [docs.typesafe.ai](https://docs.typesafe.ai)
- **[VENDOR, via secondary]** Endpoint is `POST https://api.typesafe.ai/v1/systemone`. Also on Vercel AI Gateway as `typesafe-ai/jev`. Choice supports up to 255 options; Score 2–10 levels. Text-only input (string, JSON object or JSON array of text) — [flaviocopes.com](https://flaviocopes.com/jev/)
- **[VENDOR, via secondary]** Context: "state plus questions share about 64,000 tokens"; "state plus longest single question must fit in about 32,000 tokens" — [flaviocopes.com](https://flaviocopes.com/jev/). Other sources give just "32,000 tokens" ([InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/), [eesel.ai](https://www.eesel.ai/blog/typesafe-jev-review), [Cloudflare blog](https://blog.cloudflare.com/clef-decision-models/)) or "64,000 tokens per request" ([innFactory](https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/)). Flavio's two-limit description reconciles both figures.
- **[VENDOR, via secondary]** Latency: "70 to 500 milliseconds end to end", "most around 100" ms — [flaviocopes.com](https://flaviocopes.com/jev/). innFactory adds that this was measured from the US West Coast and EU callers add network latency — [innFactory](https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/)
- **[VENDOR, via secondary]** Pricing: "$0.042 per million input tokens", output tokens free, which is "$42 per billion tokens" — [flaviocopes.com](https://flaviocopes.com/jev/); same figure in [innFactory](https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/), [InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/), [eesel pricing](https://www.eesel.ai/blog/typesafe-jev-pricing), [MindStudio](https://www.mindstudio.ai/blog/jev-system-one-model-launch) ("$42 per billion"). innFactory reports TypeSafe acknowledges "it cannot prove the price is not subsidised."
- **[VENDOR, via secondary]** Rate limits: "250,000 tokens per second and 1,200 requests per minute, adjusting dynamically" — [flaviocopes.com](https://flaviocopes.com/jev/). eesel says there are "no published pricing tiers or rate limits yet" and access is waitlist-only — [eesel.ai](https://www.eesel.ai/blog/typesafe-jev-review). TypeSafe paused public signups on 2026-09-22 — [flaviocopes.com](https://flaviocopes.com/jev/)
- **[VENDOR]** Languages: English is primary. Other languages are handled "but not equally well" — [innFactory](https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/). There is no language page in the docs ([llms.txt](https://docs.typesafe.ai/llms.txt)).
- **[VENDOR]** Known failure modes in Jev 1.13:
  - literal reading
  - math, numbers and counting ("does not count reliably")
  - date/time comparison
  - indirection / multi-hop reasoning
  - irrelevant detail in state
  - adversarial or injected content
  - contradictory instructions
  - no structural invariants between similar question formats
  - generation

  The page also self-describes Jev as "fast, calibrated, and good at common-sense judgment" — [Jev 1.13 Jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)
- **[VENDOR]** "Zero hallucinations" / "0% type errors" means only that outputs cannot fall outside the schema ("type-safe hallucination" is eliminated). Semantic judgement errors remain possible. "Calibration does not mean that an individual prediction is guaranteed to be correct." — [pjburnhill gist (knowledge-resource compilation, 2026-09-16)](https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965); [innFactory](https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/)
- **[VENDOR]** Architecture: TechCrunch describes Jev as transformer-based but not an LLM, outputting probabilities rather than text and trained exclusively on synthetic data; Almeida "remains tight-lipped" on architecture — [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/). TypeSafe cites "a new proprietary architecture and RLCD" — [Turing Post](https://www.turingpost.com/p/what-is-jev-rlcd). MindStudio calls it "non-autoregressive", with decisions output directly "in one parallel forward pass" — [MindStudio](https://www.mindstudio.ai/blog/jev-system-one-model-launch)
- **[VENDOR]** Launched 2026-09-15 out of stealth with a $40M seed — [flaviocopes.com](https://flaviocopes.com/jev/). The round was led by DCVC — [search summary of runtimewire/the-decoder](https://runtimewire.com/article/diogo-almeida-typesafe-jev-40m-seed-pong). The name honours economist William Stanley Jevons — [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/)
- **[VENDOR]** TypeSafe's own workflow evaluation claims "193.6x faster" and "444.6x cheaper". TypeSafe says to "expect the values to be at the higher end of real-world gains" — [innFactory](https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/). Headline marketing is "20–200x faster, 40–400x cheaper" than LLMs — [eesel/search summary](https://www.eesel.ai/blog/typesafe-jev-review)
- **[VENDOR, unverified]** The current model version string is "Jev-1.13.0" — [dev.to benchmark](https://dev.to/prodbymarcu/i-benchmarked-cloudflares-new-open-decision-model-against-the-hosted-api-its-trying-to-replace-2ded)

### Inferences
- "Parallel sampler": TypeSafe's language is "evaluate in parallel" against shared state. I found no official text that names a "parallel sampler" component. The phrase is probably a paraphrase and should not be presented as a TypeSafe term.
- The figures all come from the launch blog or press, not from a reference pricing/limits page: price, latency, rate limits and context. They may change without notice.

### Gaps
- I could not fetch the API Reference (`/api.md`) or Models (`/models.md`) pages directly, so request/response schemas and model versioning are unverified first-hand.
- I found no official typesafe.ai pricing page. The price figure is relayed by secondary sources that agree with each other.

## Q2. Has TypeSafe published a paper, model card, or calibration evaluation (reliability diagrams, ECE, Brier)? Are there independent calibration tests?

### Takeaway
No. TypeSafe has published no paper, technical report, model card, calibration curve, ECE or Brier score. RLCD is defined only at the slogan level. I found no independent calibration study (reliability diagram, ECE or Brier) of Jev.

### Cited Findings
- **[VENDOR]** RLCD = "Reinforcement Learning for Calibrated Decisions, which optimizes for probabilities that match outcomes" — [flaviocopes.com](https://flaviocopes.com/jev/). It "rewards ... that the stated probability matches the actual hit rate" — [innFactory](https://innfactory.ai/en/blog/jev-system-one-model-classifier-not-llm/). It is trained on synthetic data — [Turing Post](https://www.turingpost.com/p/what-is-jev-rlcd); [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/)
- **[INDEPENDENT]** "TypeSafe has not published enough technical detail to separate RLCD's contribution from the model architecture, data, or serving stack." Confidence calculation reportedly differs between the hosted API and an adapter layer (unpublished) — [Turing Post](https://www.turingpost.com/p/what-is-jev-rlcd)
- **[INDEPENDENT]** "As of September 19, 2026, TypeSafe has released no paper, training details or calibration curves for RLCD", and TypeSafe "does not claim RLCD improves Jev's accuracy" — search-result summary of RLCD explainers ([Turing Post](https://www.turingpost.com/p/what-is-jev-rlcd), [jevaiguide.com](https://jevaiguide.com/concepts/rlcd/)). This is aggregator-level evidence but consistent with the docs index.
- **[VENDOR]** The docs index has no model card or calibration page — [llms.txt](https://docs.typesafe.ai/llms.txt)
- **[VENDOR, Cloudflare]** A "Jev Decision Index" exists: a 43-benchmark suite said to be "maintained by Typesafe AI" that "tracks accuracy, latency, and calibration metrics", with a live demo site — [Cloudflare blog](https://blog.cloudflare.com/clef-decision-models/). Cloudflare's published table shows accuracy and latency but no calibration numbers. The Hugging Face card for Clef also lists no ECE or Brier — [HF Cloudflare/clef](https://huggingface.co/Cloudflare/clef)

### Inferences
- The only calibration-related artefact found is the Decision Index, which reportedly tracks calibration. If it is TypeSafe-maintained, it is still a vendor benchmark. Anyone needing ECE or Brier must compute them on their own labelled data. Turing Post advises the same: "test any threshold on your own data".

### Gaps
- I could not locate or read the Jev Decision Index site, so I could not check whether it reports ECE or Brier per model.
- No independent reliability diagrams exist that I could find.

## Q3. What have independent developers and journalists measured (latency, accuracy, cost), and what about the conflicting pricing reports?

### Takeaway
Independent evidence is mostly anecdotal or small-N. Partner quotes (Vercel, Bryo) report 5–20x speed or cost gains. One small public benchmark (n=42) found about 71% task accuracy for Jev at p50 225 ms. Cloudflare measured Jev at about 524 ms median, which is above TypeSafe's "most around 100 ms". The pricing "conflict" is a unit restatement: $0.042 per million is the same as $42 per billion.

### Cited Findings
- **[INDEPENDENT/partner]** Vercel saw "5 to 18 times" faster results. Bryo AI said "10 to 20 times" cheaper than Gemini. Demand briefly knocked the API offline — [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/). A Vercel CEO quote reads "18x faster (p95) and more accurate" — [eesel.ai](https://www.eesel.ai/blog/typesafe-jev-review)
- **[INDEPENDENT]** InfoQ reports:
  - Vercel AI Gateway reached about 13% paid-team adoption within 24 hours ("twice the GPT-5.6 family rate")
  - Netlify, LangChain and Elixir integrations
  - An "OpenChamber analysis of user reports" with median 7x speedup, 30x cost savings and 76 ms latency

  — [InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/)
- **[INDEPENDENT]** Small benchmark, n=42 real coding-agent decisions. Jev-1.13.0 hosted got 71.4% overall (computer-use 10/10, supervision 8/12, triage 12/20) at p50 225 ms including network. Clef-flash 9B (bf16, RTX 3090) got 66.7% at p50 315 ms local. Agreement was 35/42. The author's caveats: single person, subjective gold labels — [dev.to/prodbymarcu](https://dev.to/prodbymarcu/i-benchmarked-cloudflares-new-open-decision-model-against-the-hosted-api-its-trying-to-replace-2ded)
- **[INDEPENDENT, competitor]** Cloudflare's run on the Jev Decision Index:
  - Jev: median 524.1 ms, p95 536.0 ms, BFCL 95.75%, BANKING77 F1 79.74%
  - Clef: median 209.3 ms, BFCL 98.47%, BANKING77 F1 94.20%

  — [Cloudflare blog](https://blog.cloudflare.com/clef-decision-models/). This conflicts with TypeSafe's "most around 100 ms" ([flaviocopes](https://flaviocopes.com/jev/)). Network location probably differs.
- **[INDEPENDENT]** The eesel.ai support trial (284 chats, 100-ticket validation) reported 93% triage accuracy and 100% spam detection with zero false positives — [eesel.ai](https://www.eesel.ai/blog/typesafe-jev-review). The same review cites a "7% factual error rate in generated replies". Jev does not generate text, so that metric must refer to another model in the pipeline. Treat the eesel review cautiously, since it is a vendor-marketing blog.
- **[INDEPENDENT, negative]**
  - HN: the 256-comment launch thread pushed back on "can't hallucinate". The CEO reportedly agreed that Jev can be confidently wrong, and a commenter noted Jev "can still emit a completely wrong valid value" — [eesel/search summary](https://www.eesel.ai/blog/typesafe-jev-review); [InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/); [HN item](https://news.ycombinator.com/item?id=49847030)
  - A dev.to post argues that 25 lines of Python do some showcased tasks for free — [dev.to/jamilxt](https://dev.to/jamilxt/typesafes-jev-costs-400x-less-than-an-llm-these-25-lines-of-python-do-the-same-job-for-free-3ddd)
  - Flavio lists where Jev breaks: counting, dates, math, writing, and "typed output does not guarantee correct routing" — [flaviocopes.com](https://flaviocopes.com/jev/)
- **[INDEPENDENT]** Demo costs: about 1 cent for a 2-minute Minecraft session and about 10 cents for a 15-minute drone simulation. All demos are simulators with no real hardware — [MindStudio](https://www.mindstudio.ai/blog/jev-system-one-model-launch)
- **Pricing reconciliation:** TechCrunch says input is "metered by the billion, not the million" and gives no number — [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/). Other sources give $0.042/M = $42/B ([flaviocopes](https://flaviocopes.com/jev/), [MindStudio](https://www.mindstudio.ai/blog/jev-system-one-model-launch)). The Register prices Clef at $0.24/M, "about 5.7x costlier than Jev's $0.042/M" — [The Register](https://www.theregister.com/ai-and-ml/2026/10/01/cloudflare-tries-to-outplay-jev-with-open-weight-clef-models/5300649)

### Inferences
- The figures are consistent: $42 per billion = $0.042 per million. Any source quoting "$0.042 per billion" or "$42 per million" is a misreading.
- The latency gap (TypeSafe's ~100 ms typical vs 225 ms in the dev.to test vs 524 ms in Cloudflare's run) probably reflects client location and payload size; TypeSafe hosts on the US West Coast. Cloudflare is a competitor and chose the setup.

### Gaps
- I did not read the HN thread directly; HN content above comes via secondary summaries.
- I found no X threads with measured numbers.
- "OpenChamber" (InfoQ) is unidentified and its methodology is unknown.

## Q4. Cloudflare Clef: does it exist, what is it, how does it work, and are there benchmarks or other open re-implementations?

### Takeaway
Clef exists. Cloudflare announced Clef and Clef-flash in late September or early October 2026. Both are Apache-2.0 open-weight decision models on Hugging Face and Workers AI, described as fully Jev-API-compatible. Each pairs a frozen Qwen backbone with LoRA adapters and a joint "schema head" that emits one logit per allowed option. This is a learned head, not raw LM-token logit scoring. Training uses label-smoothed cross-entropy plus Brier loss. All benchmarks are Cloudflare's own, apart from one small dev.to test.

### Cited Findings
- **[VENDOR, Cloudflare]** Two models, Clef and Clef-flash, hosted on Workers AI. Weights are on HF under Apache 2.0. Cloudflare calls them "fully Jev-API compatible" and says Clef "is currently the leader" on the Jev Decision Index. A new RL fine-tuning platform launches alongside them; it starts as FDE-led, with a self-serve plan built from AI Gateway capture, Containers RL sandbox, Trainer and Workers AI BYOM — [Cloudflare blog](https://blog.cloudflare.com/clef-decision-models/)
- **[VENDOR, Cloudflare]** Architecture:
  - Frozen backbones: "Qwen3.8-27B" for Clef and "Qwen3.5-9B" for Clef-flash
  - "Rank-256 low-rank adapters"
  - "Prefill-only pass, then scores the valid schema choices in parallel"
  - Two-stage attention routing in which each valid choice extracts relevant context and fields cross-attend before scoring

  — [Cloudflare blog](https://blog.cloudflare.com/clef-decision-models/). The HF card adds a "joint schema head" that reads final hidden states and "scores all options of all questions jointly", with "one logit per allowed option" and a per-question softmax — [HF Cloudflare/clef](https://huggingface.co/Cloudflare/clef)
- **[VENDOR, Cloudflare]** Training: "label-smoothed cross-entropy for valid schema outputs paired with a Brier loss to refine probability calibration", on "internal synthetic datasets permutating field orders, prompts, and schema structures" — [Cloudflare blog](https://blog.cloudflare.com/clef-decision-models/). The HF card calls the post-training "supervised" — [HF](https://huggingface.co/Cloudflare/clef). The WebFetch summary of the blog also said Cloudflare uses "RLCD" as a secondary target. That may be an artefact of the summarisation tool, since RLCD is TypeSafe's proprietary term. It needs a direct read before being repeated.
- **[VENDOR, Cloudflare]** Benchmarks on the Jev Decision Index:

  | Model | Median latency | p95 latency | BFCL | BANKING77 F1 |
  |---|---|---|---|---|
  | Clef | 209.3 ms | 238.6 ms | 98.47% | 94.20% |
  | Clef-flash | 38.8 ms | 122.4 ms | 98.76% | 90.93% |
  | Jev | 524.1 ms | 536.0 ms | 95.75% | 79.74% |

  The card adds MMLU 90.3%, GSM8K 80.8%, ARC-C 97.7%, and workflow exact-action accuracy of 62.9–86.2% — [Cloudflare blog](https://blog.cloudflare.com/clef-decision-models/); [HF](https://huggingface.co/Cloudflare/clef). The Register notes these are self-reported and await independent verification — [The Register](https://www.theregister.com/ai-and-ml/2026/10/01/cloudflare-tries-to-outplay-jev-with-open-weight-clef-models/5300649)
- **[INDEPENDENT]**
  - Training data is proprietary despite the "open source" label
  - VRAM: about 41 GB for Clef-flash and 85 GB for Clef, at single concurrency and 64k context
  - Workers AI price is $0.24/M tokens
  - Clef handles images and video; Jev is text-only

  — [The Register](https://www.theregister.com/ai-and-ml/2026/10/01/cloudflare-tries-to-outplay-jev-with-open-weight-clef-models/5300649). The blog also contrasts Jev's 32k context with Clef's 64k — [Cloudflare blog](https://blog.cloudflare.com/clef-decision-models/). The HF card instead says the default bounded input is 16,384 tokens — [HF](https://huggingface.co/Cloudflare/clef). This is an unresolved inconsistency.
- **[INDEPENDENT]** In practice, available GGUF quantizations lack the decision head, so running Clef-flash needs PyTorch plus custom code. Results are in Q3: 66.7% vs Jev 71.4% on n=42 — [dev.to/prodbymarcu](https://dev.to/prodbymarcu/i-benchmarked-cloudflares-new-open-decision-model-against-the-hosted-api-its-trying-to-replace-2ded)

### Inferences
- Clef is a reasonable public template for how Jev might work: a frozen LLM backbone, a prefill-only pass, and a classification head over enumerated options trained with a proper scoring rule (Brier). TechCrunch also says Jev is suspected to build on an open-weight LLM. Jev's actual design remains unconfirmed.
- Using a TypeSafe-maintained index as the competitive yardstick cuts both ways. Independent replication is needed.

### Gaps
- **"Qwen3.8-27B" looks odd.** I could not verify that a Qwen3.8 release exists; Qwen3 and Qwen3.5 are the known lines. It may be a real new release or a typo or fabrication propagated across sources (Register, HF card, blog). Check the HF `base_model` link directly.
- I found no other open re-implementations of Jev beyond Clef. The Pi harness that TechCrunch mentions is a model harness, not a decision model.
- No independent calibration (ECE/Brier) measurements exist for Clef either.

## Q5. Who founded TypeSafe, and what prior work might hint at RLCD's design?

### Takeaway
TypeSafe was founded by Diogo Almeida (CEO, ex-OpenAI and Google Brain, InstructGPT co-author), Sasha Sheng (COO, ex-Meta/FAIR) and Erik Gafni (CTO, repeat founder). Almeida's InstructGPT/RLHF background is the obvious lineage for "RL for calibrated decisions". Whether RLCD works like RLHF with a calibration reward (for example a proper scoring rule) is speculation.

### Cited Findings
- **[INDEPENDENT]** The co-founders are Diogo Almeida, Erik Gafni and Sasha Sheng. Almeida is a "former OpenAI researcher who helped build ChatGPT" — [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/); [InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/)
- **[INDEPENDENT]** Roles and backgrounds:
  - Almeida: CEO, ex-Google Brain and OpenAI, InstructGPT contributor
  - Sheng: COO, ex-Meta/FAIR
  - Gafni: CTO, founder of Ravel (multimodal AI for DNA sequencing), early employee at Invitae and Freenome

  — search summary of [TypeSafe team page](https://typesafe.ai/team), [ai.engineer](https://ai.engineer/speakers/diogo-almeida), [The Decoder](https://the-decoder.com/former-openai-researcher-builds-an-ai-model-that-judges-options-instead-of-writing-text/). These pages were not fetched individually.
- Almeida is a listed co-author of "Training language models to follow instructions with human feedback" (Ouyang et al., 2022, InstructGPT) — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **Inconsistency:** MindStudio spells the founder's name "Diego Almeida" — [MindStudio](https://www.mindstudio.ai/blog/jev-system-one-model-launch). Every other source says Diogo.
- **Overclaim:** TechCrunch says Almeida "invented RLHF", and InfoQ says he "co-invented RLHF" ([TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/); [InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/)). RLHF for deep RL is generally credited to Christiano et al. (2017, [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)). His documented contribution is InstructGPT, which applied RLHF to LLMs.

### Inferences
- **[SPECULATION]** By analogy with RLHF and RLVR, RLCD probably rewards a proper scoring rule (log-loss or Brier) over outcomes on synthetic labelled decisions, so that stated probabilities match hit rates. Cloudflare's explicit Brier-loss objective for Clef is consistent with this direction, but it says nothing direct about TypeSafe's method.
- **[SPECULATION]** It is unclear what RL adds over plain supervised training with a proper scoring rule; supervised training already rewards calibration. TypeSafe has not explained this.

### Gaps
- I found no prior papers by Almeida, Sheng or Gafni specifically on calibration.
- I did not fetch the typesafe.ai team page or the TypeSafe GitHub (typesafe-ai.github.io), so I cannot confirm any open-source code from TypeSafe.
