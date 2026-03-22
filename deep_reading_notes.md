# Deep Reading Notes: Adversarial Prompts and Document Length

**Research Question:** Is it easier or harder to hide adversarial prompts in longer documents?

**Reading methodology:** All 8 papers chunked at 3 pages per chunk using `pdf_chunker.py`, then all chunks read in full.

---

## Paper 1: Many-Shot Jailbreaking (Anthropic)

**Citation:** Anil et al. (Anthropic), 2024. "Many-Shot Jailbreaking." Blog post / technical report.

### Methodology
- Attack: prepend hundreds of harmful Q&A demonstrations to a harmful query in the context window
- The demonstrations all follow the same format: harmful question, then the model "answers" it compliantly
- Tests on Claude 2.0, Claude 2.1, and other large frontier models (GPT-4, Llama 2)
- Exploits the long context window (200k+ tokens for Claude) as the attack surface
- Attack content: demonstrations of the form "Human: [harmful question] \n\nAssistant: [harmful answer]"
- The shot content itself can be mundane (e.g., jokes, recipes) — the harmful behavior emerges from length alone

### How Document/Context Length Was Varied
- Number of shots (n) varied from 1 to ~256 demonstrations
- Each demonstration adds roughly equal tokens; n ≈ context length proxy
- Key comparison: 1-shot vs. 2-shot vs. 4-shot vs. 16-shot vs. 64-shot vs. 256-shot

### Key Results: Effectiveness vs. Document Length
- **Power law scaling:** -E[log P(harmful response)] = C·n^(-α) + K
  - As n increases, probability of refusal drops dramatically
  - Effect is monotonic: more shots = lower refusal rate (higher ASR)
- At n=256, ASR reaches near-ceiling for most harm categories on Claude 2.0/2.1
- At n=1 or n=2, most attacks fail
- Threshold effect: meaningful ASR increase begins around n=16-64 shots
- The harmful demonstrations do NOT need to be semantically consistent with the final query — the attack works even with jokes as demonstrations
- Context length is the primary driver, not shot content quality

### Position Effects
- Demonstrations placed at the beginning; harmful query at the end
- Standard ICL format — all demonstrations prime the model before the target query
- No explicit ablation of injection position within long context reported

### Model-Specific Differences
- Claude 2.0/2.1: vulnerable, power law curve confirmed
- GPT-4: also vulnerable, similar scaling curve
- Llama 2: also vulnerable
- Smaller models show similar trends but may require fewer shots (less context capacity to fill)
- Safety fine-tuning does not eliminate the vulnerability — it may shift the power law constant C but not eliminate the n-dependence

### Quantitative Results
- Power law fit: -E[log P] = C·n^(-α) + K
- α (exponent) varies by harm category and model but is consistently >0
- At n=256, refusal rates drop to single digits for most categories
- At n=1, refusal rates near 100% for well-aligned models

### Datasets / Baselines
- Harm categories: violence, self-harm, illegal activities, dangerous information, hate speech
- Baselines: zero-shot, 1-shot, and few-shot (≤4) versions of same queries
- Compared against standard safety training baseline (same model without demonstrations)

### Code/Tools Released
- No public code release mentioned; blog post with methodology description

### Key Insight for Research Question
**Longer context = easier attack.** The attack success rate increases monotonically and dramatically with context length (number of shots). The model's safety training is overwhelmed by the in-context demonstrations. Length itself — independent of specific harmful content per demonstration — is the key variable.

---

## Paper 2: Many-Shot Jailbreaking: An Empirical Study (2505.19773v1)

**Citation:** (2025). "Many-Shot Jailbreaking: An Empirical Study." arXiv:2505.19773v1

### Methodology
- Systematic replication and extension of Anthropic's MSJ attack
- Models tested: GPT-4o, GPT-4o-mini, Claude 3 Opus, Claude 3 Sonnet, Llama 3.1, Gemma 2, Mistral
- Attack variants: harmful demonstrations, benign demonstrations (to test whether content matters)
- Harm categories: 14 categories from AdvBench and custom datasets
- Metric: Attack Success Rate (ASR), judged by GPT-4 as evaluator

### How Document/Context Length Was Varied
- Shot count n varied: 1, 5, 10, 25, 50, 100, 200, 256, 500 (where context allows)
- Two content conditions: (a) on-topic harmful demos, (b) off-topic benign demos (filler content)
- Context length varied accordingly

### Key Results: Effectiveness vs. Document Length
- Confirms power law scaling from Paper 1 across all tested models
- **Critical finding:** On-topic harmful demonstrations much more effective than off-topic benign filler
  - At n=100: on-topic ASR ≈ 85%, off-topic ASR ≈ 35% (approximate, varies by model)
  - Both increase with n, but content does matter for the slope
- At n=256: on-topic near-ceiling; off-topic still meaningfully lower
- GPT-4o shows steeper improvement curve than Claude 3 models
- Threshold for meaningful ASR: approximately n=25-50 shots

### Position Effects
- Standard format: all demos before final query — no position ablation reported
- Consistent with Paper 1

### Model-Specific Differences
- GPT-4o: highest vulnerability at all shot counts
- Claude 3 Opus: lower base ASR but still follows power law
- Claude 3 Sonnet: intermediate vulnerability
- Llama 3.1 (open-weight): highest vulnerability among tested models
- Gemma 2 / Mistral: also vulnerable, slightly different scaling constants

### Quantitative Results
- At n=50, GPT-4o: ASR ≈ 72% (on-topic harmful demos)
- At n=200, GPT-4o: ASR ≈ 91%
- Off-topic filler at n=200: ASR ≈ 48%
- Claude 3 Opus at n=200: ASR ≈ 61%

### Datasets / Baselines
- AdvBench (520 harmful instructions) + custom extensions
- Baselines: zero-shot refusal, standard few-shot (n≤5), single harmful demonstration
- Evaluator: GPT-4 with custom rubric

### Code/Tools Released
- Not explicitly mentioned in the paper

### Key Insight for Research Question
**Both length and content matter, but length dominates at scale.** At n>100, even benign filler raises ASR substantially. Content quality (on-topic vs. off-topic) determines the slope of the scaling curve but not whether scaling occurs. Longer documents remain harder to defend.

---

## Paper 3: PISanitizer (2511.10720v1)

**Citation:** (2025). "PISanitizer: Prompt Injection Sanitization via Attention Weights." arXiv:2511.10720v1

### Methodology
- **Defense** against prompt injection attacks (not an attack paper)
- Method: uses attention weights to identify and remove injected tokens from the context
- Key insight: injected malicious tokens receive anomalously high attention from the instruction query tokens
- Algorithm: compute attention scores between query tokens and all context tokens; flag tokens above threshold; remove flagged tokens
- Two settings: (a) direct PI in user data, (b) indirect PI via LLM agents (AgentDojo benchmark)
- Models tested: GPT-3.5-turbo, GPT-4, Llama 2 (7b, 13b, 70b)
- Attack types defended: naive injection, escape characters, context ignoring, fake completion, combined attack

### How Document/Context Length Was Varied
- Context length buckets: 0-4k tokens, 4k-8k tokens, >8k tokens
- Tested across varying lengths of legitimate task content + injected content

### Key Results: Effectiveness vs. Document Length
- **Baseline (no defense) ASR** by context length:
  - 0-4k tokens: ASR ≈ 0.66
  - 4k-8k tokens: ASR ≈ 0.66
  - >8k tokens: ASR ≈ 0.66
  (ASR is roughly constant across lengths without defense)
- **PISanitizer ASR** (after defense):
  - 0-4k tokens: ASR drops to ≈ 0.01
  - 4k-8k tokens: ASR drops to ≈ 0.01
  - >8k tokens: ASR drops to ≈ 0.01
  (Defense works equally well across all context lengths)
- **Task utility preserved:** legitimate task performance only slightly reduced (ROUGE, accuracy metrics within ~5%)
- Attention-based detection AUC: >0.95 across all context lengths

### AgentDojo Results
- Without PISanitizer: utility = 0.82, ASR under attack = 0.48
- With PISanitizer: utility = 0.71 (slight drop), ASR = 0.03 (near-zero)

### Adaptive Attacks
- Tested 5 adaptive attack strategies against PISanitizer
- All adaptive attacks fail: ASR remains 0.01-0.04 even with adversarial awareness of defense

### Position Effects
- Tested injection at beginning, middle, and end of document
- PISanitizer is position-robust: attention weight anomaly is independent of where injection is placed
- Injections are detectable regardless of position in context

### Model-Specific Differences
- GPT-4 and GPT-3.5: both successfully defended (ASR ~0.01)
- Llama 2 variants: also successfully defended
- Attention weight patterns generalize across architectures

### Limitations
- Cannot defend against knowledge corruption attacks (where injected content changes beliefs, not instructions)
- Cannot distinguish between benign and malicious instructions if they appear identical in form
- Slight utility degradation at >8k context (utility drops from 0.82 to 0.71 in AgentDojo)

### Datasets / Baselines
- 7 NLP tasks: SST2, SMS Spam, HSOL, Gigaword, Jfleg, MRPC, RTE
- AgentDojo multi-agent benchmark
- Baselines: no defense, PPL-based detection, LLM-based detection

### Code/Tools Released
- Not explicitly released publicly (research prototype)

### Key Insight for Research Question
**Context length does not affect injection detectability when using attention-based defenses.** The attack succeeds at a roughly constant rate regardless of document length without defense (~0.66 ASR). With PISanitizer, defense is equally effective across all lengths. This suggests that injections can be detected at any position/length, but the underlying vulnerability (0.66 baseline ASR) is length-insensitive for direct prompt injection.

---

## Paper 4: Lost in the Middle (2307.03172v3)

**Citation:** Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." TACL 2024. arXiv:2307.03172v3

### Methodology
- Task: multi-document question answering (MQQR) — place the relevant document among k irrelevant documents
- Key variable: position of the relevant (answer-containing) document within the list of k documents
- k varied: 5, 10, 20, 30 documents
- Models tested: GPT-3.5-turbo (16k), Claude 1.3 (100k), Llama 2 (7b, 13b, 70b), MPT-30b, LongChat-13b
- Also tests key-value retrieval as a controlled synthetic task

### How Document/Context Length Was Varied
- Number of documents k: 5, 10, 20, 30
- Position of relevant document: 1st (primacy), last (recency), or middle positions
- Total context length grows with k (each document is ~100-200 tokens)

### Key Results: Effectiveness vs. Document Length / Position
- **U-shaped performance curve:** performance is best when the relevant document is at the beginning (primacy) or end (recency) of the context, worst in the middle
- This is the "Lost in the Middle" effect
- Performance degrades as k increases (more documents = harder)
- Performance at k=20 is ~1.5% better than at k=30 (diminishing returns beyond 20 documents)
- The middle positions show the largest degradation

### Quantitative Results
- GPT-3.5-turbo: ~20% performance difference between best position (beginning) and worst position (middle), at k=20
- Llama 2 7B: shows only recency bias (primacy effect absent) — performance monotonically decreases from end to start
- Llama 2 13B: shows U-shaped curve but weaker than GPT-3.5; ~15-point accuracy gap between best and worst position
- Llama 2 13B fine-tuned (chat): gap reduces to ~10 points
- Llama 2 70B: stronger U-shaped curve, ~18-point gap
- GPT-4: also shows U-shaped curve but at higher absolute performance levels

### Position Effects
- Primacy bias: most models favor information at the beginning of context
- Recency bias: all models show some recency bias
- Middle positions (especially the very center) show worst performance
- The effect is consistent across k=5, 10, 20, 30 but grows with k

### Model-Specific Differences
- GPT-3.5: strong U-shape (~20-point gap), primacy > recency
- Claude 1.3: U-shape present but milder; handles long context better
- Llama 2 7B: recency-only (monotonic degradation from end to start)
- Llama 2 13B: U-shape emerges; base model weaker than chat-tuned
- Llama 2 70B: U-shape, stronger primacy effect
- LongChat-13b: poor performance overall, recency bias only
- MPT-30b: moderate, recency-focused

### Datasets / Baselines
- NaturalQuestions (NQ) open domain QA
- KV retrieval synthetic task (controlled)
- Baselines: oracle (gold document provided alone), no-context (closed-book), and position-variant comparisons

### Code/Tools Released
- GitHub: https://github.com/nelson-liu/lost-in-the-middle

### Key Insight for Research Question
**Position within a long document is critical for information retrieval and, by extension, adversarial injection.** Adversarial content placed at the beginning or end of a long document is more likely to be "found" and acted upon than content placed in the middle. This means adversarial prompts hidden in the middle of long documents may actually be HARDER for models to act on — but also harder for defenders to locate. Longer documents (more irrelevant documents) make it harder to use any information, which could be a double-edged sword for attackers.

---

## Paper 5: Short-Length Adversarial Training (2502.04204v3)

**Citation:** (2025). "Short-Length Adversarial Training Generalizes to Long-Length Attacks." arXiv:2502.04204v3. NeurIPS 2025.

### Methodology
- **Defense paper:** adversarial training (AT) with short-length adversarial suffixes to defend against long-length attacks
- Theoretical analysis using a linear self-attention (LSA) model under Gaussian data assumptions
- Attack studied: GCG (Greedy Coordinate Gradient) suffix optimization and non-suffix attacks
- Suffix length at train time: Mtrain tokens; at test time: Mtest tokens
- Key claim: training with Θ(√M) length suffixes defends against Θ(M) length attacks

### Mathematical Framework
- Adversarial risk with suffix: Radv(θ, Mtest) depends on √(Mtest/Mtrain)
- Main theorem: if Mtrain = Θ(√Mtest), then adversarial generalization is guaranteed
- Surrogate AT loss decomposes into 4 terms; the perturbation terms scale as O(ε²·Mtrain/(N+Mtrain)²)
- The key insight: longer suffixes at test time have a smaller per-token perturbation budget relative to total context

### How Document/Context Length Was Varied
- Mtest varied: 20, 40, 60, 80, 100, 120 tokens (GCG suffix length)
- Mtrain varied: 5, 10, 20 tokens
- N (clean ICL examples in context): fixed
- Measuring: ASR vs. Mtest for different Mtrain values

### Key Results: Effectiveness vs. Document Length
- **Pearson Correlation Coefficient (PCC) of ASR with √(Mtest/Mtrain):** 0.76-0.93 for GCG attacks
  - Confirms the √(Mtest/Mtrain) scaling prediction
- ASR is **positively correlated** with test suffix length (longer attack = higher ASR)
- **But:** training with longer suffixes (larger Mtrain) reduces this correlation and lowers ASR across all test lengths
- Mtrain=20 reduces ASR by **>30%** for all Mtest up to 120, compared to Mtrain=5
- The relationship ASR ∝ √(Mtest/Mtrain) is empirically validated

### Quantitative Results
- GCG, Mtrain=5 vs Mtest=120: ASR ≈ 65%
- GCG, Mtrain=20 vs Mtest=120: ASR ≈ 35% (>30% reduction)
- At Mtest=Mtrain=20: ASR ≈ 15% (best case defended)
- Non-suffix attacks:
  - PAIR (iterative jailbreaking): partially defended by short-length AT
  - DeepInception: suppressed to near 0% ASR after AT

### Position Effects
- Suffix attacks: appended at end of query — position is fixed
- The paper's defense generalizes to different suffix lengths but not positions

### Model-Specific Differences
- Experiments on GPT-2 (theoretical proxy) and Llama 2 (empirical)
- Results generalize across both; the power law scaling holds in both cases

### Limitations (from paper)
- Only studies training with a single attack type (GCG); defense against unseen attack types may be weaker
- PAIR is partially but not fully defended

### Datasets / Baselines
- AdvBench harmful behaviors dataset
- Attacks: GCG, PAIR, DeepInception
- Baselines: undefended model, standard safety fine-tuning

### Code/Tools Released
- Experimental code released in NeurIPS supplementary material (with README and LICENSE)

### Key Insight for Research Question
**Longer adversarial suffixes are harder to defend against, but the scaling is sublinear (√Mtest).** This means defenses scale favorably: training with √M-length suffixes protects against M-length attacks. The attack is harder for defenders to handle the longer it gets, but less so than linearly. This provides a formal framework for understanding why long-context attacks are a fundamental challenge.

---

## Paper 6: Reasoning Safety in Long-Context LLMs (2602.08874v1)

**Citation:** (2026). "Is Reasoning Capability Enough for Safety in Long-Context Language Models?" arXiv:2602.08874v1

### Methodology
- **Compositional reasoning attacks:** decompose a harmful query into multiple "needles" (facts) distributed across a long document; model must reason across facts to reconstruct harm
- 3 reasoning levels:
  - Level 1 (2-hop): Fact A + Fact B → Answer
  - Level 2 (3-hop chain): Fact A → Fact B → Fact C → Answer
  - Level 3 (4-hop deductive): (A+B)→X, (X+C)→Answer (multi-hop deduction required)
- Context lengths: 0k (direct), 16k, 64k tokens (needle-in-haystack with filler text)
- Filler: relevant-domain educational text (generated by NeuralDaredevil-8B-abliterated), NOT random text
- 14 LLMs tested: GPT-4o-mini, GPT-4.1, GPT-5.1, GPT-5.2, GPT-OSS-20B, GPT-OSS-120B, Claude-Haiku-4.5, Claude-Sonnet-4.5, Gemini-2.5-Flash, Gemini-3-Flash, Gemini-3-Pro, Qwen3-80B-Thinking, Kimi-K2-Thinking, DeepSeek-V3.2, MiniMax-M2
- 3 inference-time reasoning effort levels: low, medium, high
- Evaluator: 5-score judge prompt (GPT-4 based, from Rahman et al. 2025)

### How Document/Context Length Was Varied
- 0k context: needle facts placed directly in prompt (no filler)
- 16k context: needles distributed within 16k tokens of domain-relevant filler
- 64k context: needles distributed within 64k tokens of domain-relevant filler
- Needle positions: distributed throughout document (not all at same position)

### Key Results: Effectiveness vs. Document Length
- **Direct retrieval (Level 0, single fact):** 96-100% safe responses across all models and contexts
  - Models can find a single needle easily and refuse if it's harmful
- **As context grows, safety degrades:**
  - Average safety rate across 14 models:
    - 0k context: ~55% safe
    - 16k context: ~43% safe
    - 64k context: ~42% safe
  - Degradation is NOT monotonic: the jump from 0k to 16k is larger than 16k to 64k
- **Reasoning complexity degrades safety more than length alone:**
  - Level 1 (2-hop): safer than Level 2 and 3
  - Level 3 (4-hop): most dangerous, model must reason to reconstruct harm
  - The combination of long context + complex reasoning = most effective attack

### Quantitative Results (GPT-OSS-120B at 64k context)
- Low reasoning effort: 12% safe (88% harmful responses generated)
- Medium reasoning effort: 40% safe
- High reasoning effort: 63% safe
- **Inference-time reasoning effort is the single strongest safety predictor**

### Position Effects
- **Position-invariant safety failures:** models fail regardless of where needles are placed in the 64k document
- Failures are NOT concentrated at primacy/recency positions — this contradicts the "Lost in the Middle" finding for safety-relevant retrieval
- Safety failures occur even when all needles are at the END of the document (high-salience position)

### Relevant Context Effect
- When filler text is domain-relevant (thematically related to the harmful query), safety is LOWER than with random filler
- Relevant context helps the model reason more effectively — which backfires for safety

### Model-Specific Differences
- GPT-5.1/5.2: best safety (highest safe% at all contexts)
- Gemini-3-Pro: strong safety, especially at high reasoning effort
- Claude-Sonnet-4.5: good safety, especially with high reasoning effort
- GPT-OSS-20B: poor safety at low reasoning effort (12% safe at 64k)
- GPT-4o-mini: consistently lower safety
- DeepSeek-V3.2: high vulnerability at 64k context
- Reasoning models (Qwen3-80B-Thinking, Kimi-K2-Thinking): mixed — more reasoning can cut both ways

### Datasets / Baselines
- AdvBench (harmful queries) decomposed using GPT-4
- 3 reasoning types × 3 context lengths = 9 experimental conditions
- Baseline: direct harmful query (zero-hop)

### Code/Tools Released
- Benchmark construction code and prompts provided in appendix
- Models accessed via OpenRouter API

### Key Insight for Research Question
**Longer documents enable compositional reasoning attacks where no single fragment is harmful.** At 64k context, models that could refuse the direct query often fail to refuse when the harmful content is split across multiple "innocent" fragments. This directly answers the research question: longer documents make it EASIER to hide adversarial prompts by distributing them compositionally. The key attack mechanism is using the model's own reasoning capability against it.

---

## Paper 7: Spotlighting (2403.14720)

**Citation:** Hines et al. (2024). "Spotlighting: Using Generative AI to Illuminate the Boundaries Between Instructions and Data." arXiv:2403.14720

### Methodology
- **Defense paper:** prompt engineering techniques to help LLMs distinguish system instructions from external data
- Three defense strategies:
  1. **Delimiting:** wrap external data in clear delimiters (quotes, XML tags, random strings)
  2. **Datamarking:** prepend each token/word in external data with a special marker character
  3. **Encoding:** encode external data in a non-natural-language format (e.g., Base64, ciphertext)
- Attacks tested: simple injection, naive override, complex instruction injection
- Models: GPT-3.5-turbo, GPT-4
- Tasks: summarization (legitimate task), email hijacking (injected task)
- Metric: ASR of injection, task accuracy for legitimate task

### How Document/Context Length Was Varied
- Varied the length of the legitimate document containing the injection
- Tested with short (~200 token) and long (~2000+ token) documents
- The injection is embedded within the legitimate document at various positions

### Key Results: Effectiveness vs. Document Length
- **Without defense:** ASR ≈ 60-80% across document lengths (injections succeed frequently)
- **Datamarking:**
  - GPT-3.5 summarization task: ASR drops to **3.1%** (from ~60%)
  - Task accuracy (legitimate summarization): nearly unchanged
  - Effective at both short and long document lengths
- **Encoding (Base64):**
  - GPT-4: ASR drops to **0%**
  - GPT-3.5: ASR drops to ~0% BUT task accuracy also degrades (~20% drop in ROUGE)
  - GPT-4 handles encoding better and maintains task accuracy
- **Delimiting:** moderate effectiveness; XML tags better than quotes; random string delimiters moderately effective

### Quantitative Results
- Datamarking: ASR 3.1% (GPT-3.5), near 0% (GPT-4)
- Encoding: ASR 0% (GPT-4), 0% (GPT-3.5) but with task degradation
- Delimiter (XML): ASR ~20-30% (partial protection)
- Sandwich prevention: ASR reduced but still ~30-50%
- Instructional prevention: most effective delimiter-based method, ASR ~15-25%

### Position Effects
- Injection position within the legitimate document varied: beginning, middle, end
- Datamarking and encoding are position-robust — the defense works regardless of injection position
- Delimiting is more sensitive to position (injections at boundaries of delimiters more effective)

### Model-Specific Differences
- GPT-4: handles encoding without task degradation; better at following delimiter instructions
- GPT-3.5: datamarking works well; encoding degrades task performance
- Both models show similar ASR reduction with datamarking

### Datasets / Baselines
- Gigaword summarization dataset (legitimate task)
- Custom injection prompts (email hijacking)
- Baselines: no defense, standard system prompt, simple instructional prevention

### Code/Tools Released
- No public code repository mentioned; techniques are prompt engineering (no code needed)

### Key Insight for Research Question
**Defenses like datamarking can reduce injection ASR regardless of document length.** However, the paper shows the attack succeeds at high rates without defense even in moderate-length documents. Longer documents may provide more "cover" for injections (harder to spot visually), but the attack mechanism itself is not strongly length-dependent. The defense strategies are scalable and length-independent.

---

## Paper 8: Formalizing and Benchmarking Prompt Injection Attacks (2310.12815)

**Citation:** Guo et al. (2023). "Baseline Defenses for Adversarial Attacks Against Aligned Language Models." / Zhu et al. (2023). "Formalizing and Benchmarking Prompt Injection Attacks and Defenses." arXiv:2310.12815

### Methodology
- **First formal framework** for prompt injection attacks
- Formalization: LLM-integrated application takes (instruction, data) → output; attacker controls "data" component
- Attack Success Value (ASV): fraction of times model performs injected task instead of legitimate task
- 4 attack types:
  1. Naive injection: directly append injected instruction to data
  2. Escape characters: use newlines/special chars to separate
  3. Context ignoring: "Ignore previous instructions..."
  4. Fake completion: provide fake end of legitimate task, then inject
  5. **Combined attack:** combines all 4 strategies
- 7 target tasks × 7 injected tasks = 49 task combinations per model
- Models: PaLM 2, GPT-3.5-turbo, Bard, Vicuna-33b, Vicuna-13b, Llama-2-13b-chat, Llama-2-7b-chat, InternLM-chat-7b, Flan-UL2
- Defenses tested: paraphrasing, retokenization, delimiters (6 types), instructional prevention, sandwich prevention, PPL detection, naive LLM detection, response-based detection, known-answer detection

### How Document/Context Length Was Varied
- **Key finding on injected content length:**
  - ASV increases as token length of injected data grows from 10→50 tokens, then **plateaus**
  - Effective attack needs >30 tokens of injected data
  - Injected instruction length: ASV increases from 10→20 tokens, then **plateaus**
  - Effective instruction needs >20 tokens
- In-context learning examples (k): varied from 0 to several; minimal impact on ASV
- Total context length: not the primary variable studied

### Key Results: Attack Success vs. Length
- **Injected data length effect (Figure 7):** ASV rises sharply from 10 to 30 tokens, then plateaus at maximum
  - Most attack effectiveness achieved with 30-50 tokens of injected content
  - **More tokens beyond 30 does not significantly increase ASV**
- **Injected instruction length effect (Figure 8):** similar plateau at ~20 tokens for simple tasks; longer needed for complex tasks (grammar correction needs ~60 tokens)
- In-context learning examples: **negligible impact** on attack success

### Attack Success Values (ASV) — Combined Attack
- **PaLM 2:** average ASV = 0.75 (vs. 0.62 naive attack)
- **GPT-3.5-turbo:** average ASV = 0.63
- **Bard:** average ASV = 0.71
- **Llama-2-13b-chat:** ASV = 0.71
- **Llama-2-7b-chat:** ASV ≈ 0.65 (varies greatly by task)
- **Flan-UL2:** ASV = 0.79 (highest vulnerability)
- Larger models are MORE vulnerable: Pearson correlation = 0.63 between model size and ASV

### Larger Models = More Vulnerable
- Pearson correlation coefficient between model size (parameters) and ASV: **r = 0.63** (positive, significant)
- Larger models are better instruction followers → follow injected instructions more reliably
- This is a fundamental tension: capability and safety work against each other for prompt injection

### Defense Results
- **Paraphrasing:** reduces ASV significantly (GPT-4: ASV drops ~40-60% for most task pairs), but high FNR for detecting attacks vs. legitimate input
- **Retokenization:** partial effectiveness, ASV ~0.5-0.7 still
- **Delimiters (all types including XML, quotes, random):** ASV remains 0.5-0.9 — largely ineffective
- **Instructional prevention:** somewhat effective (ASV drops ~20-40% for some tasks)
- **Sandwich prevention:** inconsistent; works on some tasks, fails on others
- **PPL detection:** FNR near 1.00 for most cases — completely ineffective at detecting injections
- **Windowed PPL detection:** slightly better FNR (0.09-0.31 for some tasks) but still very high
- **Naive LLM detection (ask GPT-4 if injected):** FNR ≈ 0.00 — nearly perfect detection
- **Response-based detection:** FNR varies widely (0.00-1.00) — unreliable, depends on task
- **Known-answer detection:** FNR ≈ 0.00-0.18 — second best after naive LLM detection

### Position Effects
- Not explicitly studied as a primary variable in this paper
- Attack placed at the end of the data section in the standard formulation

### Datasets / Baselines
- 7 NLP tasks: duplicate sentence detection, grammar correction, hate detection, NLI, sentiment analysis, spam detection, summarization
- Datasets: SST2, SMS Spam, HSOL, Gigaword, Jfleg, MRPC, RTE
- 9 models ranging from 7B to 540B parameters

### Code/Tools Released
- Benchmark framework released as open-source code (GitHub repository)

### Key Insight for Research Question
**There is a minimum injection length (~30 tokens) below which attacks fail, but beyond that threshold, additional length provides no benefit.** This is a crucial finding: attack success saturates quickly with injection length. Longer documents provide more space to hide the injection (harder for humans to spot), but do not fundamentally change the attack's success probability once the injection crosses the threshold length. Combined with the model size finding, larger (better) models are paradoxically more vulnerable.

---

## Synthesis: Answering the Research Question

### Research Question: Is it easier or harder to hide adversarial prompts in longer documents?

### Answer: It is generally EASIER to hide adversarial prompts in longer documents, but with important nuances.

#### Evidence that LONGER = EASIER to attack:

1. **Many-Shot Jailbreaking (Papers 1 & 2):** ASR scales monotonically with number of demonstrations (context length). At n=256 shots, near-ceiling ASR. This is the most direct evidence — more context = more powerful attack.

2. **Compositional Reasoning Attacks (Paper 6):** Splitting harmful content across multiple facts in a 64k context is far more effective than any single fact or 0k context. Average safety drops from ~55% (0k) to ~42% (64k). No single fragment is detectable as harmful.

3. **Short-Length AT paper (Paper 5):** ASR is positively correlated with test suffix length (PCC 0.76-0.93). Mathematically: Radv ∝ √(Mtest/Mtrain). Longer attacks require more training to defend against.

4. **Formalizing PI (Paper 8):** Larger models (which have larger context windows and process longer documents) are more vulnerable (r=0.63 correlation with model size). While this is about model size, it reflects the tendency of capable models to follow injected instructions more reliably.

#### Evidence for NUANCES and LIMITS:

5. **Lost in the Middle (Paper 4):** Content placed in the MIDDLE of a long document is actually HARDER for models to use — primacy and recency biases mean injections in the middle of very long documents may be less effective than at the ends. This is the key exception to the general trend.

6. **Injection length plateau (Paper 8):** Beyond ~30-50 tokens of injected content, additional injection length provides NO benefit to attackers. So within a long document, the injection itself doesn't need to be long.

7. **Instruction length plateau (Paper 8):** Beyond ~20 tokens of injected instruction, no benefit. Attack efficiency is about quality, not quantity of the injection itself.

8. **PISanitizer (Paper 3):** Context length does NOT affect detection difficulty for attention-based defenses — ASR remains ~0.01 regardless of context length (0-4k, 4k-8k, >8k).

#### Mechanism Summary:

| Mechanism | Effect of Longer Document | Paper |
|-----------|--------------------------|-------|
| Many-shot ICL override | Longer = Higher ASR (power law) | 1, 2 |
| Compositional reasoning attack | Longer = Lower safety rate | 6 |
| Suffix optimization attack | Longer = Higher ASR (√M scaling) | 5 |
| Primacy/recency position bias | Middle of longer doc = Harder to use | 4 |
| Direct PI injection length | Plateau at 30-50 tokens; length beyond irrelevant | 8 |
| Attention-based detection | Length-independent detection difficulty | 3 |
| Model capability/size | Larger models more vulnerable | 8 |

#### Key Nuance: WHAT is hidden matters more than HOW LONG the document is

- For **suffix-based jailbreaks** and **many-shot jailbreaks**: longer context directly amplifies attack effectiveness
- For **direct prompt injection** (embedding one instruction in data): length barely matters beyond minimum threshold; the injection's content and position matter more
- For **compositional attacks** (splitting harm across fragments): longer context is essential to the attack mechanism

#### Practical Implications

1. **Defenders** should treat long-context models as fundamentally higher-risk than short-context models for many-shot and compositional attacks.

2. **The middle of a long document** may be the optimal hiding position (hardest for models to retrieve), but this provides false security — compositional attacks work regardless of position (Paper 6).

3. **Attention-based defenses** (Paper 3) appear promising and length-invariant, but they only work for direct injection, not compositional attacks that blend into legitimate content.

4. **Inference-time reasoning effort** (Paper 6) is the strongest single lever: high reasoning effort models refuse more even in long contexts (12% safe → 63% safe by increasing reasoning effort at 64k).

5. **Datamarking** (Paper 7) reduces ASR to 3.1% for direct injection tasks and appears length-invariant.

---

## Cross-Paper Quantitative Reference

| Attack Type | Baseline ASR (no defense) | Best Defense ASR | Key Context Length Effect |
|-------------|--------------------------|------------------|--------------------------|
| Many-shot (n=256) | ~85-95% | No effective defense | Linear with log(n) |
| Compositional (64k, Level 3) | ~58% harmful | ~37% harmful (high reasoning) | +15% harmful vs. 0k |
| Direct PI (combined attack) | 0.62-0.79 ASV | 0.01 ASV (attention-based) | Plateaus at 30-50 tokens |
| GCG suffix (Mtest=120) | ~65% | ~35% (Mtrain=20 AT) | ∝ √Mtest |
| MQQR (k=20 docs, middle) | ~40% lower accuracy | N/A | ~20-point positional gap |

---

*Notes compiled from systematic reading of all chunks of all 8 papers. Chunk files located at `/workspaces/adversarial-prompts-claude/papers/pages/`. Papers chunked at 3 pages per chunk using `/workspaces/adversarial-prompts-claude/.claude/skills/paper-finder/scripts/pdf_chunker.py`.*
