# Resources Catalog

## Summary
This document catalogs all resources gathered for investigating whether it is easier or harder to hide adversarial prompts in longer documents.

## Papers
Total papers downloaded: 20 unique papers (31 files including duplicates)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Many-Shot Jailbreaking | Anil et al. (Anthropic) | 2024 | papers/anthropic_many_shot_jailbreaking.pdf | Power law: ASR scales with context length |
| What Really Matters in Many-Shot Attacks | - | 2025 | papers/2505.19773v1_many_shot_empirical_study.pdf | Empirical replication, content vs length effects |
| Is Reasoning Enough for Safety in Long-Context? | - | 2026 | papers/2602.08874v1_reasoning_safety_long_context.pdf | Compositional attacks in 64k context |
| Lost in the Middle | Liu et al. | 2023 | papers/2307.03172v3_lost_in_the_middle.pdf | U-shaped position effect |
| Short-Length Adversarial Training | - | 2025 | papers/2502.04204v3_short_length_adv_training.pdf | ASR ∝ √(Mtest/Mtrain) |
| PISanitizer | Geng et al. | 2025 | papers/2511.10720v1_pisanitizer.pdf | Attention-based defense for long contexts |
| Formalizing Prompt Injection | Zhu et al. | 2023 | papers/2310.12815_formalizing_benchmarking_pi.pdf | Formal framework, injection length plateau |
| Spotlighting | Hines et al. | 2024 | papers/2403.14720_spotlighting.pdf | Datamarking defense |
| Indirect Prompt Injection | Greshake et al. | 2023 | papers/2302.12173_indirect_prompt_injection_greshake.pdf | Foundational indirect PI paper |
| Ignore Previous Prompt | Perez & Ribeiro | 2022 | papers/2211.09527_ignore_previous_prompt.pdf | First prompt injection study |
| StruQ | Chen et al. | 2024 | papers/2402.06363_struq.pdf | Structured query defense |
| InjecAgent | Zhan et al. | 2024 | papers/2403.02691_injecagent.pdf | Agent-based injection benchmark |
| Neural Exec | Pasquini et al. | 2024 | papers/2403.03792_neural_exec.pdf | Learned execution triggers |
| PI Detection and Removal | Chen et al. | 2025 | papers/2502.16580_pi_detection_removal.pdf | Indirect PI detection |
| PI against LLM-Integrated Apps | Liu et al. | 2023 | papers/2306.05499_pi_llm_integrated_liu.pdf | Houyi attack framework |
| SecAlign | Wu et al. | 2024 | papers/2410.05451_secalign.pdf | Preference optimization defense |
| Hidden Prompts in Manuscripts | - | 2025 | papers/2507.06185_hidden_prompts_manuscripts.pdf | Academic peer review attacks |
| PI Breaks MCQ | - | 2025 | papers/2508.13214_pi_mcq.pdf | Simple injection breaks MCQ tasks |
| PI Attack on LLM-as-Judge | - | 2024 | papers/2403.17710_pi_llm_judge.pdf | Optimization-based judge attacks |
| BIPIA Benchmark | Yi et al. | 2023 | papers/2312.14197_bipia_yi.pdf | Position-aware indirect PI benchmark |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 7

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| deepset/prompt-injections | HuggingFace | 662 samples | Binary classification | datasets/prompt-injections/ | Labeled injection payloads |
| protectai/prompt-injection-validation | HuggingFace | 3,227 samples | Classification | datasets/protectai-prompt-injection-validation/ | Includes 125 BIPIA samples |
| BIPIA benchmark data | code/bipia/ | ~1,200 contexts + 25 attacks | Indirect injection | datasets/bipia/ | Position testing (begin/middle/end) |
| ethz-spylab/ctf-satml24 | HuggingFace | 44 defenses | CTF defense | datasets/ctf-satml24/ | Real CTF defense prompts |
| openai/mrcr | HuggingFace | 2,400 samples (~1.4GB) | Long-context retrieval | datasets/mrcr/ | Adapt for adversarial testing |
| JailbreakBench/JBB-Behaviors | HuggingFace | 200 behaviors | Harm benchmarking | datasets/jailbreakbench/ | Standardized harm targets |
| TrustAIRLab/in-the-wild-jailbreak | HuggingFace | 1,405 prompts | Jailbreak analysis | datasets/in-the-wild-jailbreak/ | Real multi-platform jailbreaks |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 6

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| BIPIA | Yi et al. | Position-based indirect injection benchmark | code/bipia/ | Begin/middle/end position testing |
| open-prompt-injection | Zhu et al. | 5 attacks × 10 defenses framework | code/open-prompt-injection/ | ASR metric, comprehensive benchmark |
| needle-haystack | Kamradt | Length × depth heatmap testing | code/needle-haystack/ | **Most directly useful** for our experiments |
| ctf-platform | Debenedetti et al. | Real CTF platform | code/ctf-platform/ | Prompt extraction/injection challenges |
| universal-prompt-injection | Liu et al. | Gradient-based attack generation | code/universal-prompt-injection/ | Optimized universal attacks |
| PISanitizer | Geng et al. | Long-context injection defense | code/pisanitizer/ | Attention-based detection |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder with diligent mode for two queries: "adversarial prompts hidden in long documents prompt injection" and "long context window attacks LLM document length prompt injection effectiveness"
2. Downloaded 119 + 55 papers from Semantic Scholar, filtered to relevance >= 2
3. Deep-read 8 most relevant papers using PDF chunker (all chunks)
4. Searched HuggingFace Hub for prompt injection datasets
5. Searched GitHub for benchmark implementations

### Selection Criteria
- Prioritized papers studying length/position effects on prompt injection (Papers 1-6)
- Included foundational prompt injection papers for context (Papers 9-10)
- Selected datasets with labeled injection data and position metadata
- Chose code repos that support length × position experimental design

### Challenges Encountered
- Many-shot Jailbreaking paper not on arXiv (hosted on Anthropic CDN)
- Several paper-finder results had duplicate entries under different names
- BIPIA full dataset requires running code to generate; partial data available via ProtectAI

### Gaps and Workarounds
- No single dataset tests injection effectiveness across multiple document lengths — will need to construct this using needle-haystack framework
- AdvBench not directly downloadable as standalone dataset — available through JailbreakBench and various code repos

## Recommendations for Experiment Design

### Primary Approach: Needle-in-Haystack for Adversarial Prompts
Adapt the needle-haystack framework (code/needle-haystack/) to test adversarial prompts:
1. Replace the "needle" (factual statement) with an adversarial instruction
2. Vary document length from 1k to 128k tokens
3. Vary injection position from 0% to 100% depth
4. Measure ASR instead of retrieval accuracy
5. Generate a 2D heatmap: ASR(length, position)

### Primary Datasets
1. **JailbreakBench** (200 behaviors) — standardized harm targets for injection content
2. **BIPIA** — pre-built position-aware injection framework
3. **Paul Graham essays** or similar — clean filler text for haystack documents

### Baseline Methods
1. Direct injection (no hiding) — upper bound on ASR
2. Random position injection — baseline for position effects
3. Many-shot jailbreaking at fixed positions — compare against literature

### Evaluation
1. ASR judged by GPT-4 or Claude (automated)
2. Test 3+ models: Claude, GPT-4, Llama 3+
3. Statistical significance: 50+ trials per condition
