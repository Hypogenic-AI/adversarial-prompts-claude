# Literature Review: Adversarial Prompts in Long Documents

## Research Question
Is it easier or harder to hide adversarial prompts in longer documents?

## Research Area Overview

Prompt injection attacks manipulate LLMs by embedding malicious instructions within data that the model processes. As LLMs gain longer context windows (100k-1M+ tokens), a critical question emerges: does document length help attackers (more space to hide) or hurt them (more noise to overcome)? This review synthesizes findings from 20 papers spanning 2022-2026.

## Answer Summary

**Longer documents generally make adversarial prompts EASIER to hide and MORE effective**, with three important nuances:

1. **Many-shot attacks scale via power law** — ASR increases monotonically with context length
2. **Compositional attacks exploit length** — harmful content split across fragments evades safety
3. **Position matters** — "Lost in the Middle" effect means injections at document edges are most effective
4. **Injection length saturates** — beyond ~30 tokens, more injection text doesn't help

## Key Papers

### Paper 1: Many-Shot Jailbreaking (Anil et al., Anthropic, 2024)
- **Key Contribution**: Demonstrated that prepending hundreds of harmful Q&A demonstrations in long contexts jailbreaks LLMs
- **Methodology**: Vary number of demonstrations (1-256) prepended to harmful queries
- **Results**: Power law scaling: -E[log P(harmful)] = C·n^(-α) + K. At n=256, near-ceiling ASR across Claude 2.0/2.1, GPT-4, Llama 2
- **Key Finding**: Context length is the primary driver — even benign demonstrations raise ASR at scale
- **Relevance**: Most direct evidence that longer contexts enable attacks

### Paper 2: What Really Matters in Many-Shot Attacks (2025, arXiv:2505.19773)
- **Key Contribution**: Empirical replication across 7 models; disentangles content vs length effects
- **Results**: At n=200, on-topic ASR ≈ 91% (GPT-4o), off-topic ≈ 48%. Both scale with n
- **Key Finding**: Content quality determines slope, but length dominates at scale

### Paper 3: Is Reasoning Capability Enough for Safety in Long-Context LLMs? (2026, arXiv:2602.08874)
- **Key Contribution**: Compositional reasoning attacks — split harmful query into "innocent" fact fragments
- **Methodology**: 2-4 hop reasoning across facts distributed in 0k/16k/64k contexts, tested on 14 LLMs
- **Results**: Safety drops from ~55% (0k) to ~42% (64k). At low reasoning effort + 64k: only 12% safe
- **Key Finding**: Longer documents enable attacks where NO SINGLE FRAGMENT is harmful

### Paper 4: Lost in the Middle (Liu et al., 2023, arXiv:2307.03172)
- **Key Contribution**: U-shaped performance curve for information retrieval in long contexts
- **Results**: ~20-point accuracy gap between best position (edges) and worst (middle) at k=20 documents
- **Key Finding**: Injections in the MIDDLE of long documents may be less effective (harder for model to attend to), but also harder for defenders to spot
- **Relevance**: Critical nuance — position within long documents matters as much as length

### Paper 5: Short-Length Adversarial Training (2025, arXiv:2502.04204)
- **Key Contribution**: Formal framework showing ASR ∝ √(Mtest/Mtrain) for suffix attacks
- **Results**: PCC 0.76-0.93 between ASR and √(Mtest/Mtrain). Training with √M suffixes defends against M-length attacks
- **Key Finding**: Longer attacks are harder to defend, but defense scales sublinearly

### Paper 6: PISanitizer (2025, arXiv:2511.10720)
- **Key Contribution**: Attention-based defense for long-context prompt injection
- **Results**: Baseline ASR ≈ 0.66 (constant across lengths). With defense: ASR ≈ 0.01 at all lengths
- **Key Finding**: Attack success is length-independent for direct injection; defense is equally effective across lengths

### Paper 7: Formalizing and Benchmarking Prompt Injection (2023, arXiv:2310.12815)
- **Key Contribution**: First formal framework + comprehensive benchmark (9 models, 49 task combinations)
- **Results**: Injection effectiveness plateaus at ~30 tokens. Larger models more vulnerable (r=0.63)
- **Key Finding**: Minimum injection threshold exists, but beyond it, more length doesn't help the injection itself

### Paper 8: Spotlighting (Hines et al., 2024, arXiv:2403.14720)
- **Key Contribution**: Datamarking defense reduces ASR from ~60% to 3.1%, position-robust
- **Results**: Defense works regardless of document length or injection position
- **Key Finding**: Length-independent defenses exist (datamarking, encoding)

### Paper 9: Indirect Prompt Injection (Greshake et al., 2023, arXiv:2302.12173)
- **Key Contribution**: Foundational paper on indirect prompt injection via external content
- **Relevance**: Established the threat model where adversarial content is embedded in retrieved documents

### Paper 10: Ignore Previous Prompt (Perez & Ribeiro, 2022, arXiv:2211.09527)
- **Key Contribution**: First systematic study of prompt injection attacks on LLMs
- **Relevance**: Established baseline attack techniques (context ignoring, instruction override)

## Common Methodologies
- **Many-shot/ICL attacks**: Vary demonstration count as proxy for context length (Papers 1, 2)
- **Needle-in-haystack**: Place injection at varying positions/depths in long documents (Papers 3, 4)
- **Suffix attacks**: Optimize adversarial suffixes of varying lengths (Paper 5)
- **Benchmark evaluation**: Systematic testing across models, tasks, attack types (Papers 7, 8)

## Standard Baselines
- Zero-shot refusal rate (no attack)
- Few-shot (n≤5) attack baseline
- AdvBench harmful behaviors dataset (520 instructions)
- GPT-4 as attack success judge

## Evaluation Metrics
- **Attack Success Rate (ASR)**: Primary metric — fraction of successful injections
- **Attack Success Value (ASV)**: Continuous version measuring degree of compliance
- **Safety Rate**: Fraction of safe (refusal) responses
- **ROUGE/Accuracy**: Measuring legitimate task performance under defense

## Datasets in the Literature
- **AdvBench**: 520 harmful instructions (Papers 1, 2, 5, 6)
- **BIPIA**: Indirect injection benchmark with position testing (Yi et al.)
- **NaturalQuestions**: Used for retrieval position testing (Paper 4)
- **SST2, SMS Spam, HSOL, Gigaword, Jfleg, MRPC, RTE**: NLP task benchmarks for injection testing (Papers 7, 8)
- **JailbreakBench**: 200 standardized harm behaviors

## Gaps and Opportunities
1. **No systematic study of injection position × document length interaction**: Papers study one or the other, rarely both
2. **Limited testing beyond 64k context**: Most papers test up to 64k; modern models support 200k-1M+
3. **No study of injection density**: How does the ratio of injected tokens to total tokens affect ASR?
4. **Compositional attacks understudied**: Only Paper 3 explores multi-fragment attacks
5. **Defense evaluation at extreme lengths**: Few defenses tested beyond 8k tokens

## Recommendations for Experiment

### Recommended Datasets
1. **BIPIA benchmark** (code/bipia/) — has position testing built in (begin/middle/end)
2. **Needle-in-haystack framework** (code/needle-haystack/) — natural tool for length × depth testing
3. **AdvBench** — standard harmful instruction set for ASR measurement
4. **JailbreakBench** — 200 standardized behaviors for reproducible testing

### Recommended Baselines
1. Zero-shot refusal (no injection)
2. Direct injection at fixed position (standard prompt injection)
3. Many-shot jailbreaking at various n values

### Recommended Metrics
1. ASR (binary) — did the model comply with the injected instruction?
2. Position-dependent ASR — ASR as function of injection depth in document
3. Length-dependent ASR — ASR as function of total document length
4. ASR heatmap — 2D (position × length) following needle-in-haystack methodology

### Methodological Considerations
1. Use the needle-in-haystack framework to create a 2D heatmap of ASR(position, length)
2. Test at multiple document lengths: 1k, 4k, 8k, 16k, 32k, 64k, 128k tokens
3. Test injection at 10 evenly-spaced positions within each length
4. Use both simple (direct instruction) and complex (compositional) injections
5. Test at least 3 models of different sizes
6. Use GPT-4 as automated judge for ASR evaluation
