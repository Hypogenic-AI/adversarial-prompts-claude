# Downloaded Papers

This directory contains research papers related to studying adversarial prompts in long documents.

## Papers List

### Core Papers on Prompt Injection

1. **[Indirect Prompt Injection](2302.12173_indirect_prompt_injection.pdf)**
   - **Title:** Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection
   - **Authors:** Greshake, K., Abdelnabi, S., Mishra, S., et al.
   - **Year:** 2023
   - **arXiv:** 2302.12173
   - **Why relevant:** Foundational paper on indirect prompt injection - attacks via external content

2. **[HouYi - Prompt Injection Framework](2306.05499_houyi_prompt_injection.pdf)**
   - **Title:** Prompt Injection attack against LLM-integrated Applications
   - **Authors:** Liu et al.
   - **Year:** 2023
   - **arXiv:** 2306.05499
   - **Why relevant:** Black-box prompt injection attack techniques and framework

3. **[BIPIA Benchmark](2312.14197_BIPIA_benchmark.pdf)**
   - **Title:** Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models
   - **Authors:** Yi, J., Xie, Y., Zhu, B., et al.
   - **Year:** 2024
   - **arXiv:** 2312.14197
   - **Why relevant:** First benchmark for indirect PI - tests position effects (beginning/middle/end)

4. **[Prompt Injection Formalization](2310.12815_prompt_injection_formalization.pdf)**
   - **Title:** Formalizing and Benchmarking Prompt Injection Attacks and Defenses
   - **Authors:** Liu et al.
   - **Year:** 2024 (USENIX Security)
   - **arXiv:** 2310.12815
   - **Why relevant:** Unified framework for attacks/defenses, systematic evaluation

5. **[TensorTrust Dataset](2311.01011_tensortrust.pdf)**
   - **Title:** Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game
   - **Authors:** Toyer, S., Watkins, O., et al.
   - **Year:** 2023
   - **arXiv:** 2311.01011
   - **Why relevant:** Largest human-generated attack dataset (126K attacks, 46K defenses)

6. **[Universal Prompt Injection](2403.04957_automatic_universal_injection.pdf)**
   - **Title:** Automatic and Universal Prompt Injection Attacks against Large Language Models
   - **Authors:** Liu et al.
   - **Year:** 2024
   - **arXiv:** 2403.04957
   - **Why relevant:** Automated gradient-based attack generation

7. **[SaTML CTF Dataset](2406.07954_satml_ctf_dataset.pdf)**
   - **Title:** Dataset and Lessons Learned from the 2024 SaTML LLM Capture-the-Flag Competition
   - **Authors:** Debenedetti, E., Rando, J., et al.
   - **Year:** 2024
   - **arXiv:** 2406.07954
   - **Why relevant:** 137K attack chats dataset, multi-turn attacks

### Long Context and Position Effects

8. **[Lost in the Middle](2307.03172_lost_in_the_middle.pdf)**
   - **Title:** Lost in the Middle: How Language Models Use Long Contexts
   - **Authors:** Liu, N.F., Lin, K., Hewitt, J., et al.
   - **Year:** 2024 (TACL)
   - **arXiv:** 2307.03172
   - **Why relevant:** U-shaped performance curve - models struggle with middle context

9. **[FiLM - Long Context Utilization](2404.16811_film_long_context.pdf)**
   - **Title:** Make Your LLM Fully Utilize the Context
   - **Authors:** Various
   - **Year:** 2024
   - **arXiv:** 2404.16811
   - **Why relevant:** Addresses lost-in-the-middle problem with training

10. **[RULER Benchmark](2404.06654_ruler_benchmark.pdf)**
    - **Title:** RULER: What's the Real Context Size of Your Long-Context Language Models?
    - **Authors:** Hsieh, C.P., et al.
    - **Year:** 2024
    - **arXiv:** 2404.06654
    - **Why relevant:** Shows effective context size < claimed context size

### RAG Security

11. **[BadRAG](2406.00083_badrag.pdf)**
    - **Title:** BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models
    - **Authors:** Various
    - **Year:** 2024
    - **arXiv:** 2406.00083
    - **Why relevant:** Shows minimal poisoning (0.04% of corpus) achieves 98% attack success

## Summary Statistics

- **Total Papers:** 11
- **Core PI Papers:** 7
- **Long Context Papers:** 3
- **RAG Security Papers:** 1
- **Year Range:** 2023-2024
- **Total Size:** ~18 MB

## Key Themes

1. **Indirect Prompt Injection** - Attacks via external content, not direct user input
2. **Position Effects** - Where in context adversarial content appears matters
3. **Lost in the Middle** - Models struggle with mid-context information
4. **Universal Attacks** - Some attacks transfer across contexts
5. **Defense Brittleness** - All tested defenses can be bypassed
