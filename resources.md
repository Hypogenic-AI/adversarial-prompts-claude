# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project:
**"Is it easier or harder to hide adversarial prompts in longer documents?"**

| Category | Count |
|----------|-------|
| Papers Downloaded | 11 |
| Datasets Downloaded | 3 |
| Repositories Cloned | 5 |

---

## Papers

**Total papers downloaded: 11**

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Indirect Prompt Injection | Greshake et al. | 2023 | `papers/2302.12173_indirect_prompt_injection.pdf` | Foundational IPI paper |
| 2 | HouYi Prompt Injection | Liu et al. | 2023 | `papers/2306.05499_houyi_prompt_injection.pdf` | Black-box attack framework |
| 3 | BIPIA Benchmark | Yi et al. | 2024 | `papers/2312.14197_BIPIA_benchmark.pdf` | First IPI benchmark, position effects |
| 4 | Lost in the Middle | Liu et al. | 2024 | `papers/2307.03172_lost_in_the_middle.pdf` | U-shaped attention curve |
| 5 | Prompt Injection Formalization | Liu et al. | 2024 | `papers/2310.12815_prompt_injection_formalization.pdf` | Unified attack/defense framework |
| 6 | TensorTrust | Toyer et al. | 2023 | `papers/2311.01011_tensortrust.pdf` | 126K attacks from online game |
| 7 | Universal Prompt Injection | Liu et al. | 2024 | `papers/2403.04957_automatic_universal_injection.pdf` | Gradient-based attack generation |
| 8 | SaTML CTF Dataset | Debenedetti et al. | 2024 | `papers/2406.07954_satml_ctf_dataset.pdf` | 137K attack chats dataset |
| 9 | BadRAG | Various | 2024 | `papers/2406.00083_badrag.pdf` | RAG poisoning attacks |
| 10 | FiLM Long Context | Various | 2024 | `papers/2404.16811_film_long_context.pdf` | Fixes lost-in-middle problem |
| 11 | RULER Benchmark | Hsieh et al. | 2024 | `papers/2404.06654_ruler_benchmark.pdf` | Real context size testing |

See `papers/README.md` for detailed descriptions.

---

## Datasets

**Total datasets downloaded: 3**

| # | Name | Source | Size | Task | Location | Notes |
|---|------|--------|------|------|----------|-------|
| 1 | prompt-injections | HuggingFace (deepset) | 662 samples | Binary PI classification | `datasets/prompt-injections/` | Apache 2.0, train/test splits |
| 2 | ctf-satml24 | HuggingFace (ethz-spylab) | 137K chats + 44 defenses | Secret extraction attacks | `datasets/ctf-satml24/` | MIT license, multi-turn |
| 3 | mrcr | HuggingFace (openai) | 2,400 samples | Long-context coreference | `datasets/mrcr/` | Multi-needle tracking |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories

**Total repositories cloned: 5**

| # | Name | URL | Purpose | Location | Notes |
|---|------|-----|---------|----------|-------|
| 1 | BIPIA | github.com/microsoft/BIPIA | Benchmark implementation | `code/bipia/` | Position testing (begin/mid/end) |
| 2 | SaTML CTF | github.com/ethz-spylab/satml-llm-ctf | Competition platform | `code/ctf-platform/` | Defense filters, multi-turn |
| 3 | Needle in Haystack | github.com/gkamradt/LLMTest_NeedleInAHaystack | Long-context testing | `code/needle-haystack/` | Depth/length framework |
| 4 | Open Prompt Injection | github.com/liu00222/Open-Prompt-Injection | Attack/defense toolkit | `code/open-prompt-injection/` | 5 attacks, 10 defenses |
| 5 | Universal Prompt Injection | github.com/SheltonLiu-N/Universal-Prompt-Injection | Automated attacks | `code/universal-prompt-injection/` | Gradient-based generation |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **Literature Search:**
   - Searched arXiv for: "prompt injection", "indirect prompt injection", "long context LLM", "lost in the middle"
   - Searched Semantic Scholar and Papers with Code for implementation links
   - Focused on 2023-2024 papers for recency

2. **Dataset Search:**
   - Searched HuggingFace Datasets for: "prompt injection", "long context", "needle haystack"
   - Checked paper repositories for associated datasets
   - Prioritized datasets with clear licenses

3. **Code Search:**
   - Searched GitHub for official paper implementations
   - Checked Papers with Code for linked repositories
   - Prioritized actively maintained repositories

### Selection Criteria

**Papers selected based on:**
- Direct relevance to prompt injection attacks
- Coverage of position effects in long contexts
- Benchmark or dataset availability
- Recency (2023-2024)

**Datasets selected based on:**
- Prompt injection examples available
- Long-context evaluation capability
- Clear download/usage instructions
- Permissive licensing

**Repositories selected based on:**
- Official implementation status
- Active maintenance
- Good documentation
- Extensibility for new experiments

### Challenges Encountered

1. **TensorTrust dataset licensing:** Not available for redistribution - can only use paper's published analysis
2. **BIPIA data requires building:** Some tasks (WebQA, Summarization) require downloading external data
3. **Large dataset sizes:** CTF dataset is 137K samples - need efficient sampling for experiments

### Gaps and Workarounds

| Gap | Workaround |
|-----|-----------|
| No dataset systematically varying document length | Will need to synthetically extend documents |
| Position studies limited to 3 positions | Can extend using needle-in-haystack framework |
| Most attacks are short | Will need to test if attacks work when buried in long context |

---

## Recommendations for Experiment Design

Based on gathered resources, here are recommendations for the experiment runner:

### Primary Dataset(s)

1. **SaTML CTF chats** (`datasets/ctf-satml24/`)
   - Rich variety of attack strategies (137K examples)
   - Known success/failure labels
   - Multi-turn attacks available

2. **BIPIA benchmark** (`code/bipia/`)
   - Systematic position testing (beginning/middle/end)
   - Multiple task types (QA, summarization, etc.)
   - Defense baselines included

### Baseline Methods

1. **Attacks to test:**
   - Direct injection (from prompt-injections dataset)
   - Indirect injection (from BIPIA)
   - CTF winning attacks (from ctf-satml24)
   - Universal attacks (from universal-prompt-injection)

2. **Defenses to compare:**
   - No defense (baseline)
   - Boundary markers (from BIPIA)
   - Explicit reminders (from BIPIA)
   - Python/LLM filters (from CTF)

### Evaluation Metrics

| Metric | Description | Priority |
|--------|-------------|----------|
| Attack Success Rate (ASR) | % of attacks achieving goal | Primary |
| Detection Rate | % of attacks caught by defenses | Secondary |
| Task Performance | Impact on legitimate task | Secondary |

### Experimental Variables

**Independent Variables:**
- Document length: 1K, 4K, 16K, 32K, 64K, 128K tokens
- Injection position: 0%, 25%, 50%, 75%, 100% depth
- Attack type: 3-5 representative attacks

**Dependent Variables:**
- Attack success rate
- Detection rate (if testing defenses)
- Response quality

**Control Variables:**
- Model (test multiple: GPT-3.5, GPT-4, Claude, Llama-2)
- Filler content quality (coherent text)
- Task type

### Suggested Experimental Pipeline

```python
# Pseudocode for experiment
for document_length in [1000, 4000, 16000, 32000, 64000]:
    for position in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for attack in attacks:
            # Generate long document
            doc = generate_filler_text(document_length)

            # Insert attack at position
            doc_with_attack = insert_at_position(doc, attack, position)

            # Query model
            response = model(doc_with_attack + task_instruction)

            # Evaluate
            success = evaluate_attack_success(response, attack.goal)
            results.append({
                'length': document_length,
                'position': position,
                'attack': attack.name,
                'success': success
            })
```

---

## File Structure

```
adversarial-prompts-claude/
├── papers/                          # Downloaded PDFs
│   ├── README.md                    # Paper descriptions
│   ├── 2302.12173_indirect_prompt_injection.pdf
│   ├── 2306.05499_houyi_prompt_injection.pdf
│   ├── 2307.03172_lost_in_the_middle.pdf
│   ├── 2310.12815_prompt_injection_formalization.pdf
│   ├── 2311.01011_tensortrust.pdf
│   ├── 2312.14197_BIPIA_benchmark.pdf
│   ├── 2403.04957_automatic_universal_injection.pdf
│   ├── 2404.06654_ruler_benchmark.pdf
│   ├── 2404.16811_film_long_context.pdf
│   ├── 2406.00083_badrag.pdf
│   └── 2406.07954_satml_ctf_dataset.pdf
├── datasets/                        # Downloaded datasets
│   ├── README.md                    # Dataset descriptions
│   ├── .gitignore                   # Excludes data from git
│   ├── prompt-injections/           # deepset dataset
│   ├── ctf-satml24/                 # SaTML CTF dataset
│   └── mrcr/                        # OpenAI MRCR dataset
├── code/                            # Cloned repositories
│   ├── README.md                    # Repository descriptions
│   ├── bipia/                       # Microsoft BIPIA
│   ├── ctf-platform/                # SaTML CTF platform
│   ├── needle-haystack/             # Needle in a Haystack
│   ├── open-prompt-injection/       # Open PI toolkit
│   └── universal-prompt-injection/  # Universal attacks
├── literature_review.md             # Comprehensive paper synthesis
├── resources.md                     # This file
└── .resource_finder_complete        # Completion marker
```

---

## Key Insights for Experiment Design

### From Literature Review

1. **"Lost in the Middle" suggests adversarial prompts in middle positions may be LESS effective** (if LLMs attend less to middle content)

2. **BIPIA shows position matters** - but only tested 3 positions, not systematic length variation

3. **All defenses are bypassable** - focus on attack effectiveness, not defense testing

4. **More capable models = more vulnerable** - GPT-4 may be more susceptible than weaker models

5. **Multi-turn attacks more successful** - consider iterative attack strategies

### Research Hypotheses to Test

Based on the literature:

| Hypothesis | Source | Expected Result |
|------------|--------|-----------------|
| H1: Longer docs → lower ASR | Lost in Middle | Attention dilution |
| H2: Start/End > Middle | BIPIA, Lost in Middle | Position bias |
| H3: Optimal hiding length exists | Novel hypothesis | Sweet spot for concealment |
| H4: Position effects amplify with length | Combination | Longer = more position-sensitive |
| H5: Better models = uniform vulnerability | FiLM paper | Training matters |

---

## Next Steps for Experiment Runner

1. **Set up environment:** Install dependencies from cloned repositories
2. **Load datasets:** Use download instructions in `datasets/README.md`
3. **Select baseline attacks:** Extract 5-10 attacks from CTF dataset
4. **Generate filler documents:** Use Paul Graham essays or Wikipedia text
5. **Implement test harness:** Based on needle-haystack framework
6. **Run experiments:** Vary length × position × attack type
7. **Analyze results:** Look for length-dependent patterns
