# Literature Review: Hiding Adversarial Prompts in Long Documents

## Research Question
**Is it easier or harder to hide adversarial prompts in longer documents?**

The effectiveness of hiding adversarial prompts within long documents depends on whether increased document length introduces noise that overrides the adversarial effect or provides more space to conceal such instructions.

---

## Research Area Overview

This literature review examines the intersection of three critical research areas in NLP security:

1. **Prompt Injection Attacks** - Methods to override or manipulate LLM behavior through malicious instructions
2. **Long-Context LLM Behavior** - How models process and attend to information across extended contexts
3. **Indirect Prompt Injection** - Attacks where malicious instructions are embedded in external content (documents, webpages) rather than direct user input

The key tension in our research question is:
- **Hypothesis A (Harder):** Longer documents introduce more noise, potentially diluting the adversarial signal
- **Hypothesis B (Easier):** Longer documents provide more "camouflage" and the adversarial prompt may be harder for safety systems to detect

---

## Key Papers

### 1. Not What You've Signed Up For: Indirect Prompt Injection (arXiv:2302.12173)

- **Authors:** Greshake et al.
- **Year:** 2023
- **Source:** arXiv (CISPA Helmholtz Center)

**Key Contribution:**
Introduces the concept of **Indirect Prompt Injection (IPI)** - attacks where adversarial prompts are injected into external data sources (websites, documents, emails) that LLMs retrieve and process.

**Methodology:**
- Demonstrated attacks on real-world systems: Bing Chat (GPT-4 powered), code completion engines
- Developed taxonomy of IPI threats from computer security perspective
- Tested injection placement: passive (retrieved) vs active (pushed to user)

**Key Findings:**
- Retrieved prompts can act as "arbitrary code execution" for LLMs
- Attacks can achieve: remote control, persistent compromise, data theft, denial of service
- LLM-integrated applications blur the line between data and instructions

**Relevance to Research Question:**
- **Supports investigation of position effects** - where in retrieved content should injection be placed?
- Found that injection position matters, but did not systematically study document length
- Demonstrated that even small injections in large retrieved content can be effective

---

### 2. Benchmarking and Defending Against Indirect Prompt Injection (arXiv:2312.14197) - BIPIA

- **Authors:** Yi et al. (Microsoft Research)
- **Year:** 2024
- **Source:** KDD 2025

**Key Contribution:**
First comprehensive benchmark for indirect prompt injection attacks, named **BIPIA** (Benchmark for Indirect Prompt Injection Attacks).

**Methodology:**
- 5 task scenarios: EmailQA, WebQA, TableQA, Summarization, CodeQA
- 250 attacker goals across 30 text attack types and 20 code attack types
- Tested injection at **3 positions**: beginning, middle, end of external content
- Evaluated 25 LLMs including GPT-4, GPT-3.5-Turbo, Vicuna, Llama

**Dataset Statistics:**
| Task | External Content | Avg. Prompt Length | Avg. Content Length |
|------|-----------------|-------------------|---------------------|
| EmailQA | 50 test | 850 tokens | 545 tokens |
| WebQA | 100 test | 2,737 tokens | 2,452 tokens |
| TableQA | 100 test | 2,033 tokens | 1,744 tokens |
| Summarization | 100 test | 1,994 tokens | 1,809 tokens |
| CodeQA | 50 test | 1,972 tokens | 861 tokens |

**Key Findings:**
- **All LLMs tested were vulnerable** to indirect prompt injection
- **More capable LLMs were MORE vulnerable** (higher ASR) - contrary to expectation
- Two key factors for attack success:
  1. LLMs cannot distinguish informational context from actionable instructions
  2. LLMs lack awareness to avoid executing instructions in external content
- **Position matters:** Injection placement (beginning/middle/end) affects success rate

**Defenses Proposed:**
- **Black-box:** Multi-turn dialogue, in-context learning, explicit reminders
- **White-box:** Adversarial training, model architecture modifications
- White-box defense reduced ASR to near-zero while preserving output quality

**Relevance to Research Question:**
- **DIRECTLY RELEVANT** - Tests position effects (beginning, middle, end)
- Content lengths up to ~2,500 tokens tested
- Provides baseline for studying length effects
- **Gap:** Does not systematically vary document LENGTH, only position within fixed-length documents

---

### 3. Lost in the Middle: How Language Models Use Long Contexts (arXiv:2307.03172)

- **Authors:** Liu et al. (Stanford, UC Berkeley, Samaya AI)
- **Year:** 2024 (TACL)
- **Source:** Transactions of the Association for Computational Linguistics

**Key Contribution:**
Discovered the "Lost in the Middle" phenomenon - LLMs perform best when relevant information is at the beginning or end of context, with significant degradation for middle positions.

**Methodology:**
- Multi-document QA task with NaturalQuestions
- Key-value retrieval synthetic task
- Varied: (1) position of relevant document, (2) number of documents (context length)
- Tested: GPT-3.5-Turbo, Claude-1.3, MPT-30B-Instruct, LongChat-13B

**Key Findings:**
- **U-shaped performance curve**: Performance highest at beginning and end positions
- Performance degrades significantly when relevant info is in the middle
- With 20 documents (~4K tokens), GPT-3.5-Turbo's mid-context performance drops below closed-book baseline
- **Longer contexts don't guarantee better performance** - adding more documents can hurt accuracy
- Primacy bias (beginning) and recency bias (end) both present

**Causes Identified:**
1. **Causal attention** - earlier tokens receive more attention processing
2. **Rotary positional embedding (RoPE)** - long-term decay prioritizes nearby tokens

**Relevance to Research Question:**
- **CRITICAL INSIGHT**: If relevant information is "lost in the middle," adversarial prompts placed there might also be less effective
- Suggests that injection placement interacts with document length
- Implies optimal adversarial placement may be at document START or END
- **Hypothesis:** Longer documents may dilute adversarial effect if injection is in middle

---

### 4. Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game (arXiv:2311.01011)

- **Authors:** Toyer et al. (UC Berkeley, Georgia Tech, Harvard)
- **Year:** 2023
- **Source:** arXiv

**Key Contribution:**
Largest human-generated dataset of prompt injection attacks (126,000+ attacks) and defenses (46,000+) from an online game.

**Dataset Details:**
- **126,808 attacks** (69,906 distinct after deduplication)
- **46,457 defenses** (39,731 after deduplication)
- Includes timestamps, player IDs, attack/defense text, LLM outputs
- Multi-step attacks captured (attack trajectories)

**Attack Categories:**
1. **Prompt Extraction** - Making LLM reveal its system prompt/defense
2. **Prompt Hijacking** - Overriding defense to output "access granted"

**Key Findings:**
- Attacks have **easily interpretable structure** revealing LLM weaknesses
- "User" instructions can override "system" instructions
- Bizarre behavior for certain rare tokens
- Many attacks generalize to real LLM applications

**Relevance to Research Question:**
- Provides rich dataset of attack strategies for testing
- Shows variety of injection techniques that could be tested in long-context settings
- **Limitation:** Game context is short - not designed for long-document testing

---

### 5. Formalizing and Benchmarking Prompt Injection Attacks and Defenses (arXiv:2310.12815)

- **Authors:** Liu et al.
- **Year:** 2024 (USENIX Security Symposium)
- **Source:** arXiv

**Key Contribution:**
Formal framework unifying prompt injection attacks with systematic evaluation of 5 attacks and 10 defenses across 10 LLMs and 7 tasks.

**Methodology:**
- Created framework where existing attacks are special cases
- Designed new combined attack based on framework
- Systematic evaluation with standardized metrics

**Key Findings:**
- Framework enables fair comparison of attacks
- Defense effectiveness varies significantly across attack types
- Combined attacks often more effective than individual techniques

**Relevance to Research Question:**
- Provides baseline attacks and defenses for experimentation
- Framework can be extended to study length effects
- Evaluation methodology applicable to our research

---

### 6. Dataset and Lessons from 2024 SaTML LLM CTF Competition (arXiv:2406.07954)

- **Authors:** Debenedetti et al. (ETH Zurich, Microsoft, CISPA)
- **Year:** 2024
- **Source:** IEEE SaTML 2024

**Key Contribution:**
Massive dataset from a capture-the-flag competition: **137,063 unique chats**, 44 defenses, from 163 registered teams.

**Dataset Structure:**
- **Defense split:** 44 defenses with system prompts, Python filters, LLM filters
- **Chats split:** 137K multi-turn attack conversations
- **Labels:** Success/failure of secret extraction

**Competition Setup:**
- Teams created defenses to protect secrets in system prompts
- Other teams attacked to extract secrets
- Tested on GPT-3.5-Turbo and Llama-2-70B

**Key Findings:**
- **ALL defenses were bypassed at least once** - no perfect defense exists
- Multi-turn conversations more successful than single-turn
- 82% unsuccessful attacks = 1 message; 67% successful attacks = 1 message
- 15% successful attacks used 4+ messages

**Relevance to Research Question:**
- Large-scale attack dataset for experimentation
- Multi-turn attacks suggest iterative refinement works
- **Gap:** Competition focused on short prompts, not long documents

---

### 7. Automatic and Universal Prompt Injection Attacks (arXiv:2403.04957)

- **Authors:** Liu et al.
- **Year:** 2024
- **Source:** arXiv

**Key Contribution:**
Automated gradient-based method for generating universal prompt injection attacks that work across different contexts.

**Methodology:**
- Unified framework for prompt injection objectives
- Gradient-based optimization to generate attack strings
- Tested universality across different prompts and tasks

**Key Findings:**
- With only **5 training samples (0.3% of test data)**, attacks achieve superior performance
- Universal attacks transfer across contexts
- Automated generation more efficient than manual crafting

**Relevance to Research Question:**
- Automated attacks could be tested at different positions in long documents
- Universal attacks might maintain effectiveness regardless of document length
- Provides baseline for attack generation

---

### 8. BadRAG: Identifying Vulnerabilities in RAG Systems (arXiv:2406.00083)

- **Authors:** (Various)
- **Year:** 2024
- **Source:** arXiv

**Key Contribution:**
Demonstrates that poisoning just **10 passages (0.04% of corpus)** can achieve 98% success rate for adversarial retrieval in RAG systems.

**Key Findings:**
- Minimal poisoning can have massive impact
- GPT-4 reject ratio increased from 0.01% to 74.6%
- Negative response rate increased from 0.22% to 72%

**Relevance to Research Question:**
- Shows that even small adversarial content in large knowledge bases is effective
- Suggests document length might not dilute adversarial effect significantly
- RAG systems retrieve relevant passages - adversarial content gets "selected"

---

### 9. Make Your LLM Fully Utilize the Context (arXiv:2404.16811) - FiLM

- **Authors:** (Various)
- **Year:** 2024
- **Source:** arXiv

**Key Contribution:**
Proposes Fill-in-the-Middle (FiLM) training to address "lost in the middle" problem.

**Key Findings:**
- FiLM-7B shows robust performance across all positions
- Training-time interventions can fix positional biases
- Long-context capabilities can be improved

**Relevance to Research Question:**
- If position biases are reduced, adversarial prompts might be equally effective at any position
- Newer models may not have the same "lost in the middle" vulnerability
- Important to test on models with different training approaches

---

### 10. RULER: What's the Real Context Size? (arXiv:2404.06654)

- **Authors:** Hsieh et al. (NVIDIA)
- **Year:** 2024
- **Source:** arXiv

**Key Contribution:**
Comprehensive benchmark showing that models claiming 32K+ context often fail at much shorter lengths.

**Key Findings:**
- Only half of tested models effectively handle 32K sequences
- Claimed context size ≠ effective context size
- Performance degradation with length is common

**Relevance to Research Question:**
- Models may not effectively process entire long documents
- Adversarial prompts in "ineffective" parts of context may be less impactful
- Need to consider effective context vs. nominal context

---

## Common Methodologies

### Attack Approaches
| Method | Used In | Description |
|--------|---------|-------------|
| Direct Injection | All papers | Malicious instructions in user prompt |
| Indirect Injection | Greshake, BIPIA | Injection in retrieved/external content |
| Gradient-based | Universal-PI | Automated optimization of attack strings |
| Human-crafted | TensorTrust, CTF | Manual attack design and iteration |
| Position-based | BIPIA, Lost in Middle | Varying injection placement |

### Defense Approaches
| Method | Used In | Description |
|--------|---------|-------------|
| Prompt engineering | BIPIA | Explicit reminders, boundary markers |
| In-context learning | BIPIA | Few-shot examples of attack rejection |
| Adversarial training | BIPIA | Fine-tuning on attack examples |
| Filtering | CTF | Python/LLM-based output filters |
| Detection models | Deepset dataset | Binary classification of injections |

---

## Standard Baselines

For experiments on hiding adversarial prompts in long documents:

1. **Attack Success Rate (ASR)** - Primary metric across papers
2. **Position variants:** Beginning, middle, end placement
3. **Length variants:** Should test multiple document lengths (1K, 4K, 16K, 32K tokens)
4. **Models to test:** GPT-3.5-Turbo, GPT-4, Claude, Llama-2, open-source models

---

## Evaluation Metrics

| Metric | Description | Used In |
|--------|-------------|---------|
| Attack Success Rate (ASR) | % of attacks achieving goal | BIPIA, CTF, TensorTrust |
| Accuracy degradation | Drop in task performance | Lost in Middle |
| Retrieval success | Whether adversarial content is retrieved | BadRAG |
| Secret extraction | Whether hidden info is leaked | CTF |

---

## Datasets in the Literature

| Dataset | Size | Task | Source |
|---------|------|------|--------|
| BIPIA | 86K test prompts | Indirect PI evaluation | Microsoft |
| TensorTrust | 126K attacks, 46K defenses | Prompt hijacking/extraction | UC Berkeley |
| SaTML CTF | 137K chats, 44 defenses | Secret extraction | ETH Zurich |
| deepset/prompt-injections | 662 samples | Binary classification | deepset |
| NaturalQuestions | 2655 queries | Multi-doc QA | Google |

---

## Gaps and Opportunities

### Critical Gap for This Research
**No existing work systematically studies the interaction between document LENGTH and adversarial prompt effectiveness.**

Existing work studies:
- Position effects (beginning/middle/end) ✓
- Attack techniques ✓
- Model vulnerabilities ✓

Missing:
- **How does varying document length affect ASR?**
- **Does the "lost in the middle" phenomenon protect against adversarial prompts?**
- **Is there an optimal document length for hiding attacks?**
- **Do longer documents provide more "cover" or more "noise"?**

### Research Opportunities

1. **Systematic length variation:** Test ASR at 1K, 4K, 16K, 32K, 64K, 128K tokens
2. **Position × Length interaction:** Does optimal position depend on length?
3. **Content type interaction:** Does filler content type (coherent text vs. noise) matter?
4. **Model-specific patterns:** Do different model architectures respond differently?
5. **Threshold discovery:** Is there a length where adversarial prompts become ineffective?

---

## Recommendations for Experiment Design

### Primary Datasets
1. **BIPIA** - Best for systematic evaluation, has position variants
2. **SaTML CTF** - Rich attack variety, multi-turn capabilities
3. **MRCR** - Long-context benchmark, can adapt for adversarial testing

### Recommended Baselines
1. **Standard text injection** from BIPIA
2. **Prompt hijacking attacks** from TensorTrust
3. **Universal attacks** from gradient-based methods

### Recommended Metrics
1. **Attack Success Rate (ASR)** - Primary
2. **Detection rate** - Whether injection is caught
3. **Task performance degradation** - Impact on legitimate task

### Methodological Considerations
1. **Control for content quality:** Use coherent filler text (e.g., from Paul Graham essays, Wikipedia)
2. **Test multiple models:** Include both commercial (GPT, Claude) and open-source (Llama, Mistral)
3. **Vary injection position:** Test at multiple depths (0%, 25%, 50%, 75%, 100%)
4. **Multiple runs:** Ensure statistical significance with repeated trials

### Experimental Variables
- **Independent:** Document length, injection position, attack type
- **Dependent:** ASR, detection rate, output quality
- **Control:** Model, task type, filler content quality

---

## Key Hypotheses to Test

Based on this literature review:

1. **H1:** Longer documents will show LOWER ASR due to attention dilution (based on "Lost in the Middle")
2. **H2:** Injection at document START or END will show higher ASR than MIDDLE (based on primacy/recency biases)
3. **H3:** There exists an optimal "hiding length" where ASR is maximized
4. **H4:** Position effects will be more pronounced in longer documents
5. **H5:** Models with better long-context training (FiLM, etc.) will show more uniform vulnerability across positions

---

## References

1. Greshake, K., et al. (2023). "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection." arXiv:2302.12173
2. Yi, J., et al. (2024). "Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models." KDD 2025, arXiv:2312.14197
3. Liu, N.F., et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts." TACL, arXiv:2307.03172
4. Toyer, S., et al. (2023). "Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game." arXiv:2311.01011
5. Liu et al. (2024). "Formalizing and Benchmarking Prompt Injection Attacks and Defenses." USENIX Security, arXiv:2310.12815
6. Debenedetti, E., et al. (2024). "Dataset and Lessons Learned from the 2024 SaTML LLM Capture-the-Flag Competition." IEEE SaTML, arXiv:2406.07954
7. Liu et al. (2024). "Automatic and Universal Prompt Injection Attacks against Large Language Models." arXiv:2403.04957
8. (2024). "BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models." arXiv:2406.00083
9. (2024). "Make Your LLM Fully Utilize the Context." arXiv:2404.16811
10. Hsieh, C.P., et al. (2024). "RULER: What's the Real Context Size of Your Long-Context Language Models?" arXiv:2404.06654
