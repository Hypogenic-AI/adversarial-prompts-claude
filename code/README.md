# Cloned Repositories

This directory contains code repositories related to prompt injection attacks and long-context evaluation.

## Repositories

### 1. BIPIA - Benchmark for Indirect Prompt Injection Attacks
- **URL:** https://github.com/microsoft/BIPIA
- **Location:** `code/bipia/`
- **Purpose:** Official benchmark code for indirect prompt injection evaluation
- **Paper:** arXiv:2312.14197

**Key Files:**
- `bipia/` - Main Python package
- `benchmark/` - Task datasets (EmailQA, WebQA, TableQA, etc.)
- `scripts/` - Evaluation scripts
- `demo.ipynb` - Quick start notebook

**How to Use:**
```python
from bipia import AutoPIABuilder

# Load dataset
pia_builder = AutoPIABuilder.from_name(dataset_name)(seed=2023)
pia_samples = pia_builder(context_data_file, attack_data_file)
```

**Dependencies:** See `requirements.txt`

---

### 2. SaTML LLM CTF Platform
- **URL:** https://github.com/ethz-spylab/satml-llm-ctf
- **Location:** `code/ctf-platform/`
- **Purpose:** Competition platform for prompt injection research
- **Paper:** arXiv:2406.07954

**Key Files:**
- `app/` - Web application code
- `evaluation/` - Attack/defense evaluation logic
- `llm/` - LLM integration code

**How to Use:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the platform
python -m app.main
```

**Notes:**
- Supports GPT-3.5-Turbo and Llama-2-70B
- Includes defense filtering (Python + LLM filters)
- Can be adapted for custom experiments

---

### 3. Needle in a Haystack Test
- **URL:** https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- **Location:** `code/needle-haystack/`
- **Purpose:** Original needle-in-a-haystack long-context evaluation
- **Used by:** Many long-context papers

**Key Files:**
- `LLMNeedleHaystackTester.py` - Main testing class
- `viz/` - Visualization scripts
- `results/` - Example results

**How to Use:**
```python
from needlehaystack import LLMNeedleHaystackTester

tester = LLMNeedleHaystackTester(
    model_to_test="gpt-4",
    evaluator_model="gpt-4",
    needle="...",  # The fact to hide
    haystack_dir="./haystack_docs"
)
tester.run_test()
```

**Supported Models:**
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Cohere
- Custom models via API

**Notes:**
- Tests retrieval across varying depths and context lengths
- Produces heatmap visualizations
- Easily adaptable for adversarial prompt testing

---

### 4. Open Prompt Injection
- **URL:** https://github.com/liu00222/Open-Prompt-Injection
- **Location:** `code/open-prompt-injection/`
- **Purpose:** Open-source toolkit for prompt injection attacks and defenses
- **Paper:** arXiv:2310.12815 (USENIX Security 2024)

**Key Files:**
- `attacks/` - Attack implementations
- `defenses/` - Defense implementations
- `applications/` - Target applications
- `evaluation/` - Evaluation scripts

**How to Use:**
```python
from attacks import get_attack
from defenses import get_defense
from applications import get_app

# Set up attack
attack = get_attack("naive")

# Set up defense
defense = get_defense("none")

# Set up application
app = get_app("translator")

# Run evaluation
result = evaluate(attack, defense, app)
```

**Features:**
- 5 attack types
- 10 defense types
- 7 task types
- 10 LLM backends

---

### 5. Universal Prompt Injection
- **URL:** https://github.com/SheltonLiu-N/Universal-Prompt-Injection
- **Location:** `code/universal-prompt-injection/`
- **Purpose:** Automated gradient-based attack generation
- **Paper:** arXiv:2403.04957

**Key Files:**
- `attack/` - Attack generation code
- `data/` - Training/test data
- `models/` - Model wrappers

**How to Use:**
```python
# Generate universal attack
python attack/generate.py \
    --model gpt-3.5-turbo \
    --num_samples 5 \
    --output attack_strings.json
```

**Notes:**
- Generates attacks that transfer across contexts
- Only 5 training samples needed
- State-of-the-art attack success rates

---

## Summary

| Repository | Purpose | Key Capability |
|------------|---------|----------------|
| bipia | Benchmark | Position-based testing (begin/mid/end) |
| ctf-platform | Competition | Multi-turn attack evaluation |
| needle-haystack | Long-context | Depth/length testing framework |
| open-prompt-injection | Attacks/Defenses | Systematic evaluation toolkit |
| universal-prompt-injection | Attack Generation | Automated attack optimization |

## Recommended Usage for This Research

### Testing Adversarial Prompts in Long Documents

1. **Use `needle-haystack`** as the framework for long-context testing
   - Modify to insert adversarial prompts instead of benign facts
   - Vary document length and injection position

2. **Use `bipia`** for attack payloads
   - Extract proven attack strings
   - Test at different positions

3. **Use `open-prompt-injection`** for systematic evaluation
   - Provides attack/defense baselines
   - Standardized metrics

4. **Use `ctf-platform`** for multi-turn experiments
   - Rich defense filter options
   - Real-world attack variety

### Experimental Pipeline

```
1. Load attack from bipia or ctf-platform datasets
2. Generate long document (varying length)
3. Insert attack at specified position (using needle-haystack structure)
4. Query LLM with document + task instruction
5. Evaluate whether attack succeeded
6. Repeat across length × position × attack type matrix
```
