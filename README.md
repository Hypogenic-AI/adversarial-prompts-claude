# Adversarial Prompts in Long Documents

**Research Question**: Is it easier or harder to hide adversarial prompts (prompt injections) in longer documents?

## Key Findings

1. **Longer documents reduce ASR for moderate attacks**: Subtle prompt injections drop from 97% success at 500 tokens to 60% at 2,000+ tokens (p<0.0001), then plateau.
2. **The strongest attacks are immune to document length**: Three "metadata-mimicking" attacks achieve 100% ASR regardless of document length (500–32,000 tokens).
3. **End-of-document position is critical for direct attacks**: Direct injection ASR jumps from 22% to 61% when placed at the end of the document (p<0.0001).
4. **Attack sophistication matters more than length**: Subtle attacks (68% ASR) are 2.6× more effective than direct attacks (26% ASR).
5. **GPT-4.1 resists direct attacks better but remains vulnerable to subtle ones**: 0% ASR on most direct attacks, but 92-100% on context confusion and metadata-mimicking injections.

## Methodology

An **adversarial needle-in-haystack** experiment: embed prompt injections at controlled positions within documents of varying lengths, then measure whether the target LLM follows the injected instruction.

- **1,230 experiments** across GPT-4.1 and GPT-4.1-mini
- **7 document lengths**: 500 to 32,000 tokens
- **7 injection positions**: 0% to 100% depth
- **10 attack types**: 5 direct + 5 subtle
- **Evaluation**: Pattern matching for attack success

## How to Reproduce

```bash
# Set up environment
uv venv && source .venv/bin/activate
uv add openai tiktoken numpy pandas matplotlib seaborn scipy tqdm

# Set API key
export OPENAI_API_KEY=your_key

# Run GPT-4.1-mini full grid (980 API calls, ~45 min)
python src/run_experiment.py --phase 1

# Run GPT-4.1 reduced grid (250 API calls, ~15 min)
python src/run_experiment.py --phase 2

# Analyze results and generate figures
python src/full_analysis.py
```

## File Structure

```
.
├── REPORT.md              # Full research report with results
├── README.md              # This file
├── planning.md            # Experimental design and hypotheses
├── literature_review.md   # Literature review and synthesis
├── resources.md           # Catalog of available resources
├── src/
│   ├── config.py          # Experiment configuration
│   ├── experiment.py      # Main experiment runner
│   ├── run_experiment.py  # Convenience runner script
│   ├── llm_client.py      # LLM API client
│   ├── document_generator.py  # Filler text + injection insertion
│   ├── analysis.py        # Analysis utilities
│   └── full_analysis.py   # Comprehensive analysis script
├── results/               # Raw results (CSV, JSON)
├── figures/               # Generated visualizations
├── papers/                # Downloaded research papers
├── datasets/              # Downloaded datasets
└── code/                  # Cloned baseline repositories
```

## Full Report

See [REPORT.md](REPORT.md) for comprehensive results, statistical analysis, and discussion.
