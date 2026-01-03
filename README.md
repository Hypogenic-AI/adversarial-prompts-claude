# Adversarial Prompts in Long Documents

**Research Question**: Is it easier or harder to hide adversarial prompts (prompt injections) in longer documents?

## Key Findings

- **Document length has no significant effect on attack success rate** (p=0.76)
- **Position within the document doesn't matter** either (p=0.63)
- **Attack type is the critical factor**: "context confusion" attacks achieved 76.7% ASR vs 0% for most other types
- **Model architecture matters**: Claude Sonnet 4 resisted all attacks (0% ASR) while GPT-4.1 was vulnerable (18.7% ASR)

## Results Summary

| Condition | Attack Success Rate |
|-----------|---------------------|
| Overall | 9.3% |
| GPT-4.1 | 18.7% |
| Claude Sonnet 4 | 0.0% |
| Short docs (500 tokens) | 8.0% |
| Long docs (8000 tokens) | 11.0% |
| Context Confusion attack | 76.7% (GPT-4.1) |

## Practical Implications

1. Don't rely on document length as a defense against prompt injection
2. Focus defenses on specific attack patterns (e.g., detecting fake document boundaries)
3. Model choice matters - Claude appears more resistant to these attacks

## Repository Structure

```
adversarial-prompts-claude/
├── REPORT.md           # Full research report with methodology and findings
├── README.md           # This file
├── planning.md         # Research plan
├── src/                # Experiment code
│   ├── config.py       # Configuration and parameters
│   ├── document_generator.py  # Document construction
│   ├── llm_client.py   # API client for GPT/Claude
│   ├── experiment.py   # Main experiment runner
│   └── analysis.py     # Statistical analysis and visualization
├── results/            # Experiment results
│   ├── full_results.csv
│   └── full_results.json
├── figures/            # Generated visualizations
│   ├── asr_by_length.png
│   ├── asr_by_position.png
│   ├── asr_heatmap.png
│   └── ...
├── datasets/           # Pre-gathered datasets
├── papers/             # Reference papers
└── code/               # Baseline implementations
```

## Reproducing Results

### Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install openai anthropic datasets pandas numpy matplotlib seaborn scipy tqdm
```

### Run Experiments

```bash
# Quick test (2 lengths, 3 positions, 2 attacks)
python src/experiment.py --quick

# Full experiment
python src/experiment.py
```

### Analyze Results

```bash
python src/analysis.py
```

## Key Files

- **REPORT.md**: Comprehensive 15-page research report with all findings
- **src/experiment.py**: Main experiment runner (300 API calls)
- **src/analysis.py**: Statistical analysis and figure generation
- **results/full_results.json**: Raw experiment data

## Citation

If you use this research, please cite:

```
@misc{adversarial-prompts-2026,
  title={Is it Easier or Harder to Hide Adversarial Prompts in Longer Documents?},
  year={2026},
  note={Experimental study on indirect prompt injection vs document length}
}
```

## License

This research was conducted for academic purposes. See individual dataset licenses in `datasets/README.md`.
