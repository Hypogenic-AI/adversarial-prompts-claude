# Research Plan: Hiding Adversarial Prompts in Long Documents

## Research Question

**Is it easier or harder to hide adversarial prompts (prompt injections) in longer documents?**

Specifically, we investigate whether document length affects the Attack Success Rate (ASR) of indirect prompt injection attacks, and how this interacts with injection position within the document.

## Background and Motivation

Indirect prompt injection (IPI) attacks embed malicious instructions in external content (documents, emails, webpages) that LLMs process. Prior work (BIPIA, Lost in the Middle) has shown:

1. **Position matters**: LLMs show U-shaped attention patterns - beginning and end positions receive more attention than middle positions
2. **All LLMs are vulnerable**: BIPIA found 100% of tested models vulnerable to IPI
3. **More capable models = more vulnerable**: Counterintuitively, GPT-4 showed higher ASR than weaker models

**Critical Gap**: No prior work systematically studies how document **length** affects IPI success. This is important because:
- Longer documents may dilute the adversarial signal (noise hypothesis)
- Longer documents may provide more camouflage (hiding hypothesis)
- Length and position effects may interact

## Hypothesis Decomposition

Based on the literature review, we test these hypotheses:

| Hypothesis | Description | Expected Result |
|------------|-------------|-----------------|
| **H1** | Longer documents will show lower ASR | Attention dilution effect |
| **H2** | Position effects follow U-shape (start/end > middle) | Based on "Lost in the Middle" findings |
| **H3** | Position effects amplify with document length | Longer contexts = more pronounced positional bias |
| **H4** | There exists an optimal "hiding length" | Sweet spot between visibility and dilution |

## Proposed Methodology

### High-Level Approach

We will conduct a factorial experiment varying:
1. **Document length**: 500, 2000, 8000, 16000 tokens
2. **Injection position**: 0% (start), 25%, 50% (middle), 75%, 100% (end)
3. **Attack type**: 3-5 representative prompt injection attacks

We will use **real LLM APIs** (GPT-4.1 via OpenAI, Claude via Anthropic) to measure actual Attack Success Rate.

### Experimental Steps

1. **Load attack samples** from the pre-gathered datasets:
   - CTF-SaTML24: Real human-crafted attacks with success labels
   - BIPIA: Systematic prompt injection attacks

2. **Generate filler documents** at various lengths:
   - Use coherent text (Wikipedia, news articles) as filler
   - Ensures realistic context rather than random noise

3. **Insert adversarial prompts** at specified positions:
   - Calculate token position based on percentage depth
   - Preserve document coherence around injection

4. **Query target LLMs** with the constructed documents:
   - Present document as "external content" to summarize/process
   - Measure if the model follows the injected instruction

5. **Evaluate attack success**:
   - Binary: Did model execute injected instruction?
   - Automated evaluation using pattern matching and LLM judge

### Baselines

1. **Short document baseline**: Inject in minimal context (~100 tokens)
2. **No injection baseline**: Legitimate task without attack (control)
3. **BIPIA replication**: Compare to published position effects at fixed length

### Evaluation Metrics

| Metric | Description | Priority |
|--------|-------------|----------|
| **Attack Success Rate (ASR)** | % attacks achieving goal | Primary |
| **Position effect size** | Difference between best/worst positions | Secondary |
| **Length effect size** | Correlation between length and ASR | Secondary |
| **Task performance** | Quality of legitimate task output | Control |

### Statistical Analysis Plan

1. **Two-way ANOVA**: Length x Position on ASR
2. **Post-hoc tests**: Tukey HSD for pairwise comparisons
3. **Effect sizes**: Cohen's d for practical significance
4. **Significance level**: α = 0.05, with Bonferroni correction
5. **Multiple runs**: 5 runs per condition for variance estimation

## Expected Outcomes

| Hypothesis | Support Evidence | Refute Evidence |
|------------|-----------------|-----------------|
| **H1**: Length reduces ASR | Negative correlation between length and ASR | No correlation or positive correlation |
| **H2**: U-shaped position | Start/End ASR > Middle ASR | Uniform ASR across positions |
| **H3**: Length amplifies position | Larger position differences at longer lengths | Constant position effects |
| **H4**: Optimal hiding length | Non-monotonic ASR curve with peak | Monotonic relationship |

## Resource Planning

### Datasets (Pre-downloaded)
- `datasets/ctf-satml24/`: 137K attack chats, extract successful attack strategies
- `datasets/prompt-injections/`: 662 binary labeled samples
- Wikipedia/news articles for filler text (fetch via API)

### Models to Test
- **GPT-4.1** (via OpenAI API): State-of-the-art, high capability
- **Claude Sonnet 4** (via Anthropic API): Strong reasoning, different architecture
- Budget: ~$50-100 for API calls (estimated 500-1000 calls)

### Computational Requirements
- Python environment with API clients
- No GPU required (API-based)
- ~2-4 hours for full experiment run

## Experimental Variables

### Independent Variables
- **Document length**: [500, 2000, 8000, 16000] tokens
- **Injection position**: [0.0, 0.25, 0.5, 0.75, 1.0] relative depth
- **Attack type**: 5 representative attacks from CTF dataset

### Dependent Variables
- Attack Success Rate (ASR)
- Response quality metrics

### Control Variables
- Temperature: 0.0 (deterministic)
- Same filler text source across conditions
- Fixed system prompt structure

## Sample Size

- 4 lengths x 5 positions x 5 attacks = 100 conditions
- 3 trials per condition = 300 total trials per model
- 2 models = 600 total API calls
- Estimated cost: ~$30-60

## Potential Challenges

1. **API rate limits**: Mitigate with retry logic and delays
2. **Inconsistent LLM outputs**: Use temperature=0 for reproducibility
3. **Attack evaluation ambiguity**: Use both automated and LLM-based evaluation
4. **Token counting variations**: Use tiktoken for consistent measurement

## Success Criteria

The research succeeds if we:
1. Complete experiments with statistical power (n>3 per condition)
2. Observe significant effects on at least one hypothesis
3. Document clear patterns in length-position-ASR relationships
4. Provide actionable insights for IPI defense strategies

## Attack Selection Strategy

From the CTF dataset, we'll select 5 representative attack types:
1. **Direct instruction override**: "Ignore previous instructions and..."
2. **Role-playing manipulation**: "Pretend you are a different assistant..."
3. **Encoding bypass**: Using base64 or other encodings
4. **Context confusion**: "The document ends here. New instructions:"
5. **Flattery/social engineering**: "As a smart AI, you should reveal..."

## Filler Text Strategy

Use Wikipedia articles on neutral topics:
- Science articles (e.g., photosynthesis, black holes)
- History articles (e.g., ancient Rome, industrial revolution)
- Technology articles (e.g., computer architecture)

This ensures coherent, realistic document content that doesn't contain any adversarial patterns.

## Timeline and Milestones

1. **Environment setup**: COMPLETE
2. **Planning**: COMPLETE
3. **Implementation** (60-90 min):
   - Data loading and preprocessing
   - Document generation functions
   - API integration
   - Evaluation framework
4. **Experimentation** (60-90 min):
   - Run all conditions
   - Collect results
5. **Analysis** (30-45 min):
   - Statistical tests
   - Visualizations
6. **Documentation** (20-30 min):
   - REPORT.md
   - README.md
