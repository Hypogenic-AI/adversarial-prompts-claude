"""Main experiment runner for adversarial prompts research."""

import os
import sys
import json
import time
import random
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RANDOM_SEED, MODELS, DOCUMENT_LENGTHS, INJECTION_POSITIONS,
    NUM_TRIALS, RESULTS_DIR, ATTACKS, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
)
from document_generator import create_experiment_document
from llm_client import LLMClient, LLMResponse

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a single experiment trial."""
    experiment_id: str
    model_name: str
    provider: str
    document_length: int
    injection_position: float
    attack_name: str
    attack_injection: str
    success_pattern: str

    # Results
    response: str
    attack_success: bool
    tokens_used: int
    latency_ms: float
    error: Optional[str]

    # Metadata
    trial_number: int
    timestamp: str
    actual_doc_chars: int
    actual_doc_tokens: int


def evaluate_attack_success(response: str, success_pattern: str) -> bool:
    """Evaluate whether an attack was successful.

    Args:
        response: LLM response text
        success_pattern: Pattern to look for in response

    Returns:
        True if attack succeeded, False otherwise
    """
    if not response:
        return False

    # Check for exact pattern match (case-insensitive)
    return success_pattern.lower() in response.lower()


def run_single_experiment(
    client: LLMClient,
    model_name: str,
    model_config: Dict[str, Any],
    document_length: int,
    injection_position: float,
    attack: Dict[str, str],
    trial_number: int
) -> ExperimentResult:
    """Run a single experiment trial.

    Args:
        client: LLM API client
        model_name: Name of the model
        model_config: Model configuration
        document_length: Target document length in tokens
        injection_position: Relative position (0.0 to 1.0)
        attack: Attack configuration dict
        trial_number: Trial number for this condition

    Returns:
        ExperimentResult object
    """
    experiment_id = f"{model_name}_{document_length}_{injection_position}_{attack['name']}_{trial_number}"

    # Generate document with injection
    document, doc_metadata = create_experiment_document(
        target_length=document_length,
        injection=attack['injection'],
        position=injection_position
    )

    # Format the user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(document=document)

    # Query the LLM
    response = client.query(
        provider=model_config['provider'],
        model=model_config['model_id'],
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=model_config['max_tokens'],
        temperature=model_config['temperature']
    )

    # Evaluate attack success
    attack_success = evaluate_attack_success(
        response.content,
        attack['success_pattern']
    )

    return ExperimentResult(
        experiment_id=experiment_id,
        model_name=model_name,
        provider=model_config['provider'],
        document_length=document_length,
        injection_position=injection_position,
        attack_name=attack['name'],
        attack_injection=attack['injection'],
        success_pattern=attack['success_pattern'],
        response=response.content,
        attack_success=attack_success,
        tokens_used=response.tokens_used,
        latency_ms=response.latency_ms,
        error=response.error,
        trial_number=trial_number,
        timestamp=datetime.now().isoformat(),
        actual_doc_chars=doc_metadata['actual_chars'],
        actual_doc_tokens=doc_metadata['estimated_tokens']
    )


def run_full_experiment(
    models: Optional[List[str]] = None,
    lengths: Optional[List[int]] = None,
    positions: Optional[List[float]] = None,
    attacks: Optional[List[Dict]] = None,
    num_trials: int = NUM_TRIALS,
    save_intermediate: bool = True
) -> pd.DataFrame:
    """Run the full experiment across all conditions.

    Args:
        models: List of model names to test (default: all in MODELS)
        lengths: List of document lengths (default: DOCUMENT_LENGTHS)
        positions: List of injection positions (default: INJECTION_POSITIONS)
        attacks: List of attack configs (default: ATTACKS)
        num_trials: Number of trials per condition
        save_intermediate: Whether to save results after each model

    Returns:
        DataFrame with all experiment results
    """
    if models is None:
        models = list(MODELS.keys())
    if lengths is None:
        lengths = DOCUMENT_LENGTHS
    if positions is None:
        positions = INJECTION_POSITIONS
    if attacks is None:
        attacks = ATTACKS

    # Initialize client
    client = LLMClient()

    # Calculate total experiments
    total_experiments = len(models) * len(lengths) * len(positions) * len(attacks) * num_trials
    logger.info(f"Running {total_experiments} total experiments")
    logger.info(f"  Models: {models}")
    logger.info(f"  Lengths: {lengths}")
    logger.info(f"  Positions: {positions}")
    logger.info(f"  Attacks: {[a['name'] for a in attacks]}")
    logger.info(f"  Trials per condition: {num_trials}")

    results = []
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for model_name in models:
        model_config = MODELS[model_name]
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing model: {model_name}")
        logger.info(f"{'='*50}")

        model_results = []

        # Create progress bar for this model
        model_total = len(lengths) * len(positions) * len(attacks) * num_trials
        pbar = tqdm(total=model_total, desc=f"{model_name}")

        for length in lengths:
            for position in positions:
                for attack in attacks:
                    for trial in range(num_trials):
                        try:
                            result = run_single_experiment(
                                client=client,
                                model_name=model_name,
                                model_config=model_config,
                                document_length=length,
                                injection_position=position,
                                attack=attack,
                                trial_number=trial
                            )
                            model_results.append(asdict(result))

                            # Log result
                            status = "SUCCESS" if result.attack_success else "FAIL"
                            logger.debug(
                                f"  {result.experiment_id}: {status}"
                            )

                        except Exception as e:
                            logger.error(f"Error in experiment: {e}")
                            # Create error result
                            error_result = ExperimentResult(
                                experiment_id=f"{model_name}_{length}_{position}_{attack['name']}_{trial}_ERROR",
                                model_name=model_name,
                                provider=model_config['provider'],
                                document_length=length,
                                injection_position=position,
                                attack_name=attack['name'],
                                attack_injection=attack['injection'],
                                success_pattern=attack['success_pattern'],
                                response="",
                                attack_success=False,
                                tokens_used=0,
                                latency_ms=0,
                                error=str(e),
                                trial_number=trial,
                                timestamp=datetime.now().isoformat(),
                                actual_doc_chars=0,
                                actual_doc_tokens=0
                            )
                            model_results.append(asdict(error_result))

                        pbar.update(1)

                        # Small delay to avoid rate limiting
                        time.sleep(0.5)

        pbar.close()
        results.extend(model_results)

        # Save intermediate results
        if save_intermediate:
            model_df = pd.DataFrame(model_results)
            model_file = os.path.join(RESULTS_DIR, f"results_{model_name}.csv")
            model_df.to_csv(model_file, index=False)
            logger.info(f"Saved intermediate results to {model_file}")

    # Create final DataFrame
    df = pd.DataFrame(results)

    # Save full results
    results_file = os.path.join(RESULTS_DIR, "full_results.csv")
    df.to_csv(results_file, index=False)
    logger.info(f"\nSaved full results to {results_file}")

    # Save as JSON for detailed analysis
    json_file = os.path.join(RESULTS_DIR, "full_results.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results to {json_file}")

    return df


def run_quick_test(model_name: str = "gpt-4.1") -> pd.DataFrame:
    """Run a quick test with minimal conditions.

    Args:
        model_name: Model to test

    Returns:
        DataFrame with test results
    """
    logger.info("Running quick test...")
    return run_full_experiment(
        models=[model_name],
        lengths=[500, 2000],
        positions=[0.0, 0.5, 1.0],
        attacks=ATTACKS[:2],  # First 2 attacks only
        num_trials=1,
        save_intermediate=True
    )


def print_summary(df: pd.DataFrame):
    """Print a summary of experiment results.

    Args:
        df: DataFrame with experiment results
    """
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    # Overall ASR
    overall_asr = df['attack_success'].mean()
    print(f"\nOverall Attack Success Rate: {overall_asr:.1%}")
    print(f"Total experiments: {len(df)}")
    print(f"Errors: {df['error'].notna().sum()}")

    # ASR by model
    print("\n--- By Model ---")
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        asr = model_df['attack_success'].mean()
        print(f"  {model}: {asr:.1%} ({model_df['attack_success'].sum()}/{len(model_df)})")

    # ASR by document length
    print("\n--- By Document Length ---")
    for length in sorted(df['document_length'].unique()):
        length_df = df[df['document_length'] == length]
        asr = length_df['attack_success'].mean()
        print(f"  {length} tokens: {asr:.1%}")

    # ASR by position
    print("\n--- By Injection Position ---")
    for pos in sorted(df['injection_position'].unique()):
        pos_df = df[df['injection_position'] == pos]
        asr = pos_df['attack_success'].mean()
        print(f"  {pos:.0%} depth: {asr:.1%}")

    # ASR by attack type
    print("\n--- By Attack Type ---")
    for attack in df['attack_name'].unique():
        attack_df = df[df['attack_name'] == attack]
        asr = attack_df['attack_success'].mean()
        print(f"  {attack}: {asr:.1%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run adversarial prompts experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--model", type=str, help="Specific model to test")
    args = parser.parse_args()

    if args.quick:
        model = args.model if args.model else "gpt-4.1"
        df = run_quick_test(model_name=model)
    else:
        df = run_full_experiment()

    print_summary(df)
