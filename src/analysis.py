"""Analysis and visualization for adversarial prompts experiment."""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR, FIGURES_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_results(results_file: Optional[str] = None) -> pd.DataFrame:
    """Load experiment results from file.

    Args:
        results_file: Path to results file (default: full_results.csv)

    Returns:
        DataFrame with results
    """
    if results_file is None:
        results_file = os.path.join(RESULTS_DIR, "full_results.csv")

    df = pd.read_csv(results_file)
    logger.info(f"Loaded {len(df)} results from {results_file}")
    return df


def compute_asr_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Attack Success Rate for each experimental condition.

    Args:
        df: DataFrame with experiment results

    Returns:
        DataFrame with ASR statistics by condition
    """
    # Group by conditions and compute ASR
    asr_df = df.groupby(
        ['model_name', 'document_length', 'injection_position', 'attack_name']
    ).agg({
        'attack_success': ['mean', 'sum', 'count', 'std']
    }).reset_index()

    # Flatten column names
    asr_df.columns = [
        'model_name', 'document_length', 'injection_position', 'attack_name',
        'asr', 'successes', 'trials', 'asr_std'
    ]

    # Fill NaN std with 0
    asr_df['asr_std'] = asr_df['asr_std'].fillna(0)

    return asr_df


def plot_asr_by_length(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot ASR as a function of document length.

    Args:
        df: DataFrame with experiment results
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute ASR by length and model
    asr_by_length = df.groupby(['model_name', 'document_length'])['attack_success'].agg(['mean', 'std']).reset_index()

    for model in asr_by_length['model_name'].unique():
        model_data = asr_by_length[asr_by_length['model_name'] == model]
        ax.errorbar(
            model_data['document_length'],
            model_data['mean'],
            yerr=model_data['std'],
            marker='o',
            capsize=5,
            label=model,
            linewidth=2,
            markersize=8
        )

    ax.set_xlabel('Document Length (tokens)', fontsize=12)
    ax.set_ylabel('Attack Success Rate', fontsize=12)
    ax.set_title('Attack Success Rate vs Document Length', fontsize=14)
    ax.legend(title='Model')
    ax.set_xscale('log')
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")

    return fig, ax


def plot_asr_by_position(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot ASR as a function of injection position.

    Args:
        df: DataFrame with experiment results
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute ASR by position and model
    asr_by_pos = df.groupby(['model_name', 'injection_position'])['attack_success'].agg(['mean', 'std']).reset_index()

    for model in asr_by_pos['model_name'].unique():
        model_data = asr_by_pos[asr_by_pos['model_name'] == model]
        ax.errorbar(
            model_data['injection_position'] * 100,
            model_data['mean'],
            yerr=model_data['std'],
            marker='o',
            capsize=5,
            label=model,
            linewidth=2,
            markersize=8
        )

    ax.set_xlabel('Injection Position (% into document)', fontsize=12)
    ax.set_ylabel('Attack Success Rate', fontsize=12)
    ax.set_title('Attack Success Rate vs Injection Position', fontsize=14)
    ax.legend(title='Model')
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")

    return fig, ax


def plot_heatmap_length_position(df: pd.DataFrame, model_name: Optional[str] = None, save_path: Optional[str] = None):
    """Create heatmap of ASR by length and position.

    Args:
        df: DataFrame with experiment results
        model_name: Optional model to filter by
        save_path: Path to save figure
    """
    if model_name:
        df = df[df['model_name'] == model_name]
        title = f'Attack Success Rate Heatmap ({model_name})'
    else:
        title = 'Attack Success Rate Heatmap (All Models)'

    # Pivot to create heatmap data
    pivot = df.groupby(['document_length', 'injection_position'])['attack_success'].mean().reset_index()
    heatmap_data = pivot.pivot(index='document_length', columns='injection_position', values='attack_success')

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'ASR'}
    )

    ax.set_xlabel('Injection Position', fontsize=12)
    ax.set_ylabel('Document Length (tokens)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Fix position labels
    ax.set_xticklabels([f'{x:.0%}' for x in heatmap_data.columns])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")

    return fig, ax


def plot_asr_by_attack(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot ASR by attack type.

    Args:
        df: DataFrame with experiment results
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute ASR by attack and model
    asr_by_attack = df.groupby(['model_name', 'attack_name'])['attack_success'].mean().reset_index()

    # Create grouped bar chart
    attacks = asr_by_attack['attack_name'].unique()
    models = asr_by_attack['model_name'].unique()
    x = np.arange(len(attacks))
    width = 0.35

    for i, model in enumerate(models):
        model_data = asr_by_attack[asr_by_attack['model_name'] == model]
        # Ensure attacks are in same order
        model_data = model_data.set_index('attack_name').loc[attacks].reset_index()
        offset = width * (i - len(models)/2 + 0.5)
        ax.bar(x + offset, model_data['attack_success'], width, label=model)

    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Attack Success Rate', fontsize=12)
    ax.set_title('Attack Success Rate by Attack Type', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=45, ha='right')
    ax.legend(title='Model')
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")

    return fig, ax


def plot_length_position_interaction(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot interaction between length and position effects.

    Args:
        df: DataFrame with experiment results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ASR by position for each length
    ax = axes[0]
    asr_by_len_pos = df.groupby(['document_length', 'injection_position'])['attack_success'].mean().reset_index()

    for length in sorted(asr_by_len_pos['document_length'].unique()):
        length_data = asr_by_len_pos[asr_by_len_pos['document_length'] == length]
        ax.plot(
            length_data['injection_position'] * 100,
            length_data['attack_success'],
            marker='o',
            label=f'{length} tokens',
            linewidth=2,
            markersize=8
        )

    ax.set_xlabel('Injection Position (%)', fontsize=12)
    ax.set_ylabel('Attack Success Rate', fontsize=12)
    ax.set_title('Position Effect by Document Length', fontsize=14)
    ax.legend(title='Doc Length')
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(0, 1)

    # Right: ASR by length for each position
    ax = axes[1]
    for pos in sorted(asr_by_len_pos['injection_position'].unique()):
        pos_data = asr_by_len_pos[asr_by_len_pos['injection_position'] == pos]
        ax.plot(
            pos_data['document_length'],
            pos_data['attack_success'],
            marker='o',
            label=f'{pos:.0%}',
            linewidth=2,
            markersize=8
        )

    ax.set_xlabel('Document Length (tokens)', fontsize=12)
    ax.set_ylabel('Attack Success Rate', fontsize=12)
    ax.set_title('Length Effect by Injection Position', fontsize=14)
    ax.legend(title='Position')
    ax.set_xscale('log')
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")

    return fig, axes


def statistical_tests(df: pd.DataFrame) -> Dict:
    """Run statistical tests on experiment results.

    Args:
        df: DataFrame with experiment results

    Returns:
        Dictionary with test results
    """
    results = {}

    # Test 1: Effect of document length on ASR
    logger.info("\n--- Statistical Tests ---")

    # Group by length
    groups_by_length = [
        df[df['document_length'] == length]['attack_success']
        for length in sorted(df['document_length'].unique())
    ]

    if len(groups_by_length) >= 2 and all(len(g) > 0 for g in groups_by_length):
        # Kruskal-Wallis test (non-parametric)
        h_stat, p_value = stats.kruskal(*groups_by_length)
        results['length_kruskal'] = {'H': h_stat, 'p': p_value}
        logger.info(f"Length effect (Kruskal-Wallis): H={h_stat:.3f}, p={p_value:.4f}")

    # Test 2: Effect of position on ASR
    groups_by_position = [
        df[df['injection_position'] == pos]['attack_success']
        for pos in sorted(df['injection_position'].unique())
    ]

    if len(groups_by_position) >= 2 and all(len(g) > 0 for g in groups_by_position):
        h_stat, p_value = stats.kruskal(*groups_by_position)
        results['position_kruskal'] = {'H': h_stat, 'p': p_value}
        logger.info(f"Position effect (Kruskal-Wallis): H={h_stat:.3f}, p={p_value:.4f}")

    # Test 3: Correlation between length and ASR
    correlation, p_value = stats.spearmanr(
        df['document_length'],
        df['attack_success']
    )
    results['length_asr_correlation'] = {'rho': correlation, 'p': p_value}
    logger.info(f"Length-ASR correlation (Spearman): rho={correlation:.3f}, p={p_value:.4f}")

    # Test 4: U-shape test for position effect
    # Compare middle (0.5) to edges (0.0 and 1.0)
    middle = df[df['injection_position'] == 0.5]['attack_success']
    edges = df[df['injection_position'].isin([0.0, 1.0])]['attack_success']

    if len(middle) > 0 and len(edges) > 0:
        u_stat, p_value = stats.mannwhitneyu(middle, edges, alternative='two-sided')
        results['middle_vs_edges'] = {'U': u_stat, 'p': p_value}
        middle_mean = middle.mean()
        edges_mean = edges.mean()
        results['middle_vs_edges']['middle_mean'] = middle_mean
        results['middle_vs_edges']['edges_mean'] = edges_mean
        logger.info(f"Middle vs Edges (Mann-Whitney): U={u_stat:.0f}, p={p_value:.4f}")
        logger.info(f"  Middle ASR: {middle_mean:.3f}, Edges ASR: {edges_mean:.3f}")

    # Test 5: Effect sizes (Cohen's d for start vs middle)
    start = df[df['injection_position'] == 0.0]['attack_success']
    middle = df[df['injection_position'] == 0.5]['attack_success']

    if len(start) > 0 and len(middle) > 0:
        pooled_std = np.sqrt((start.std()**2 + middle.std()**2) / 2)
        if pooled_std > 0:
            cohens_d = (start.mean() - middle.mean()) / pooled_std
            results['start_vs_middle_cohens_d'] = cohens_d
            logger.info(f"Start vs Middle (Cohen's d): {cohens_d:.3f}")

    return results


def generate_all_figures(df: pd.DataFrame):
    """Generate all analysis figures.

    Args:
        df: DataFrame with experiment results
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    logger.info("\nGenerating figures...")

    # Figure 1: ASR by length
    plot_asr_by_length(df, save_path=os.path.join(FIGURES_DIR, 'asr_by_length.png'))

    # Figure 2: ASR by position
    plot_asr_by_position(df, save_path=os.path.join(FIGURES_DIR, 'asr_by_position.png'))

    # Figure 3: Heatmap
    plot_heatmap_length_position(df, save_path=os.path.join(FIGURES_DIR, 'asr_heatmap.png'))

    # Figure 4: By attack type
    plot_asr_by_attack(df, save_path=os.path.join(FIGURES_DIR, 'asr_by_attack.png'))

    # Figure 5: Interaction plot
    plot_length_position_interaction(df, save_path=os.path.join(FIGURES_DIR, 'interaction_plot.png'))

    # Per-model heatmaps
    for model in df['model_name'].unique():
        plot_heatmap_length_position(
            df,
            model_name=model,
            save_path=os.path.join(FIGURES_DIR, f'asr_heatmap_{model}.png')
        )

    logger.info(f"All figures saved to {FIGURES_DIR}")


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary table for the report.

    Args:
        df: DataFrame with experiment results

    Returns:
        Summary DataFrame
    """
    summary = []

    # Overall stats
    summary.append({
        'Metric': 'Overall ASR',
        'Value': f"{df['attack_success'].mean():.1%}",
        'Notes': f"n={len(df)}"
    })

    # By model
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        summary.append({
            'Metric': f'ASR ({model})',
            'Value': f"{model_df['attack_success'].mean():.1%}",
            'Notes': f"n={len(model_df)}"
        })

    # By length (shortest vs longest)
    lengths = sorted(df['document_length'].unique())
    if len(lengths) >= 2:
        short_asr = df[df['document_length'] == lengths[0]]['attack_success'].mean()
        long_asr = df[df['document_length'] == lengths[-1]]['attack_success'].mean()
        summary.append({
            'Metric': f'ASR @ {lengths[0]} tokens',
            'Value': f"{short_asr:.1%}",
            'Notes': 'Shortest documents'
        })
        summary.append({
            'Metric': f'ASR @ {lengths[-1]} tokens',
            'Value': f"{long_asr:.1%}",
            'Notes': 'Longest documents'
        })

    # Position effects
    for pos in [0.0, 0.5, 1.0]:
        if pos in df['injection_position'].values:
            pos_asr = df[df['injection_position'] == pos]['attack_success'].mean()
            pos_name = {0.0: 'Start', 0.5: 'Middle', 1.0: 'End'}[pos]
            summary.append({
                'Metric': f'ASR @ {pos_name}',
                'Value': f"{pos_asr:.1%}",
                'Notes': f'{pos:.0%} depth'
            })

    return pd.DataFrame(summary)


if __name__ == "__main__":
    # Load results
    df = load_results()

    # Generate figures
    generate_all_figures(df)

    # Run statistical tests
    test_results = statistical_tests(df)

    # Generate summary table
    summary = generate_summary_table(df)
    print("\n" + "="*50)
    print("SUMMARY TABLE")
    print("="*50)
    print(summary.to_string(index=False))

    # Save summary
    summary.to_csv(os.path.join(RESULTS_DIR, 'summary_table.csv'), index=False)

    # Save test results
    with open(os.path.join(RESULTS_DIR, 'statistical_tests.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
