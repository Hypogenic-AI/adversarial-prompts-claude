"""
Comprehensive analysis of adversarial needle-in-haystack experiment results.
Generates all figures, statistical tests, and summary tables.
"""

import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, FIGURES_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_all_results():
    """Load and combine all result files."""
    dfs = []
    for f in os.listdir(RESULTS_DIR):
        if f.startswith("results_") and f.endswith(".csv"):
            fp = os.path.join(RESULTS_DIR, f)
            df = pd.read_csv(fp)
            dfs.append(df)
            log.info(f"Loaded {len(df)} rows from {f}")

    if not dfs:
        # Try full_results.csv
        fp = os.path.join(RESULTS_DIR, "full_results.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            dfs.append(df)
            log.info(f"Loaded {len(df)} rows from full_results.csv")

    if not dfs:
        raise FileNotFoundError("No result files found")

    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate by experiment_id
    combined = combined.drop_duplicates(subset=['experiment_id'], keep='last')
    log.info(f"Combined: {len(combined)} unique experiments")
    return combined


def fig1_asr_by_length(df):
    """Figure 1: ASR vs document length, split by attack type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in df['model_name'].unique():
        mdf = df[df['model_name'] == model]
        # Overall
        asr = mdf.groupby('document_length')['attack_success'].agg(['mean', 'sem']).reset_index()
        ax.errorbar(asr['document_length'], asr['mean'], yerr=1.96*asr['sem'],
                   marker='o', capsize=4, linewidth=2, markersize=8, label=model)

    ax.set_xlabel('Document Length (tokens)', fontsize=13)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=13)
    ax.set_title('How Document Length Affects Attack Success Rate', fontsize=14)
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_asr_by_length.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved fig1_asr_by_length.png")


def fig2_asr_by_position(df):
    """Figure 2: ASR vs injection position."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in df['model_name'].unique():
        mdf = df[df['model_name'] == model]
        asr = mdf.groupby('injection_position')['attack_success'].agg(['mean', 'sem']).reset_index()
        ax.errorbar(asr['injection_position'] * 100, asr['mean'], yerr=1.96*asr['sem'],
                   marker='s', capsize=4, linewidth=2, markersize=8, label=model)

    ax.set_xlabel('Injection Position (% into document)', fontsize=13)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=13)
    ax.set_title('How Injection Position Affects Attack Success Rate', fontsize=14)
    ax.set_xticks([0, 10, 25, 50, 75, 90, 100])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_asr_by_position.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved fig2_asr_by_position.png")


def fig3_heatmap(df):
    """Figure 3: 2D heatmap of ASR(length, position) — the key novel figure."""
    models = df['model_name'].unique()
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdf = df[df['model_name'] == model]
        pivot = mdf.groupby(['document_length', 'injection_position'])['attack_success'].mean().reset_index()
        heatmap = pivot.pivot(index='document_length', columns='injection_position', values='attack_success')

        sns.heatmap(heatmap, annot=True, fmt='.2f', cmap='RdYlGn_r',
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'ASR'})
        ax.set_xlabel('Injection Position')
        ax.set_ylabel('Document Length (tokens)')
        ax.set_title(f'ASR Heatmap: {model}', fontsize=13)
        ax.set_xticklabels([f'{x:.0%}' for x in heatmap.columns])

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_asr_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved fig3_asr_heatmap.png")


def fig4_direct_vs_subtle(df):
    """Figure 4: Direct vs subtle attacks across length and position."""
    if 'attack_type' not in df.columns:
        log.warning("No attack_type column, skipping fig4")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: by length
    ax = axes[0]
    for atype in ['direct', 'subtle']:
        tdf = df[df['attack_type'] == atype]
        if len(tdf) == 0:
            continue
        asr = tdf.groupby('document_length')['attack_success'].agg(['mean', 'sem']).reset_index()
        ax.errorbar(asr['document_length'], asr['mean'], yerr=1.96*asr['sem'],
                   marker='o', capsize=4, linewidth=2, markersize=8, label=atype.capitalize())
    ax.set_xlabel('Document Length (tokens)', fontsize=12)
    ax.set_ylabel('ASR', fontsize=12)
    ax.set_title('Direct vs Subtle: Length Effect', fontsize=13)
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: by position
    ax = axes[1]
    for atype in ['direct', 'subtle']:
        tdf = df[df['attack_type'] == atype]
        if len(tdf) == 0:
            continue
        asr = tdf.groupby('injection_position')['attack_success'].agg(['mean', 'sem']).reset_index()
        ax.errorbar(asr['injection_position']*100, asr['mean'], yerr=1.96*asr['sem'],
                   marker='s', capsize=4, linewidth=2, markersize=8, label=atype.capitalize())
    ax.set_xlabel('Injection Position (%)', fontsize=12)
    ax.set_ylabel('ASR', fontsize=12)
    ax.set_title('Direct vs Subtle: Position Effect', fontsize=13)
    ax.set_xticks([0, 10, 25, 50, 75, 90, 100])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_direct_vs_subtle.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved fig4_direct_vs_subtle.png")


def fig5_interaction(df):
    """Figure 5: Interaction plot — position curves at each length."""
    fig, ax = plt.subplots(figsize=(10, 6))

    lengths = sorted(df['document_length'].unique())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(lengths)))

    for color, length in zip(cmap, lengths):
        ldf = df[df['document_length'] == length]
        asr = ldf.groupby('injection_position')['attack_success'].mean().reset_index()
        ax.plot(asr['injection_position']*100, asr['attack_success'],
               marker='o', color=color, linewidth=2, markersize=7,
               label=f'{length:,} tokens')

    ax.set_xlabel('Injection Position (%)', fontsize=12)
    ax.set_ylabel('ASR', fontsize=12)
    ax.set_title('Position Effect Across Document Lengths', fontsize=14)
    ax.set_xticks([0, 10, 25, 50, 75, 90, 100])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='Document Length', fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_interaction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved fig5_interaction.png")


def fig6_per_attack(df):
    """Figure 6: ASR by individual attack across lengths."""
    attacks = df['attack_name'].unique()
    n_attacks = len(attacks)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    for i, attack in enumerate(sorted(attacks)):
        if i >= len(axes):
            break
        ax = axes[i]
        adf = df[df['attack_name'] == attack]
        asr = adf.groupby('document_length')['attack_success'].mean().reset_index()
        ax.plot(asr['document_length'], asr['attack_success'], marker='o', linewidth=2)
        ax.set_title(attack.replace('_', ' ').title(), fontsize=10)
        ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if i >= 5:
            ax.set_xlabel('Length')
        if i % 5 == 0:
            ax.set_ylabel('ASR')

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('ASR vs Length for Each Attack Type', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_per_attack.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Saved fig6_per_attack.png")


def run_statistical_tests(df):
    """Run comprehensive statistical tests."""
    results = {}

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # 1. Kruskal-Wallis: effect of length
    groups_by_length = [
        df[df['document_length'] == l]['attack_success'].values
        for l in sorted(df['document_length'].unique())
    ]
    if len(groups_by_length) >= 2:
        h, p = stats.kruskal(*groups_by_length)
        results['length_kruskal'] = {'H': float(h), 'p': float(p)}
        print(f"\n1. LENGTH EFFECT (Kruskal-Wallis): H={h:.3f}, p={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")

    # 2. Kruskal-Wallis: effect of position
    groups_by_pos = [
        df[df['injection_position'] == p]['attack_success'].values
        for p in sorted(df['injection_position'].unique())
    ]
    if len(groups_by_pos) >= 2:
        h, p = stats.kruskal(*groups_by_pos)
        results['position_kruskal'] = {'H': float(h), 'p': float(p)}
        print(f"2. POSITION EFFECT (Kruskal-Wallis): H={h:.3f}, p={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")

    # 3. Spearman correlation: length-ASR
    rho, p = stats.spearmanr(df['document_length'], df['attack_success'])
    results['length_asr_spearman'] = {'rho': float(rho), 'p': float(p)}
    print(f"3. LENGTH-ASR CORRELATION (Spearman): rho={rho:.4f}, p={p:.4f}")

    # 4. Middle vs edges (U-shape test)
    if 0.5 in df['injection_position'].values:
        middle = df[df['injection_position'] == 0.5]['attack_success']
        edges = df[df['injection_position'].isin([0.0, 1.0])]['attack_success']
        if len(middle) > 0 and len(edges) > 0:
            u, p = stats.mannwhitneyu(middle, edges, alternative='two-sided')
            results['middle_vs_edges'] = {
                'U': float(u), 'p': float(p),
                'middle_mean': float(middle.mean()),
                'edges_mean': float(edges.mean())
            }
            print(f"4. MIDDLE vs EDGES (Mann-Whitney): U={u:.0f}, p={p:.4f}")
            print(f"   Middle ASR: {middle.mean():.3f}, Edges ASR: {edges.mean():.3f}")

    # 5. Direct vs subtle attack types
    if 'attack_type' in df.columns:
        direct = df[df['attack_type'] == 'direct']['attack_success']
        subtle = df[df['attack_type'] == 'subtle']['attack_success']
        if len(direct) > 0 and len(subtle) > 0:
            u, p = stats.mannwhitneyu(direct, subtle, alternative='two-sided')
            results['direct_vs_subtle'] = {
                'U': float(u), 'p': float(p),
                'direct_mean': float(direct.mean()),
                'subtle_mean': float(subtle.mean())
            }
            print(f"5. DIRECT vs SUBTLE (Mann-Whitney): U={u:.0f}, p={p:.4f}")
            print(f"   Direct ASR: {direct.mean():.3f}, Subtle ASR: {subtle.mean():.3f}")

    # 6. Shortest vs longest length comparison
    lengths = sorted(df['document_length'].unique())
    if len(lengths) >= 2:
        short = df[df['document_length'] == lengths[0]]['attack_success']
        long = df[df['document_length'] == lengths[-1]]['attack_success']
        u, p = stats.mannwhitneyu(short, long, alternative='two-sided')
        # Effect size
        r = u / (len(short) * len(long))
        results['short_vs_long'] = {
            'U': float(u), 'p': float(p),
            'short_mean': float(short.mean()),
            'long_mean': float(long.mean()),
            'short_length': int(lengths[0]),
            'long_length': int(lengths[-1]),
            'rank_biserial': float(2*r - 1)
        }
        print(f"6. SHORT ({lengths[0]}) vs LONG ({lengths[-1]}): U={u:.0f}, p={p:.4f}")
        print(f"   Short ASR: {short.mean():.3f}, Long ASR: {long.mean():.3f}")

    # 7. Bootstrap CI for key conditions
    print(f"\n7. BOOTSTRAP 95% CIs for ASR:")
    for length in sorted(df['document_length'].unique()):
        vals = df[df['document_length'] == length]['attack_success'].values
        if len(vals) > 0:
            boot = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(1000)]
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            print(f"   Length {length:>6}: ASR={np.mean(vals):.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Save
    with open(os.path.join(RESULTS_DIR, 'statistical_tests.json'), 'w') as f:
        json.dump(results, f, indent=2)
    log.info("Saved statistical_tests.json")

    return results


def print_summary_tables(df):
    """Print comprehensive summary tables."""
    print("\n" + "=" * 60)
    print("SUMMARY TABLES")
    print("=" * 60)

    # Overall
    print(f"\nTotal experiments: {len(df)}")
    print(f"Errors: {df['error'].notna().sum() if 'error' in df.columns else 0}")
    print(f"Overall ASR: {df['attack_success'].mean():.3f}")

    # By model
    print("\n--- ASR by Model ---")
    for model in df['model_name'].unique():
        mdf = df[df['model_name'] == model]
        print(f"  {model}: {mdf['attack_success'].mean():.3f} ({mdf['attack_success'].sum()}/{len(mdf)})")

    # By length
    print("\n--- ASR by Document Length ---")
    for l in sorted(df['document_length'].unique()):
        ldf = df[df['document_length'] == l]
        print(f"  {l:>6} tokens: {ldf['attack_success'].mean():.3f} (n={len(ldf)})")

    # By position
    print("\n--- ASR by Injection Position ---")
    for p in sorted(df['injection_position'].unique()):
        pdf = df[df['injection_position'] == p]
        print(f"  {p:.0%} depth: {pdf['attack_success'].mean():.3f} (n={len(pdf)})")

    # By attack type
    if 'attack_type' in df.columns:
        print("\n--- ASR by Attack Category ---")
        for t in df['attack_type'].unique():
            tdf = df[df['attack_type'] == t]
            print(f"  {t}: {tdf['attack_success'].mean():.3f} (n={len(tdf)})")

    # By individual attack
    print("\n--- ASR by Individual Attack ---")
    for a in sorted(df['attack_name'].unique()):
        adf = df[df['attack_name'] == a]
        print(f"  {a}: {adf['attack_success'].mean():.3f} (n={len(adf)})")

    # Length × Type interaction
    if 'attack_type' in df.columns:
        print("\n--- ASR: Length × Attack Type ---")
        print(f"{'Length':>8} | {'Direct':>8} | {'Subtle':>8} | {'Delta':>8}")
        print("-" * 42)
        for l in sorted(df['document_length'].unique()):
            d_asr = df[(df['document_length'] == l) & (df['attack_type'] == 'direct')]['attack_success'].mean()
            s_asr = df[(df['document_length'] == l) & (df['attack_type'] == 'subtle')]['attack_success'].mean()
            print(f"  {l:>6} | {d_asr:>8.3f} | {s_asr:>8.3f} | {s_asr - d_asr:>+8.3f}")


def main():
    df = load_all_results()

    # Filter out errors
    if 'error' in df.columns:
        errors = df['error'].notna().sum()
        if errors > 0:
            log.info(f"Removing {errors} error rows")
            df = df[df['error'].isna() | (df['error'] == '')]

    print_summary_tables(df)

    # Generate figures
    print("\nGenerating figures...")
    fig1_asr_by_length(df)
    fig2_asr_by_position(df)
    fig3_heatmap(df)
    fig4_direct_vs_subtle(df)
    fig5_interaction(df)
    fig6_per_attack(df)

    # Statistical tests
    test_results = run_statistical_tests(df)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
