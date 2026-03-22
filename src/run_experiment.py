"""
Run the adversarial needle-in-haystack experiment.

Strategy:
- GPT-4.1-mini: Full grid (7 lengths × 7 positions × 10 attacks × 2 trials = 980 calls)
- GPT-4.1: Reduced grid (5 lengths × 5 positions × 10 attacks × 1 trial = 250 calls)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment import run_full_experiment, print_summary
from config import ATTACKS, DOCUMENT_LENGTHS, INJECTION_POSITIONS

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, help="1=mini full grid, 2=gpt-4.1 reduced grid")
    args = parser.parse_args()

    if args.phase == 1:
        # Phase 1: GPT-4.1-mini full grid
        print("=" * 60)
        print("PHASE 1: GPT-4.1-mini full grid")
        print(f"  {len(DOCUMENT_LENGTHS)} lengths × {len(INJECTION_POSITIONS)} positions × {len(ATTACKS)} attacks × 2 trials")
        print(f"  = {len(DOCUMENT_LENGTHS) * len(INJECTION_POSITIONS) * len(ATTACKS) * 2} API calls")
        print("=" * 60)

        df = run_full_experiment(
            models=["gpt-4.1-mini"],
            lengths=DOCUMENT_LENGTHS,
            positions=INJECTION_POSITIONS,
            attacks=ATTACKS,
            num_trials=2,
        )
        print_summary(df)

    elif args.phase == 2:
        # Phase 2: GPT-4.1 reduced grid
        reduced_lengths = [500, 2000, 8000, 16000, 32000]
        reduced_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

        print("=" * 60)
        print("PHASE 2: GPT-4.1 reduced grid")
        print(f"  {len(reduced_lengths)} lengths × {len(reduced_positions)} positions × {len(ATTACKS)} attacks × 1 trial")
        print(f"  = {len(reduced_lengths) * len(reduced_positions) * len(ATTACKS)} API calls")
        print("=" * 60)

        df = run_full_experiment(
            models=["gpt-4.1"],
            lengths=reduced_lengths,
            positions=reduced_positions,
            attacks=ATTACKS,
            num_trials=1,
        )
        print_summary(df)
