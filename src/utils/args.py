import argparse


def parse_experiment_args() -> argparse.Namespace:
    """Parse command line arguments for experiments."""
    parser = argparse.ArgumentParser(description="Train GTSRB Models")
    parser.add_argument(
        "--baseline", action="store_true", help="Train baseline model with Adam"
    )
    parser.add_argument(
        "--custom", action="store_true", help="Train custom deep model with Adam"
    )
    parser.add_argument(
        "--compare-optimizers",
        action="store_true",
        help="Compare SGD, SGD+Momentum, Adam on baseline with BatchNorm",
    )
    parser.add_argument(
        "--compare-norm",
        action="store_true",
        help="Compare baseline without vs with Batch Normalization",
    )
    args = parser.parse_args()

    flags = [args.baseline, args.custom, args.compare_optimizers, args.compare_norm]
    if not any(flags):
        print("Please specify at least one mode:")
        print("  --baseline            Train baseline model with Adam")
        print("  --custom              Train custom deep model with Adam")
        print("  --compare-optimizers  Compare SGD / SGDm / Adam (baseline + BN)")
        print("  --compare-norm        Compare without vs with Batch Normalization")
        print("\nExample: python experiments.py --compare-optimizers --compare-norm")
        exit(1)

    return args
