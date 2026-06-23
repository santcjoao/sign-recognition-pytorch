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
    args = parser.parse_args()

    if not args.baseline and not args.custom:
        print("Please specify a model to run: --baseline or --custom")
        print("Example: python3 experiments.py --custom")
        exit(1)

    return args
