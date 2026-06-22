import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from src import BASE_TRANSFORM, GTSRB_CLASSES, get_dataloaders, save_predictions

RUNS_DIR = "results/runs"


def get_ground_truth_labels():
    """Extract ground truth labels from the test DataLoader.

    Ensures exact alignment with the test set loader order.
    """
    print("Loading test DataLoader to extract ground truth labels...")
    _, _, test_loader = get_dataloaders(
        train_transform=BASE_TRANSFORM,
        batch_size=64,
    )

    y_true = []
    for _, targets in test_loader:
        y_true.extend(targets.tolist())

    return np.array(y_true)


def plot_training_curves(history: dict, exp_name: str, save_path: str):
    """Generate and save loss and accuracy training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    ax1.set_title(f"Loss Curve - {exp_name}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    ax2.plot(epochs, history["train_accuracy"], label="Train Acc", marker="o")
    ax2.plot(epochs, history["val_accuracy"], label="Val Acc", marker="o")
    ax2.set_title(f"Accuracy Curve - {exp_name}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "curves_plot.png"))
    plt.close()


def compute_metrics_and_plots(
    y_true: np.ndarray, y_pred: np.ndarray, exp_name: str, save_path: str
) -> dict:
    """Compute metrics and plot the confusion matrix heatmap."""
    # Global metrics
    global_acc = accuracy_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate per-class accuracy and macro accuracy
    class_accuracies = []
    for i in range(43):
        total_samples = cm[i].sum()
        acc = cm[i, i] / total_samples if total_samples > 0 else 0.0
        class_accuracies.append(acc)

    macro_acc = np.mean(class_accuracies)

    # Map class index to name to identify the hardest classes
    class_perf = {GTSRB_CLASSES[i]: class_accuracies[i] for i in range(43)}
    hardest_classes = sorted(class_perf.items(), key=lambda x: x[1])[:5]

    # Plot confusion matrix
    plt.figure(figsize=(22, 18))
    sns.heatmap(cm, annot=False, cmap="magma", cbar=True)
    title_text = (
        f"Confusion Matrix - {exp_name}\n"
        f"Global Acc: {global_acc:.4f} | Macro Acc: {macro_acc:.4f}"
    )
    plt.title(title_text, fontsize=16)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)

    plt.savefig(
        os.path.join(save_path, "confusion_matrix.png"), dpi=200, bbox_inches="tight"
    )
    plt.close()

    return {
        "experiment": exp_name,
        "global_acc": global_acc,
        "macro_acc": macro_acc,
        "hardest_classes": hardest_classes,
        "raw_preds": y_pred,
    }


def main():
    if not os.path.exists(RUNS_DIR):
        print(f"Directory '{RUNS_DIR}' not found. Run experiments first.")
        return

    # Load ground truth labels
    y_true = get_ground_truth_labels()

    experiments = [
        d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))
    ]
    results_summary = []

    print(f"\nProcessing {len(experiments)} experiments...")

    for exp_name in experiments:
        exp_path = os.path.join(RUNS_DIR, exp_name)

        # 1. Load history and plot curves
        history_path = os.path.join(exp_path, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            plot_training_curves(history, exp_name, exp_path)

        # 2. Load predictions and compute metrics
        preds_csv_path = os.path.join(exp_path, "predictions.csv")
        if os.path.exists(preds_csv_path):
            df_preds = pd.read_csv(preds_csv_path, comment="#")
            df_preds = df_preds.sort_values(by="image_index")
            y_pred = df_preds["predicted_class"].values

            metrics = compute_metrics_and_plots(y_true, y_pred, exp_name, exp_path)
            results_summary.append(metrics)

            print(f" -> Experiment '{exp_name}' processed successfully.")

    # 3. Determine the best model and export submission file
    if results_summary:
        print("\n" + "=" * 70)
        print("OVERALL EXPERIMENT RANKING (Sorted by Global Accuracy)")
        print("=" * 70)

        df_ranking = pd.DataFrame(results_summary).sort_values(
            by="global_acc", ascending=False
        )
        for _, row in df_ranking.iterrows():
            exp_name_str = row["experiment"]
            acc_g = row["global_acc"]
            acc_m = row["macro_acc"]
            print(
                f"Model: {exp_name_str:<30} | "
                f"Global Acc: {acc_g:.4f} | "
                f"Macro Acc: {acc_m:.4f}"
            )

        best_exp = df_ranking.iloc[0]
        print("=" * 70)
        print(f"BEST MODEL DETECTED: {best_exp['experiment']}")
        print(
            f"Global Accuracy: {best_exp['global_acc']:.4f} | "
            f"Macro Accuracy: {best_exp['macro_acc']:.4f}"
        )
        print("=" * 70)

        print("\nHardest classes (Top 5 lowest accuracy):")
        for cls_name, acc in best_exp["hardest_classes"]:
            print(f"  - {cls_name}: {acc:.2%} accuracy")

        # Export predictions via the mandatory save_predictions utility
        final_output_name = "JOAO_NATHAN_THIAGO_RESULTS.csv"
        best_preds_tensor = torch.tensor(best_exp["raw_preds"])

        print("\nGenerating final submission file via 'save_predictions'...")
        save_predictions(
            y_pred=best_preds_tensor,
            filepath=final_output_name,
            experiment_name=f"Best Model: {best_exp['experiment']}",
        )
        print(f"Success! Exported '{final_output_name}' ready for submission.")


if __name__ == "__main__":
    main()
