import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision.datasets import GTSRB

from src import BASE_TRANSFORM, GTSRB_CLASSES, get_dataloaders, save_predictions

RUNS_DIR = "results/runs"
COMPARATIVES_DIR = "results/comparatives"


def get_ground_truth_labels():
    """Extract ground truth labels from the test DataLoader."""
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

    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    ax1.set_title(f"Loss Curve - {exp_name}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

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


def plot_easiest_hardest_classes(class_accuracies: list, exp_name: str, save_path: str):
    """Plot the Top 5 and Bottom 5 classes by accuracy."""
    class_perf = [(GTSRB_CLASSES[i], acc) for i, acc in enumerate(class_accuracies)]
    class_perf.sort(key=lambda x: x[1])

    hardest = class_perf[:5]
    easiest = class_perf[-5:]

    # Combine for plotting (invert order for horizontal bar chart logic)
    to_plot = hardest + easiest[::-1]
    labels = [x[0] for x in to_plot]
    values = [x[1] for x in to_plot]
    colors = ["red"] * 5 + ["green"] * 5

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color=colors)
    plt.axvline(
        x=np.mean(class_accuracies), color="blue", linestyle="--", label="Macro Acc"
    )
    plt.title(f"Top 5 Hardest (Red) and Easiest (Green) Classes - {exp_name}")
    plt.xlabel("Accuracy")
    plt.xlim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "easiest_hardest_classes.png"))
    plt.close()


def plot_accuracy_vs_support(
    class_counts: list, class_accuracies: list, exp_name: str, save_path: str
):
    """Scatter plot: Accuracy vs. Number of examples in the test set."""
    plt.figure(figsize=(10, 6))
    plt.scatter(class_counts, class_accuracies, alpha=0.7, color="purple")
    plt.title(f"Accuracy vs. Number of Examples - {exp_name}")
    plt.xlabel("Number of Examples in Test Set")
    plt.ylabel("Class Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "acc_vs_support.png"))
    plt.close()


def plot_prediction_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exp_name: str,
    save_path: str,
    raw_test_dataset: GTSRB,
):
    """Plot a grid of 5 correct and 5 incorrect predictions using actual images."""
    print(f"[{exp_name}] Generating image examples plot...")

    correct_idx = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]

    if len(correct_idx) >= 5 and len(incorrect_idx) >= 5:
        sampled_correct = random.sample(list(correct_idx), 5)
        sampled_incorrect = random.sample(list(incorrect_idx), 5)

        fig, axes = plt.subplots(2, 5, figsize=(20, 9))
        fig.suptitle(f"Prediction Examples - {exp_name}", fontsize=16)

        # Row 0 — correct predictions (green)
        axes[0, 0].set_ylabel("Corretas", fontsize=11, color="green", labelpad=8)
        for i, idx in enumerate(sampled_correct):
            img, _ = raw_test_dataset[idx]
            ax = axes[0, i]
            ax.imshow(img)
            ax.set_title(
                f"TRUE & PRED:\n{GTSRB_CLASSES[y_true[idx]]}",
                color="green",
                fontsize=9,
            )
            ax.axis("off")

        # Row 1 — incorrect predictions (red)
        axes[1, 0].set_ylabel("Incorretas", fontsize=11, color="red", labelpad=8)
        for i, idx in enumerate(sampled_incorrect):
            img, _ = raw_test_dataset[idx]
            ax = axes[1, i]
            ax.imshow(img)
            ax.set_title(
                f"TRUE: {GTSRB_CLASSES[y_true[idx]]}\nPRED: {GTSRB_CLASSES[y_pred[idx]]}",
                color="red",
                fontsize=9,
            )
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "prediction_examples.png"))
        plt.close()
    else:
        print(f"[{exp_name}] Not enough correct or incorrect samples to plot examples.")


def plot_all_class_accuracies(class_accuracies: list, exp_name: str, save_path: str):
    """Barplot horizontal com acurácia de todas as 43 classes."""
    labels = [GTSRB_CLASSES[i] for i in range(43)]
    colors = [
        "green" if acc >= np.mean(class_accuracies) else "red"
        for acc in class_accuracies
    ]

    plt.figure(figsize=(10, 14))
    plt.barh(labels, class_accuracies, color=colors)
    plt.axvline(
        x=np.mean(class_accuracies),
        color="blue",
        linestyle="--",
        label=f"Macro Acc: {np.mean(class_accuracies):.2%}",
    )
    plt.title(f"Acurácia por Classe (todas as 43) - {exp_name}")
    plt.xlabel("Acurácia")
    plt.xlim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "all_class_accuracies.png"))
    plt.close()


def compute_metrics_and_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exp_name: str,
    save_path: str,
    raw_test_dataset: GTSRB,
) -> dict:
    """Compute all metrics and generate individual experiment plots."""
    global_acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(43)))

    class_accuracies = []
    class_counts = []
    for i in range(43):
        total_samples = cm[i].sum()
        class_counts.append(total_samples)
        acc = cm[i, i] / total_samples if total_samples > 0 else 0.0
        class_accuracies.append(acc)
    macro_acc = np.mean(class_accuracies)

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(22, 18))
    sns.heatmap(cm, annot=False, cmap="magma", cbar=True)
    plt.title(
        f"Confusion Matrix - {exp_name}\nGlobal Acc: {global_acc:.4f} | Macro Acc: {macro_acc:.4f}",
        fontsize=16,
    )
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.savefig(
        os.path.join(save_path, "confusion_matrix.png"), dpi=200, bbox_inches="tight"
    )
    plt.close()

    # 2. Hardest/Easiest plot
    plot_easiest_hardest_classes(class_accuracies, exp_name, save_path)

    # 3. Accuracy vs Support plot
    plot_accuracy_vs_support(class_counts, class_accuracies, exp_name, save_path)

    # 4. Plot all class' acuracies individually
    plot_all_class_accuracies(class_accuracies, exp_name, save_path)

    # 5. Image Examples Plot
    plot_prediction_examples(y_true, y_pred, exp_name, save_path, raw_test_dataset)

    # Calculate Top Confusions
    confusions = []
    for i in range(43):
        for j in range(43):
            if i != j and cm[i, j] > 0:
                confusions.append((GTSRB_CLASSES[i], GTSRB_CLASSES[j], cm[i, j]))
    top_confusions = sorted(confusions, key=lambda x: x[2], reverse=True)[:5]

    class_perf = {GTSRB_CLASSES[i]: class_accuracies[i] for i in range(43)}
    hardest_classes = sorted(class_perf.items(), key=lambda x: x[1])[:5]

    return {
        "experiment": exp_name,
        "global_acc": global_acc,
        "macro_acc": macro_acc,
        "hardest_classes": hardest_classes,
        "top_confusions": top_confusions,
        "raw_preds": y_pred,
    }


def generate_comparative_plots(df: pd.DataFrame):
    """Generate comparative plots isolating optimizers, BN, and Augmentation."""
    df = df.copy()

    os.makedirs(COMPARATIVES_DIR, exist_ok=True)
    print(f"\nGenerating comparative plots in '{COMPARATIVES_DIR}'...")

    # Extract features from experiment names
    df["optimizer"] = df["experiment"].apply(lambda x: x.split("_")[-1].upper())
    df["has_bn"] = df["experiment"].apply(
        lambda x: "Com BN" if "_bn" in x else "Sem BN"
    )
    df["has_aug"] = df["experiment"].apply(
        lambda x: "Com Aug" if "_aug" in x else "Sem Aug"
    )

    # 1. Optimizers comparative
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="optimizer", y="global_acc", errorbar=None, palette="Set2")
    plt.title("Comparativo: Otimizadores (Acurácia Média Global)")
    plt.ylim(df["global_acc"].min() - 0.05, min(1.0, df["global_acc"].max() + 0.05))
    plt.savefig(os.path.join(COMPARATIVES_DIR, "compare_optimizers.png"))
    plt.close()

    # 2. Batch Normalization comparative
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="has_bn", y="global_acc", errorbar=None, palette="Set1")
    plt.title("Comparativo: Batch Normalization vs Baseline")
    plt.ylim(df["global_acc"].min() - 0.05, min(1.0, df["global_acc"].max() + 0.05))
    plt.savefig(os.path.join(COMPARATIVES_DIR, "compare_batchnorm.png"))
    plt.close()

    # 3. Data Augmentation comparative
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="has_aug", y="global_acc", errorbar=None, palette="Set3")
    plt.title("Comparativo: Data Augmentation vs Baseline")
    plt.ylim(df["global_acc"].min() - 0.05, min(1.0, df["global_acc"].max() + 0.05))
    plt.savefig(os.path.join(COMPARATIVES_DIR, "compare_augmentation.png"))
    plt.close()


def main():
    if not os.path.exists(RUNS_DIR):
        print(f"Directory '{RUNS_DIR}' not found. Run experiments first.")
        return

    y_true = get_ground_truth_labels()

    print("Loading raw test dataset for image examples...")
    raw_test_dataset = GTSRB(root="./data", split="test", download=True)

    experiments = [
        d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))
    ]
    results_summary = []

    print(f"\nProcessing {len(experiments)} experiments...")

    for exp_name in experiments:
        exp_path = os.path.join(RUNS_DIR, exp_name)

        history_path = os.path.join(exp_path, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            plot_training_curves(history, exp_name, exp_path)

        preds_csv_path = os.path.join(exp_path, "predictions.csv")
        if os.path.exists(preds_csv_path):
            df_preds = pd.read_csv(preds_csv_path, comment="#")
            df_preds = df_preds.sort_values(by="image_index")
            y_pred = df_preds["predicted_class"].values

            metrics = compute_metrics_and_plots(
                y_true, y_pred, exp_name, exp_path, raw_test_dataset
            )
            results_summary.append(metrics)
            print(f" -> Experiment '{exp_name}' processed successfully.")

    if results_summary:
        df_ranking = pd.DataFrame(results_summary).sort_values(
            by="global_acc", ascending=False
        )

        # Drop complex objects (raw_preds, top_confusions, hardest_classes) for the CSV table
        df_csv = df_ranking.drop(
            columns=["raw_preds", "top_confusions", "hardest_classes"]
        )
        csv_out_path = os.path.join("results", "metrics_summary.csv")
        df_csv.to_csv(csv_out_path, index=False)
        print(f"\n=> General metrics table saved to: {csv_out_path}")

        # Generate the overarching comparative plots
        generate_comparative_plots(df_ranking)

        best_exp = df_ranking.iloc[0]

        print("\n" + "=" * 80)
        print(f"BEST MODEL DETECTED: {best_exp['experiment']}")
        print(
            f"Global Accuracy: {best_exp['global_acc']:.4f} | Macro Accuracy: {best_exp['macro_acc']:.4f}"
        )
        print("=" * 80)

        print("\n[!] Top 5 Erros Frequentes (Pares de Confusão):")
        for real_cls, pred_cls, count in best_exp["top_confusions"]:
            print(
                f"  -> '{real_cls}' foi incorretamente previsto como '{pred_cls}' {count} vezes."
            )

        print("\n[!] Top 5 Classes Mais Difíceis (Menor Acurácia):")
        for cls_name, acc in best_exp["hardest_classes"]:
            print(f"  -> {cls_name}: {acc:.2%}")

        # Export predictions using the mandatory format
        final_output_name = "JOAO_NATHAN_THIAGO_RESULTS.csv"
        best_preds_tensor = torch.tensor(best_exp["raw_preds"])

        print(f"\nGenerating final submission file '{final_output_name}'...")
        save_predictions(
            y_pred=best_preds_tensor,
            filepath=final_output_name,
            experiment_name=f"Best Model: {best_exp['experiment']}",
        )
        print("Done!")


if __name__ == "__main__":
    main()
