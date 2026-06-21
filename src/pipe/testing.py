from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from src import TestPredictions


class Tester:
    def __init__(self, model: nn.Module, device: str):
        self.model = model.to(device)
        self.device = device

    def test(self, test_loader: Any) -> TestPredictions:
        self.model.eval()

        all_preds = []
        all_trues = []

        print("Testing best model...")
        pbar = tqdm(test_loader, desc="[Test]")

        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().tolist())
                all_trues.extend(targets.cpu().tolist())

        correct = sum(p == t for p, t in zip(all_preds, all_trues, strict=True))
        acc = correct / len(all_trues) if all_trues else 0.0
        print(f"Test Accuracy: {acc:.4f}")
        return TestPredictions(y_true=all_trues, y_pred=all_preds)
