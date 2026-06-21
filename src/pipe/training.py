from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from src import History, Hyperparameters

OPTIMIZERS = {"adam", "sgd", "sgdm"}


class Trainer:
    def __init__(self, model: nn.Module, hyper: Hyperparameters, device: str):
        self.model = model.to(device)
        self.hyper = hyper
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_name = self.hyper.optimizer_name.lower().strip()
        if opt_name not in OPTIMIZERS:
            opts = ", ".join(OPTIMIZERS)
            raise ValueError(
                f"Erro: '{opt_name}' não é um otimizador válido. Escolha entre: {opts}"
            )

        if opt_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=self.hyper.learning_rate
            )
        elif opt_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.hyper.learning_rate)
        else:
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.hyper.learning_rate,
                momentum=self.hyper.momentum,
            )

    def train(self, train_loader: Any, val_loader: Any) -> History:
        history = History()
        best_val_acc = 0.0

        import copy

        best_state_dict = None

        print(
            f"Treinando por {self.hyper.epochs} épocas com "
            f"{self.hyper.optimizer_name.upper()}..."
        )

        for epoch in range(1, self.hyper.epochs + 1):
            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.hyper.epochs} [Train]")

            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total

            # Save history
            history.append(
                epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc
            )

            print(
                f"Epoch Results {epoch}: "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Train Acc: {epoch_train_acc:.4f} || "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Val Acc: {epoch_val_acc:.4f}"
            )

            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_state_dict = copy.deepcopy(self.model.state_dict())
                print(f"Best model saved. (Val Acc: {best_val_acc:.4f})")

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        print(f"Training finished. Best Validation Accuracy: {best_val_acc:.4f}")
        return history
