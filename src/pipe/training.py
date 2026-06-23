from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
)
from tqdm import tqdm

from src import History, Hyperparameters

OPTIMIZERS = {"adam", "sgd", "sgdm"}

SCHEDULERS = {
    "reduce_on_plateau": ReduceLROnPlateau,
    "cosine_annealing": CosineAnnealingLR,
    "step": StepLR,
    "exponential": ExponentialLR,
}


class Trainer:
    def __init__(self, model: nn.Module, hyper: Hyperparameters, device: str):
        self.model = model.to(device)
        self.hyper = hyper
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler: Optional[Any] = self._build_scheduler()

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

    def _build_scheduler(self) -> Optional[Any]:
        cfg = self.hyper.scheduler
        if cfg is None:
            return None

        name = cfg.name.lower().strip()
        scheduler_cls = SCHEDULERS.get(name)

        if scheduler_cls is None:
            supported = ", ".join(f"'{k}'" for k in SCHEDULERS)
            raise ValueError(f"Unknown scheduler: '{cfg.name}'. Supported: {supported}")

        return scheduler_cls(self.optimizer, **cfg.kwargs)

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

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(epoch_val_loss)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Scheduler step → LR: {current_lr:.2e}")

            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_state_dict = copy.deepcopy(self.model.state_dict())
                print(f"Best model saved. (Val Acc: {best_val_acc:.4f})")

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        print(f"Training finished. Best Validation Accuracy: {best_val_acc:.4f}")
        return history
