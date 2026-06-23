from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from torch import Tensor
from torch.utils.data import DataLoader

TransformCallable = Callable[[Any], Tensor]
LoadersTuple = Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]

BASE_RUNS_PATH = "results/runs/"


# Configs
@dataclass
class ModelConfig:
    name: str
    dropout: float = 0.0
    data_augmentation: bool = False
    batch_normalization: bool = False
    image_size: int = 32
    base_channels: int = 32
    num_blocks: int = 3


@dataclass
class SchedulerConfig:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


# Hyperparameters
@dataclass
class Hyperparameters:
    epochs: int
    batch_size: int
    optimizer_name: str
    learning_rate: float
    momentum: float = 0.0
    scheduler: Optional[SchedulerConfig] = None


# Results
@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)

    def append(self, t_loss: float, t_acc: float, v_loss: float, v_acc: float) -> None:
        self.train_loss.append(t_loss)
        self.train_accuracy.append(t_acc)
        self.val_loss.append(v_loss)
        self.val_accuracy.append(v_acc)


@dataclass
class TestPredictions:
    y_true: List[int]
    y_pred: List[int]


# Experiments
@dataclass
class ExperimentPlan:
    experiment_name: str
    model_config: ModelConfig
    hyperparameters: Hyperparameters

    @property
    def run_path(self) -> str:
        import os

        return os.path.join(BASE_RUNS_PATH, self.experiment_name)


@dataclass
class ExperimentResult:
    plan: ExperimentPlan
    history: History
    test_predictions: TestPredictions
    model_state_path: Optional[str] = None


@dataclass
class ExperimentSuite:
    experiment_plans: List[ExperimentPlan]

    def __iter__(self):
        return iter(self.experiment_plans)
