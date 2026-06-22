from typing import Callable, Dict

import torch.nn as nn
import torchvision.models as models

from src.core.types import ModelConfig
from src.models.baseline import Baseline


def _build_baseline(config: ModelConfig, num_classes: int) -> nn.Module:
    return Baseline(
        use_batch_norm=config.batch_normalization,
        dropout_rate=config.dropout,
        num_classes=num_classes,
    )


def _build_resnet18(config: ModelConfig, num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_mobilenet_v3(config: ModelConfig, num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


MODEL_REGISTRY: Dict[str, Callable[[ModelConfig, int], nn.Module]] = {
    "baseline": _build_baseline,
    "resnet18": _build_resnet18,
    "mobilenet_v3_large": _build_mobilenet_v3,
}


def build_model(config: ModelConfig, num_classes: int = 43) -> nn.Module:
    name = config.name.lower().strip()

    # Support sub-matching case the name contains 'baseline' (ex: 'baseline_batch')
    key = "baseline" if "baseline" in name else name
    builder = MODEL_REGISTRY.get(key)

    if builder is None:
        supported = ", ".join(f"'{k}'" for k in MODEL_REGISTRY.keys())
        raise ValueError(
            f"Modelo desconhecido: '{config.name}'. Modelos suportados: {supported}"
        )

    return builder(config, num_classes)
