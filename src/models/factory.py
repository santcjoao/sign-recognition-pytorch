from typing import Callable, Dict

import torch.nn as nn
import torchvision.models as models

from src.core.types import ModelConfig
from src.models.baseline import Baseline
from src.models.custom import Custom


def _build_baseline(config: ModelConfig, num_classes: int) -> nn.Module:
    return Baseline(
        use_batch_norm=config.batch_normalization,
        image_size=config.image_size,
        num_classes=num_classes,
    )

def _build_custom(config: ModelConfig, num_classes: int) -> nn.Module:
    return Custom(
        use_batch_norm=config.batch_normalization,
        dropout_rate=config.dropout,
        image_size=config.image_size,
        base_channels=config.base_channels,
        num_blocks=config.num_blocks,
        num_classes=num_classes,
    )


MODEL_REGISTRY: Dict[str, Callable[[ModelConfig, int], nn.Module]] = {
    "baseline": _build_baseline,
    "custom": _build_custom,
}


def build_model(config: ModelConfig, num_classes: int = 43) -> nn.Module:
    name = config.name.lower().strip()

    # Support sub-matching case the name contains 'baseline' (ex: 'baseline_batch')
    keys = [
        "baseline",
        "custom",
    ]
    builder = None
    for key in keys:
        if key in name:
            builder = MODEL_REGISTRY.get(key)
            break

    if builder is None:
        supported = ", ".join(f"'{k}'" for k in MODEL_REGISTRY.keys())
        raise ValueError(
            f"Modelo desconhecido: '{config.name}'. Modelos suportados: {supported}"
        )

    return builder(config, num_classes)
