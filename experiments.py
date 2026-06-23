import random

import numpy as np
import torch
import torchvision.transforms as T

from src import (
    ExperimentPlan,
    ExperimentSuite,
    Hyperparameters,
    ModelConfig,
    Pipeline,
    SchedulerConfig,
)


# ==============================================================================
# 0. REPRODUCIBILITY
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# 1. HELPERS
# ==============================================================================
def CREATE_PLAN(model: ModelConfig, hp: Hyperparameters) -> ExperimentPlan:
    return ExperimentPlan(
        experiment_name=f"{model.name}_{hp.optimizer_name}",
        model_config=model,
        hyperparameters=hp,
    )


def MAKE_SUITE(
    model: ModelConfig, optimizers: list[Hyperparameters]
) -> ExperimentSuite:
    return ExperimentSuite([CREATE_PLAN(model, hp) for hp in optimizers])


def COMBINE_SUITES(*suites: ExperimentSuite) -> ExperimentSuite:
    return ExperimentSuite([plan for suite in suites for plan in suite])


# ==============================================================================
# 2. DATA AUGMENTATIONS
# ==============================================================================
CUSTOM_TRANSFORM = T.Compose(
    [
        T.RandomEqualize(p=0.5),
        T.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ]
)
IDENTITY_TRANSFORM = T.Compose([])


# ==============================================================================
# 3. MODEL CONFIGS
# ==============================================================================
BASELINE = ModelConfig(
    name="baseline",
    data_augmentation=False,
    batch_normalization=False,
)

BASELINE_BN = ModelConfig(
    name="baseline_bn",
    data_augmentation=False,
    batch_normalization=True,
)

BASELINE_AUG = ModelConfig(
    name="baseline_aug",
    data_augmentation=True,
    batch_normalization=False,
)

BASELINE_BN_AUG = ModelConfig(
    name="baseline_bn_aug",
    data_augmentation=True,
    batch_normalization=True,
)

CUSTOM = ModelConfig(
    name="custom",
    dropout=0.5,
    data_augmentation=True,
    batch_normalization=True,
    image_size=32,
    base_channels=32,
)

CUSTOM_48 = ModelConfig(
    name="custom_48",
    dropout=0.5,
    data_augmentation=True,
    batch_normalization=True,
    image_size=48,
    base_channels=64,
)

CUSTOM_DEEP = ModelConfig(
    name="custom_deep",
    dropout=0.5,
    data_augmentation=True,
    batch_normalization=True,
    image_size=48,
    base_channels=64,
    num_blocks=4,
)


# ==============================================================================
# 4. HYPERPARAMETERS
# ==============================================================================
ADAM = Hyperparameters(
    epochs=25, batch_size=64, optimizer_name="adam", learning_rate=0.001
)
SGD = Hyperparameters(
    epochs=25, batch_size=64, optimizer_name="sgd", learning_rate=0.01
)
SGDM = Hyperparameters(
    epochs=25, batch_size=64, optimizer_name="sgdm", learning_rate=0.01, momentum=0.9
)

# Custom model: more epochs + ReduceLROnPlateau
ADAM_SCHEDULED = Hyperparameters(
    epochs=40,
    batch_size=64,
    optimizer_name="adam",
    learning_rate=0.001,
    scheduler=SchedulerConfig(
        name="reduce_on_plateau",
        kwargs={"mode": "min", "factor": 0.5, "patience": 4},
    ),
)

ADAM_COSINE = Hyperparameters(
    epochs=25,
    batch_size=64,
    optimizer_name="adam",
    learning_rate=0.001,
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={"T_max": 50, "eta_min": 1e-6},
    ),
)

OPTIMIZERS = [ADAM, SGD, SGDM]


# ==============================================================================
# 5. EXPERIMENT SUITES
# ==============================================================================
BASELINE_SUITE = MAKE_SUITE(BASELINE, OPTIMIZERS)
BASELINE_BN_SUITE = MAKE_SUITE(BASELINE_BN, OPTIMIZERS)
BASELINE_AUG_SUITE = MAKE_SUITE(BASELINE_AUG, OPTIMIZERS)
BASELINE_BN_AUG_SUITE = MAKE_SUITE(BASELINE_BN_AUG, OPTIMIZERS)
CUSTOM_SUITE = MAKE_SUITE(CUSTOM, [ADAM_SCHEDULED, SGD, SGDM])

ALL = COMBINE_SUITES(
    BASELINE_SUITE,
    BASELINE_BN_SUITE,
    BASELINE_AUG_SUITE,
    BASELINE_BN_AUG_SUITE,
    CUSTOM_SUITE,
)

FAST = MAKE_SUITE(BASELINE, [ADAM])
FAST_CUSTOM = MAKE_SUITE(CUSTOM, [ADAM_SCHEDULED])
FAST_CUSTOM_48 = MAKE_SUITE(CUSTOM_48, [ADAM_SCHEDULED])
FAST_CUSTOM_DEEP = MAKE_SUITE(CUSTOM_DEEP, [ADAM_COSINE])


# ==============================================================================
# 6. ORCHESTRATION
# ==============================================================================
if __name__ == "__main__":
    from torchvision.datasets import GTSRB

    print("Configuring seeds for reproducibility...")
    set_seed(42)

    print("Verifying/Downloading GTSRB dataset...")
    GTSRB(root="./data", split="train", download=True)
    GTSRB(root="./data", split="test", download=True)
    print("Dataset verified successfully!\n")

    # Select experiment suite and transforms for execution
    CHOOSEN_SUITE = FAST_CUSTOM_DEEP
    CHOOSEN_TRANSFORM = CUSTOM_TRANSFORM

    pipeline = Pipeline(suite=CHOOSEN_SUITE, transform=CHOOSEN_TRANSFORM)
    pipeline.run()
