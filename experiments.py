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
BASELINE_TRANSFORM = T.Compose(
    [
        T.RandomRotation(15),
        T.RandomAffine(degrees=0.1, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ]
)

CUSTOM_TRANSFORM = T.Compose(
    [
        T.RandomAutocontrast(p=0.5),
        T.RandomEqualize(p=0.5),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        T.RandomPerspective(distortion_scale=0.2, p=0.2),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    ]
)


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

CUSTOM_MODEL = ModelConfig(
    name="custom",
    dropout=0.3,
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

ADAM_CUSTOM = Hyperparameters(
    epochs=25,
    batch_size=64,
    optimizer_name="adam",
    learning_rate=0.001,
    label_smoothing=0.0,
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={"T_max": 25, "eta_min": 1e-6},
    ),
)

OPTIMIZERS = [ADAM, SGD, SGDM]


# ==============================================================================
# 5. EXPERIMENT SUITES
# ==============================================================================
# Comparative suites
COMPARE_OPTIMIZERS_SUITE = MAKE_SUITE(BASELINE_BN, OPTIMIZERS)
COMPARE_NORM_SUITE = ExperimentSuite(
    [CREATE_PLAN(BASELINE, ADAM), CREATE_PLAN(BASELINE_BN, ADAM)]
)

# Single run suites
RUN_BASELINE_ADAM = MAKE_SUITE(BASELINE, [ADAM])
RUN_CUSTOM_ADAM = MAKE_SUITE(CUSTOM_MODEL, [ADAM_CUSTOM])


# ==============================================================================
# 6. ORCHESTRATION
# ==============================================================================
if __name__ == "__main__":
    from torchvision.datasets import GTSRB

    from src.utils import parse_experiment_args

    args = parse_experiment_args()

    print("Configuring seeds for reproducibility...")
    set_seed(42)

    print("Verifying/Downloading GTSRB dataset...")
    GTSRB(root="./data", split="train", download=True)
    GTSRB(root="./data", split="test", download=True)
    print("Dataset verified successfully!\n")

    if args.baseline:
        print(">>> Starting BASELINE run...")
        pipeline = Pipeline(suite=RUN_BASELINE_ADAM, transform=BASELINE_TRANSFORM)
        pipeline.run()

    if args.custom:
        print(">>> Starting CUSTOM run...")
        pipeline = Pipeline(suite=RUN_CUSTOM_ADAM, transform=CUSTOM_TRANSFORM)
        pipeline.run()

    if args.compare_optimizers:
        print(
            ">>> Starting COMPARE-OPTIMIZERS run (baseline_bn with Adam/SGD/SGDm)..."
        )
        pipeline = Pipeline(suite=COMPARE_OPTIMIZERS_SUITE, transform=BASELINE_TRANSFORM)
        pipeline.run()

    if args.compare_norm:
        print(
            ">>> Starting COMPARE-NORMALIZATION run (baseline vs baseline_bn)..."
        )
        pipeline = Pipeline(suite=COMPARE_NORM_SUITE, transform=BASELINE_TRANSFORM)
        pipeline.run()

