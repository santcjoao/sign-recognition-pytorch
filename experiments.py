import torchvision.transforms as T

from src import (
    ExperimentPlan,
    ExperimentSuite,
    Hyperparameters,
    ModelConfig,
    Pipeline,
)

# ==============================================================================
# 1. CONFIGURAÇÕES DOS MODELOS
# ==============================================================================
BASELINE = ModelConfig(
    name="baseline",
    data_augmentation=False,
    batch_normalization=False,
)

BASELINE_BN_AUG = ModelConfig(
    name="baseline_bn_aug",
    data_augmentation=True,
    batch_normalization=True,
)

BASELINE_BN = ModelConfig(
    name="baseline_bn", data_augmentation=False, batch_normalization=True
)

BASELINE_AUG = ModelConfig(
    name="baseline_aug", data_augmentation=True, batch_normalization=False
)

# ==============================================================================
# 2. HIPERPARAMETRIZAÇÕES
# ==============================================================================
WITH_ADAM = Hyperparameters(
    epochs=25, batch_size=64, optimizer_name="adam", learning_rate=0.001
)
WITH_SGD = Hyperparameters(
    epochs=25, batch_size=64, optimizer_name="sgd", learning_rate=0.01
)
WITH_SGDM = Hyperparameters(
    epochs=25, batch_size=64, optimizer_name="sgdm", learning_rate=0.01, momentum=0.9
)


# ==============================================================================
# 3. HELPER & DECLARAÇÃO DOS EXPERIMENTOS
# ==============================================================================
def create_plan(model: ModelConfig, hp: Hyperparameters) -> ExperimentPlan:
    return ExperimentPlan(
        experiment_name=f"{model.name}_{hp.optimizer_name}",
        model_config=model,
        hyperparameters=hp,
    )


# Planos Individuais do Baseline
BASELINE_ADAM = create_plan(BASELINE, WITH_ADAM)
BASELINE_SGD = create_plan(BASELINE, WITH_SGD)
BASELINE_SGDM = create_plan(BASELINE, WITH_SGDM)

# Planos Individuais do Baseline com Batch/Augmentation
BASELINE_BATCH_ADAM = create_plan(BASELINE_BN_AUG, WITH_ADAM)
BASELINE_BATCH_SGD = create_plan(BASELINE_BN_AUG, WITH_SGD)
BASELINE_BATCH_SGDM = create_plan(BASELINE_BN_AUG, WITH_SGDM)

# ==============================================================================
# 4. DEFINIÇÃO DAS SUÍTES DE EXECUÇÃO
# ==============================================================================
BASELINE_SUITE = ExperimentSuite([BASELINE_ADAM, BASELINE_SGD, BASELINE_SGDM])
BASELINE_BATCH_SUITE = ExperimentSuite(
    [BASELINE_BATCH_ADAM, BASELINE_BATCH_SGD, BASELINE_BATCH_SGDM]
)
ALL = ExperimentSuite(
    BASELINE_SUITE.experiment_plans + BASELINE_BATCH_SUITE.experiment_plans
)
FAST = ExperimentSuite([BASELINE_ADAM])

# ==============================================================================
# 5. AUGMENTATIONS CUSTOMIZADAS
# ==============================================================================
EXAMPLE_TRANSFORM = T.Compose(
    [
        T.RandomRotation(15),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomCrop(28, padding=2),
        T.RandomHorizontalFlip(
            p=0.5
        ),  # pode não ser bom considerando placas que indiquem direção
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ]
)
IDENTITY_TRANSFORM = T.Compose([])

# ==============================================================================
# 6. ORQUESTRAÇÃO
# ==============================================================================
if __name__ == "__main__":
    from torchvision.datasets import GTSRB

    print("Verifying/Downloading GTSRB dataset...")
    GTSRB(root="./data", split="train", download=True)
    GTSRB(root="./data", split="test", download=True)
    print("Dataset verified successfully!\n")

    # TODO
    # Faze experimento todos juntos e depois treinar o
    # melhor modelo individualmente?
    CHOOSEN_SUITE = BASELINE_SUITE
    CHOOSEN_TRANSFORM = IDENTITY_TRANSFORM

    pipeline = Pipeline(suite=CHOOSEN_SUITE, transform=CHOOSEN_TRANSFORM)
    pipeline.run()
