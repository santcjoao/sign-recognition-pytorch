"""GTSRB — carregamento de dados e exportacao de predicoes.

NAO MODIFIQUE ESTE ARQUIVO.
Ele garante que todos os grupos usem o mesmo split de dados e o mesmo
formato de saida, permitindo comparacao justa entre os trabalhos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB

# ======================================================================
# Constantes
# ======================================================================

NUM_CLASSES: int = 43

GTSRB_CLASSES: dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh > 3.5t",
    11: "Right-of-way next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5t",
}

# ======================================================================
# Split deterministico
# ======================================================================

_SEED: int = 42
_VAL_RATIO: float = 0.2


def _get_split_indices(dataset_size: int) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(_SEED)
    indices = rng.permutation(dataset_size).tolist()
    split = int(dataset_size * (1 - _VAL_RATIO))
    return indices[:split], indices[split:]


# ======================================================================
# Carregamento de dados
# ======================================================================

_MEAN = [0.3403, 0.3121, 0.3214]
_STD = [0.2724, 0.2608, 0.2669]


def get_dataloaders(
    img_size: int = 32,
    batch_size: int = 128,
    train_transform: transforms.Compose | None = None,
    num_workers: int = 2,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Cria data loaders de treino, validacao e teste.

    O split train/val e fixo (seed=42, 80/20) para todos os grupos.

    Parameters
    ----------
    img_size : int
        Tamanho da imagem (quadrada).
    batch_size : int
        Tamanho do batch.
    train_transform : transforms.Compose | None
        Transform customizado para treino (data augmentation, etc.).
        Se ``None``, usa apenas resize + normalize.
    num_workers : int
        Workers do DataLoader.
    data_root : str
        Diretorio raiz para download do dataset.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``

    Example
    -------
    >>> train_loader, val_loader, test_loader = get_dataloaders(img_size=32)
    """
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )

    if train_transform is None:
        train_transform = eval_transform

    full_train = GTSRB(
        root=data_root, split="train", transform=train_transform, download=True
    )
    val_base = GTSRB(
        root=data_root, split="train", transform=eval_transform, download=True
    )
    test_dataset = GTSRB(
        root=data_root, split="test", transform=eval_transform, download=True
    )

    train_idx, val_idx = _get_split_indices(len(full_train))

    train_loader = DataLoader(
        Subset(full_train, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_base, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ======================================================================
# Exportacao de predicoes
# ======================================================================


def save_predictions(
    y_pred: np.ndarray | torch.Tensor,
    filepath: str | Path,
    experiment_name: str = "",
) -> None:
    """Salva predicoes em CSV padronizado para entrega.

    O arquivo gerado tem exatamente 12.630 linhas de dados (uma por
    imagem de teste), com o formato::

        image_index,predicted_class
        0,14
        1,1
        ...
        12629,35

    Parameters
    ----------
    y_pred : np.ndarray | torch.Tensor
        Classes preditas, shape ``(N,)``.
    filepath : str | Path
        Caminho do CSV de saida.
    experiment_name : str
        Nome do experimento (gravado como comentario no cabecalho).

    Example
    -------
    >>> save_predictions(y_pred, "results/predicoes_baseline.csv",
    ...                  experiment_name="Baseline CNN SGD")
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        if experiment_name:
            f.write(f"# experiment: {experiment_name}\n")
        f.write("image_index,predicted_class\n")
        for idx, pred in enumerate(y_pred):
            f.write(f"{idx},{int(pred)}\n")
