import torch
import torch.nn as nn
import torch.nn.functional as F


class Custom(nn.Module):
    """
    Custom CNN with 3 convolutional blocks and dropout regularization.
    Deeper than the baseline to capture finer spatial features.

    Architecture:
        Conv1(3→32)  → BN → ReLU → Pool  (32x32 → 16x16)
        Conv2(32→64) → BN → ReLU → Pool  (16x16 →  8x8)
        Conv3(64→128)→ BN → ReLU → Pool  ( 8x8  →  4x4)
        Flatten → FC(2048→512) → ReLU → Dropout → FC(512→num_classes)
    """

    def __init__(
        self,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.4,
        image_size: int = 32,
        base_channels: int = 32,
        num_classes: int = 43,
    ) -> None:
        super().__init__()

        self.image_size = (image_size, image_size)

        # Dynamic channel sizes
        c1 = base_channels  # Default 32, Doubled 64
        c2 = base_channels * 2  # Default 64, Doubled 128
        c3 = base_channels * 4  # Default 128, Doubled 256

        # Conv 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1) if use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv 2
        self.conv2 = nn.Conv2d(
            in_channels=c1, out_channels=c2, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(c2) if use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv 3
        self.conv3 = nn.Conv2d(
            in_channels=c2, out_channels=c3, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(c3) if use_batch_norm else nn.Identity()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout2d(p=0.1)

        # Dynamically compute flatten size
        conv_output_size = self._get_conv_output_size()

        # Fully connected
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(512, num_classes)

    def _get_conv_output_size(self) -> int:
        dummy = torch.zeros(1, 3, self.image_size[0], self.image_size[1])
        with torch.no_grad():
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.dropout_conv(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        return int(x.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout_conv(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
