import torch
import torch.nn as nn
import torch.nn.functional as F


class Custom(nn.Module):
    def __init__(
        self,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        image_size: int = 32,
        num_classes: int = 43,
    ) -> None:
        super().__init__()

        self.image_size = (image_size, image_size)

        # Conv 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv 2
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Get flatten size
        conv_output_size = self._get_conv_output_size()

        # FC
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(256, num_classes)

    def _get_conv_output_size(self) -> int:
        dummy = torch.zeros(1, 3, self.image_size[0], self.image_size[1])
        with torch.no_grad():
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
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

        # Flatten
        x = torch.flatten(x, 1)

        # FC
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
