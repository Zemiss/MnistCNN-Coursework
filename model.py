import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """ConvNet for MNIST, following the PyTorch tutorial structure."""

    def __init__(self):
        super().__init__()
        # input: 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 10, 5)  # 10 x 24 x 24
        self.conv2 = nn.Conv2d(10, 20, 3)  # 20 x 10 x 10 after conv1 pool
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 10 x 12 x 12
        out = self.conv2(out)
        out = F.relu(out)  # 20 x 10 x 10
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


# Backward-compatible alias for older runs in this project.
Net = ConvNet
