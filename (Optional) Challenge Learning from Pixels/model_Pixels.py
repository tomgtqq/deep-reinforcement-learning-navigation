import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions, seed):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=10, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv3d(in_channels=10, out_channels=32, kernel_size=3, padding = 1)
        self.pool = nn.MaxPool3d((1,2,2))
        self.fc1 = nn.Linear(4704, num_actions)

    def forward(self, x):
        # print(x.size())
        output = self.pool(F.relu(self.conv1(x)))
        # print(output.size())
        output = self.pool(F.relu(self.conv2(output)))
        output = output.contiguous().view(output.size(0), -1)
        # print(output.size())
        output = self.fc1(output)

        return output
