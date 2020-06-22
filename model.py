'''
    Write a model for gesture classification.
'''
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv1d(6, 20, 11).double()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(20, 10, 14).double()
        self.conv3 = nn.Conv1d(10, 20, 5).double()
        self.fc1 = nn.Linear(20 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 26)
        self.bn1 = nn.BatchNorm1d(20).double()
        self.bn2 = nn.BatchNorm1d(10).double()
        self.bn3 = nn.BatchNorm1d(20).double()

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = x.view(-1, 20 * 6)
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x