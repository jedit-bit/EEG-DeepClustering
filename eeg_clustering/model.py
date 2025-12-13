import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureGenerator(nn.Module):
    def __init__(self, input_channels=3, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, input_channels, 1, padding='same')
        self.conv2 = nn.Conv1d(input_channels, input_channels, 2, padding='same')
        self.conv3 = nn.Conv1d(input_channels, input_channels, 3, padding='same')
        self.conv4 = nn.Conv1d(input_channels, input_channels, 1, padding='same')
        self.gru = nn.GRU(input_channels, 128, batch_first=True)
        self.gru_norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128, 150)
        self.fc2 = nn.Linear(150, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.fc4 = nn.Linear(300, 200)
        self.fc5 = nn.Linear(200, 100)
        self.fc6 = nn.Linear(100, feature_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        x = self.gru_norm(h.squeeze(0))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.bn1(x)
        x = F.relu(self.fc3(x))
        x = self.bn2(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = F.normalize(x, p=2, dim=1)
        return x

class ClusteringClassifier(nn.Module):
    def __init__(self, feature_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim * num_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, diff_vectors):
        x = F.relu(self.fc1(diff_vectors))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

