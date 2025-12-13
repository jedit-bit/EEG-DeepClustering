import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureGenerator(nn.Module):
    def __init__(self, input_channels=3, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool = nn.AvgPool1d(2)
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.gru_norm = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, feature_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        x = self.gru_norm(h.squeeze(0))
        x = F.relu(self.bn1(self.fc1(x)))
        x = torch.tanh(self.fc2(x))
        x = F.normalize(x, p=2, dim=1)
        return x

class ClassifierHead(nn.Module):
    def __init__(self, feature_dim=64, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

class ClusteringHead(nn.Module):
    def __init__(self, feature_dim=64, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, feat, dictionary):
        diff = feat.unsqueeze(1) - dictionary.unsqueeze(0)
        diff = diff.view(diff.size(0), -1)
        out = self.fc(diff)
        return out

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=64, num_domains=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains + 1)
        )
        
    def forward(self, x):
        return self.fc(x)
