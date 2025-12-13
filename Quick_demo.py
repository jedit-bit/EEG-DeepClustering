import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((16, 16)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_domains = 5

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

feature_dim = 64
φ = FeatureGenerator(feature_dim=feature_dim).to(device)
C = ClassifierHead(feature_dim=feature_dim, num_classes=num_classes).to(device)
P = ClusteringHead(feature_dim=feature_dim, num_classes=num_classes).to(device)
D = DomainDiscriminator(feature_dim=feature_dim, num_domains=num_domains).to(device)

opt_φ = optim.Adam(φ.parameters(), lr=0.001)
opt_C = optim.Adam(C.parameters(), lr=0.001)
opt_P = optim.Adam(P.parameters(), lr=0.001)
opt_D = optim.Adam(D.parameters(), lr=0.001)

criterion_ce = nn.CrossEntropyLoss()

def create_dictionary(loader, model):
    model.eval()
    representatives = torch.zeros(num_classes, feature_dim, device=device)
    found = torch.zeros(num_classes, dtype=torch.bool)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), 3, -1)
            images = (images - images.mean(dim=2, keepdim=True)) / (images.std(dim=2, keepdim=True) + 1e-6)
            feats = model(images)
            for i in range(images.size(0)):
                c = labels[i].item()
                if not found[c]:
                    representatives[c] = feats[i]
                    found[c] = True
            if found.all():
                break
    return representatives

dictionary_features = create_dictionary(train_loader, φ)

epochs = 10
for epoch in range(epochs):
    φ.train(); C.train(); P.train(); D.train()
    total_loss = 0; total_correct = 0; total_samples = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.size(0), 3, -1)
        images = (images - images.mean(dim=2, keepdim=True)) / (images.std(dim=2, keepdim=True) + 1e-6)
        
        domain_labels = torch.randint(0, num_domains, (images.size(0),), device=device)
        adv_labels = torch.full((images.size(0),), num_domains, device=device)
        
        feats = φ(images).detach()
        logits_D = D(feats)
        loss_D = criterion_ce(logits_D, domain_labels)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()
        
        feats = φ(images)
        logits_C = C(feats)
        loss_C = criterion_ce(logits_C, labels)
        logits_P = P(feats, dictionary_features)
        loss_P = criterion_ce(logits_P, labels)
        logits_D_adv = D(feats)
        loss_adv = criterion_ce(logits_D_adv, adv_labels)
        
        loss = loss_C + loss_P + 0.1*loss_adv
        opt_φ.zero_grad(); opt_C.zero_grad(); opt_P.zero_grad()
        loss.backward()
        opt_φ.step(); opt_C.step(); opt_P.step()
        
        with torch.no_grad():
            sim = torch.matmul(feats, dictionary_features.T)
            preds = sim.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
            total_loss += loss.item() * images.size(0)
    
    dictionary_features = create_dictionary(train_loader, φ)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/total_samples:.4f}, "
          f"Train Acc: {total_correct/total_samples:.4f}")

φ.eval(); P.eval()
total_correct = 0; total_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.size(0), 3, -1)
        images = (images - images.mean(dim=2, keepdim=True)) / (images.std(dim=2, keepdim=True) + 1e-6)
        feats = φ(images)
        sim = torch.matmul(feats, dictionary_features.T)
        preds = sim.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

print(f"My test accuracy: {total_correct/total_samples:.4f}")
