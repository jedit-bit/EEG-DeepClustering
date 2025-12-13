import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import FeatureGenerator, ClassifierHead, ClusteringHead, DomainDiscriminator
from utils import create_dictionary

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 10
num_domains = 5
feature_dim = 64
batch_size = 64
epochs = 30

train_loader, test_loader = get_dataloaders(batch_size)

φ = FeatureGenerator(feature_dim=feature_dim).to(device)
C = ClassifierHead(feature_dim=feature_dim, num_classes=num_classes).to(device)
P = ClusteringHead(feature_dim=feature_dim, num_classes=num_classes).to(device)
D = DomainDiscriminator(feature_dim=feature_dim, num_domains=num_domains).to(device)

opt_φ = optim.Adam(φ.parameters(), lr=0.0005)
opt_C = optim.Adam(C.parameters(), lr=0.001)
opt_P = optim.Adam(P.parameters(), lr=0.001)
opt_D = optim.Adam(D.parameters(), lr=0.001)

criterion_ce = nn.CrossEntropyLoss()

for epoch in range(epochs):
    dictionary_features = create_dictionary(train_loader, φ, num_classes, feature_dim, device)
    φ.train(); C.train(); P.train(); D.train()
    total_loss = 0; total_correct = 0; total_samples = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.size(0), 3, -1)
        images = (images - images.mean(dim=2, keepdim=True)) / (images.std(dim=2, keepdim=True) + 1e-6)

        feats_detach = φ(images).detach()
        domain_labels = torch.randint(0, num_domains, (images.size(0),), device=device)
        logits_D = D(feats_detach)
        loss_D = criterion_ce(logits_D, domain_labels)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        feats = φ(images)
        logits_C = C(feats)
        loss_C = criterion_ce(logits_C, labels)
        logits_P = P(feats, dictionary_features)
        loss_P = criterion_ce(logits_P, labels)
        adv_labels = torch.full((images.size(0),), num_domains, device=device)
        logits_D_adv = D(feats)
        loss_adv = criterion_ce(logits_D_adv, adv_labels)

        loss = loss_C + loss_P + 0.05*loss_adv
        opt_φ.zero_grad(); opt_C.zero_grad(); opt_P.zero_grad()
        loss.backward()
        opt_φ.step(); opt_C.step(); opt_P.step()

        with torch.no_grad():
            sim = torch.matmul(feats, dictionary_features.T)
            preds = sim.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
            total_loss += loss.item() * images.size(0)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/total_samples:.4f}, Train Acc: {total_correct/total_samples:.4f}")

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
