import torch
import torch.nn.functional as F
import torch.optim as optim
from eeg_clustering.dataset import get_cifar10_loaders
from eeg_clustering.model import FeatureGenerator, ClassifierHead, ClusteringHead, DomainDiscriminator
from eeg_clustering.utils import create_dictionary

def train_and_evaluate(epochs=10, batch_size=64, feature_dim=64, num_classes=10, num_domains=5, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    
    φ = FeatureGenerator(feature_dim=feature_dim).to(device)
    C = ClassifierHead(feature_dim=feature_dim, num_classes=num_classes).to(device)
    P = ClusteringHead(feature_dim=feature_dim, num_classes=num_classes).to(device)
    D = DomainDiscriminator(feature_dim=feature_dim, num_domains=num_domains).to(device)
    
    opt_φ = optim.Adam(φ.parameters(), lr=0.001)
    opt_C = optim.Adam(C.parameters(), lr=0.001)
    opt_P = optim.Adam(P.parameters(), lr=0.001)
    opt_D = optim.Adam(D.parameters(), lr=0.001)
    
    criterion_ce = torch.nn.CrossEntropyLoss()
    
    dictionary_features = create_dictionary(train_loader, φ, feature_dim, num_classes, device)
    
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
        
        dictionary_features = create_dictionary(train_loader, φ, feature_dim, num_classes, device)
        
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

if __name__ == "__main__":
    train_and_evaluate()
