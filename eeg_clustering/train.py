import torch
import torch.nn.functional as F
import torch.optim as optim
from eeg_clustering.dataset import get_cifar10_loaders
from eeg_clustering.model import FeatureGenerator, ClusteringClassifier
from eeg_clustering.utils import create_dictionary

def train_and_evaluate(
    epochs=30, batch_size=64, feature_dim=128, num_classes=10, device=None
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    
    feature_model = FeatureGenerator(feature_dim=feature_dim).to(device)
    cluster_model = ClusteringClassifier(feature_dim=feature_dim, num_classes=num_classes).to(device)
    
    opt_feat = optim.Adam(feature_model.parameters(), lr=0.001)
    opt_cluster = optim.Adam(cluster_model.parameters(), lr=0.001)
    
    dictionary_features = create_dictionary(train_loader, feature_model, num_classes=num_classes, device=device)
    
    for epoch in range(epochs):
        feature_model.train()
        cluster_model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), 3, 64*64)
            images = (images - images.mean(dim=2, keepdim=True)) / (images.std(dim=2, keepdim=True) + 1e-6)
            
            features = feature_model(images)
            diff_vectors = torch.cat([features - dictionary_features[c].expand(features.size(0), -1)
                                      for c in range(num_classes)], dim=1)
            
            cluster_probs = cluster_model(diff_vectors)
            target = F.one_hot(labels, num_classes=num_classes).float()
            
            loss = F.binary_cross_entropy(cluster_probs, target)
            
            opt_feat.zero_grad(); opt_cluster.zero_grad()
            loss.backward()
            opt_feat.step(); opt_cluster.step()
            
            total_loss += loss.item() * images.size(0)
            total_correct += (cluster_probs.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/total_samples:.4f}, "
              f"Train Acc: {total_correct/total_samples:.4f}")
    

    feature_model.eval()
    cluster_model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), 3, 64*64)
            images = (images - images.mean(dim=2, keepdim=True)) / (images.std(dim=2, keepdim=True) + 1e-6)
            
            features = feature_model(images)
            diff_vectors = torch.cat([features - dictionary_features[c].expand(features.size(0), -1)
                                      for c in range(num_classes)], dim=1)
            cluster_probs = cluster_model(diff_vectors)
            
            total_correct += (cluster_probs.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)
    
    print(f"My test accuracy: {total_correct/total_samples:.4f}")

if __name__ == "__main__":
    train_and_evaluate()

