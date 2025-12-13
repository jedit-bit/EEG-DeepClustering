import torch

def create_dictionary(loader, model, feature_dim=64, num_classes=10, device='cuda'):
    model.eval()
    representatives = torch.zeros(num_classes, feature_dim, device=device)
    found = torch.zeros(num_classes, dtype=torch.bool, device=device)
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
