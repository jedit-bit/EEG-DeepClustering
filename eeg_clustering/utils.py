import torch

def create_dictionary(loader, model, num_classes=10, device='cuda'):
    representatives = torch.zeros(num_classes, 3, 64*64, device=device)
    counts = torch.zeros(num_classes, dtype=torch.bool, device=device)
    
    for images, labels in loader:
        images = images.to(device).view(images.size(0), 3, 64*64)
        images = (images - images.mean(dim=2, keepdim=True)) / (images.std(dim=2, keepdim=True) + 1e-6)
        
        for i in range(images.size(0)):
            c = labels[i].item()
            if not counts[c]:
                representatives[c] = images[i]
                counts[c] = True
        if counts.all():
            break
    
    with torch.no_grad():
        dictionary_features = model(representatives).detach()
    return dictionary_features

