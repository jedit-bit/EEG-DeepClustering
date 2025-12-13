import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar10_loaders(batch_size=64, resize=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

