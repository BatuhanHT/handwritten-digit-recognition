"""MNIST el yazısı rakam tanıma - CNN eğitimi."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout(x.flatten(1))
        return self.fc2(F.relu(self.fc1(x)))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    device = get_device()
    print(f"Cihaz: {device}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data_dir = ROOT / "data"
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000)

    model = CNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{args.epochs}  loss={total / len(train_ds):.4f}  test_acc={acc:.4%}")

    out = ROOT / "models" / "mnist_cnn.pt"
    torch.save(model.state_dict(), out)
    print(f"Model kaydedildi: {out}")


if __name__ == "__main__":
    main()
