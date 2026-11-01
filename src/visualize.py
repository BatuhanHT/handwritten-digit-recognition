"""Kayıtlı modelle test görüntülerinde tahmin yapar ve sonucu PNG olarak kaydeder."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from train import CNN, ROOT, get_device

device = get_device()
model = CNN().to(device)
model.load_state_dict(torch.load(ROOT / "models" / "mnist_cnn.pt", map_location=device))
model.eval()

tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
ds = datasets.MNIST(ROOT / "data", train=False, download=True, transform=tfm)

n = 20
x = torch.stack([ds[i][0] for i in range(n)]).to(device)
with torch.no_grad():
    pred = model(x).argmax(1).cpu()

fig, axes = plt.subplots(4, 5, figsize=(8, 7))
for ax, i, p in zip(axes.flat, range(n), pred):
    label = ds[i][1]
    ax.imshow(ds[i][0][0] * 0.3081 + 0.1307, cmap="gray")
    ax.set_title(f"Tahmin: {p.item()} (Gerçek: {label})", fontsize=9,
                 color="green" if p.item() == label else "red")
    ax.axis("off")
fig.suptitle("El Yazısı Rakam Tanıma - Örnek Tahminler")
fig.tight_layout()
out = ROOT / "models" / "tahminler.png"
fig.savefig(out, dpi=120)
print(f"Kaydedildi: {out}")
