"""Test setinde değerlendirme: metrikler, karışıklık matrisi, eğitim eğrileri, hatalı tahminler."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms

from train import CNN, ROOT, get_device

OUT = ROOT / "models" / "degerlendirme"
OUT.mkdir(parents=True, exist_ok=True)

device = get_device()
model = CNN().to(device)
model.load_state_dict(torch.load(ROOT / "models" / "mnist_cnn.pt", map_location=device))
model.eval()

tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
ds = datasets.MNIST(ROOT / "data", train=False, download=True, transform=tfm)

preds = []
with torch.no_grad():
    for i in range(0, len(ds), 1000):
        x = torch.stack([ds[j][0] for j in range(i, i + 1000)]).to(device)
        preds.append(model(x).argmax(1).cpu())
pred = torch.cat(preds).numpy()
true = ds.targets.numpy()

rep = classification_report(true, pred, digits=4, output_dict=True)
cm = confusion_matrix(true, pred)
(OUT / "metrikler.json").write_text(json.dumps({"report": rep, "confusion_matrix": cm.tolist()}, indent=2))
print(classification_report(true, pred, digits=4))

# Karışıklık matrisi
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm, cmap="Blues")
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek"); ax.set_title("Karışıklık Matrisi (10.000 test görüntüsü)")
fig.tight_layout(); fig.savefig(OUT / "karisiklik_matrisi.png", dpi=150); plt.close(fig)

# Eğitim eğrileri
h = json.loads((ROOT / "models" / "history.json").read_text())
ep = [r["epoch"] for r in h]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(ep, [r["loss"] for r in h], "o-"); a1.set_title("Eğitim Kaybı"); a1.set_xlabel("Epoch"); a1.grid(alpha=.3)
a2.plot(ep, [r["test_acc"] * 100 for r in h], "o-", color="tab:green"); a2.set_title("Test Doğruluğu (%)")
a2.set_xlabel("Epoch"); a2.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "egitim_egrileri.png", dpi=150); plt.close(fig)

# Hatalı tahminler
wrong = np.where(pred != true)[0]
print(f"Hatalı: {len(wrong)}")
fig, axes = plt.subplots(3, 6, figsize=(10, 5.5))
for ax, k in zip(axes.flat, wrong[:18]):
    ax.imshow(ds[k][0][0] * 0.3081 + 0.1307, cmap="gray")
    ax.set_title(f"Gerçek {true[k]} / Tahmin {pred[k]}", fontsize=9, color="red"); ax.axis("off")
fig.suptitle(f"Hatalı Tahminlerden Örnekler (toplam {len(wrong)})")
fig.tight_layout(); fig.savefig(OUT / "hatali_tahminler.png", dpi=150); plt.close(fig)

# En sık karışan çiftler
off = [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j]
print("En sık karışanlar (adet, gerçek, tahmin):", sorted(off, reverse=True)[:5])
