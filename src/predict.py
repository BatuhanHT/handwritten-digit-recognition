"""Tek bir çizimden rakam tahmini: MNIST tarzı ön işleme + CNN, her adım terminale yazılır.

Kullanım:
  python src/predict.py --image rakam.png   # tek görüntü; adımlar ve tahmin terminale yazılır
  python src/predict.py --serve             # Electron demosu: stdin'den JSON satırları okur

Ön işleme, sunumdaki 4 adımı izler:
  1. Çizim         280×280 tuval gri tonlamaya çevrilir
  2. Ters çevir    açık zemin koyulaştırılır (MNIST: siyah zemin, beyaz rakam), boşluk kırpılır
  3. 28×28'e getir uzun kenar 20 piksele ölçeklenir, kütle merkezine göre ortalanır, normalize edilir
  4. CNN + softmax her rakam için olasılık; en yükseği tahmindir
"""
import argparse
import base64
import io
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train import CNN, ROOT

MEAN, STD = 0.1307, 0.3081
MODEL_PATH = ROOT / "models" / "mnist_cnn.pt"
INK_THRESHOLD = 30  # 0–255; bunun üstündeki pikseller rakamın parçası sayılır
MIN_DIGIT_RATIO = 0.2  # rakamın uzun kenarı görüntünün en az bu oranı olmalı (280 tuvalde 56 piksel)


def load_model(path=MODEL_PATH):
    model = CNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def pct(p):
    """0.9921 → '%99,2' (Türkçe yazım). %100,0 gibi yanıltıcı yuvarlama yerine '%99,9+'."""
    if p >= 0.9995:
        return "%99,9+"
    return "%" + f"{p * 100:.1f}".replace(".", ",")


def signed(n):
    return f"+{n}" if n > 0 else ("−" + str(-n) if n < 0 else "0")


def to_gray(img):
    """Saydam alanları beyaz kabul ederek 0–255 gri tonlamalı float dizi döndürür."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg
    return np.asarray(img.convert("L"), dtype=np.float32)


def preprocess(img, invert=None):
    """Çizimi MNIST formatına getirir. Her adımın ara görüntüsünü ve bilgisini döndürür.

    invert=None zemini kenar piksellerinden tahmin eder; Electron tuvali hep beyaz olduğu için
    sunucu kipi invert=True verir.
    """
    gray = to_gray(img)
    h0, w0 = gray.shape

    if invert is None:
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        invert = bool(border.mean() > 127)
    inverted = bool(invert)
    ink = 255.0 - gray if inverted else gray

    mask = ink > INK_THRESHOLD
    if not mask.any():
        raise ValueError("Tuval boş: önce bir rakam çizin")
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    crop = ink[y0:y1 + 1, x0:x1 + 1]

    ch, cw = crop.shape
    # Kalın kalemle çizilen çok küçük rakam ya da tek tık, ölçeklenince şişip yanlış ama emin bir tahmin verir
    if max(ch, cw) < MIN_DIGIT_RATIO * max(h0, w0):
        raise ValueError("Rakam çok küçük, daha büyük çizin")
    scale = 20.0 / max(ch, cw)
    nh, nw = max(1, round(ch * scale)), max(1, round(cw * scale))
    small = np.asarray(
        Image.fromarray(crop.astype(np.uint8)).resize((nw, nh), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )

    total = small.sum()
    cy = float((small.sum(axis=1) * np.arange(nh)).sum() / total)
    cx = float((small.sum(axis=0) * np.arange(nw)).sum() / total)
    top = min(max(int(round(14 - cy)), 0), 28 - nh)
    left = min(max(int(round(14 - cx)), 0), 28 - nw)
    img28 = np.zeros((28, 28), dtype=np.float32)
    img28[top:top + nh, left:left + nw] = small

    x = torch.from_numpy(img28 / 255.0).reshape(1, 1, 28, 28)
    x = (x - MEAN) / STD
    return {
        "size": (w0, h0),
        "inverted": inverted,
        "bbox": (x0, y0, x1, y1),
        "crop": crop.astype(np.uint8),
        "scaled": (nw, nh),
        "shift": (left - (28 - nw) // 2, top - (28 - nh) // 2),
        "img28": np.clip(img28, 0, 255).astype(np.uint8),
        "tensor": x,
    }


def forward_steps(model, x):
    """CNN.forward ile aynı hesabı katman katman yapar; her ara çıktıyı döndürür."""
    with torch.no_grad():
        a = F.relu(model.conv1(x))
        p1 = F.max_pool2d(a, 2)
        p2 = F.max_pool2d(F.relu(model.conv2(p1)), 2)
        flat = model.dropout(p2.flatten(1))
        h = F.relu(model.fc1(flat))
        logits = model.fc2(h)
    steps = [
        ("conv1 3×3 + ReLU + max pool", p1),
        ("conv2 3×3 + ReLU + max pool", p2),
        ("düzleştir + dropout (yalnız eğitimde)", flat),
        ("fc1 + ReLU", h),
        ("fc2 (her rakam için bir skor)", logits),
    ]
    return steps, logits, a


def shape_text(t):
    return "×".join(str(d) for d in t.shape[1:])


def png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def feature_maps(conv1_out, k=8):
    """conv1 çıktısından en güçlü k haritayı 0–255'e ölçekleyip döndürür."""
    maps = conv1_out[0]
    idx = torch.argsort(maps.mean(dim=(1, 2)), descending=True)[:k]
    out = []
    for i in idx:
        m = maps[i].numpy()
        m = (m - m.min()) / (m.max() - m.min() + 1e-8) * 255
        out.append(m.astype(np.uint8))
    return out


def run(model, img, out=print, invert=None):
    """Bir görüntüyü sınıflandırır; adımları `out` ile satır satır yazar."""
    t0 = time.perf_counter()
    pre = preprocess(img, invert=invert)
    steps, logits, conv1_out = forward_steps(model, pre["tensor"])
    probs = F.softmax(logits, dim=1)[0]
    ms = (time.perf_counter() - t0) * 1000
    digit = int(probs.argmax())

    w0, h0 = pre["size"]
    x0, y0, x1, y1 = pre["bbox"]
    nw, nh = pre["scaled"]
    dx, dy = pre["shift"]
    out(f"[1/4] Çizim: {w0}×{h0} piksel görüntü alındı, gri tonlamaya çevrildi")
    if pre["inverted"]:
        out("[2/4] Ters çevir + kırp: renkler ters çevrildi (siyah zemin, beyaz rakam)")
    else:
        out("[2/4] Ters çevir + kırp: zemin zaten koyu, ters çevirmeye gerek yok")
    out(f"      rakamın kutusu x={x0}–{x1}, y={y0}–{y1} → {x1 - x0 + 1}×{y1 - y0 + 1} piksel kırpıldı")
    out(f"[3/4] 28×28'e getir: uzun kenar 20 piksele ölçeklendi ({nw}×{nh})")
    out(f"      kütle merkezine göre ortalandı (kaydırma dx={signed(dx)}, dy={signed(dy)})")
    out("      normalize edildi: (x − 0,1307) / 0,3081 → tensör 1×1×28×28")
    out("[4/4] CNN + softmax:")
    for name, t in steps:
        out(f"      {name:<38}→ {shape_text(t)}")
    out(f"      {'softmax':<38}→ 10 olasılık")
    out("Olasılıklar:")
    for d, p in enumerate(probs.tolist()):
        bar = "█" * round(p * 20)
        mark = "  ◀" if d == digit else ""
        out(f"  {d}  {bar:<20}  {pct(p):>6}{mark}")
    ms_text = f"{ms:.1f}".replace(".", ",")
    out(f"Tahmin: {digit}  (güven {pct(float(probs[digit]))}, süre {ms_text} ms)")

    return {
        "digit": digit,
        "probs": [round(p, 6) for p in probs.tolist()],
        "ms": round(ms, 2),
        "images": {
            "crop": png_b64(pre["crop"]),
            "img28": png_b64(pre["img28"]),
            "maps": [png_b64(m) for m in feature_maps(conv1_out)],
        },
    }


def emit(tag, payload):
    print(f"@@{tag} {json.dumps(payload, ensure_ascii=False)}", flush=True)


def serve(model_path):
    rel = model_path.relative_to(ROOT) if model_path.is_relative_to(ROOT) else model_path
    print(f"Model yükleniyor: {rel}", flush=True)
    if not model_path.exists():
        print(f"Hata: model dosyası yok ({rel}). Önce: python src/train.py --epochs 5", flush=True)
        emit("FATAL", {"message": f"Model dosyası bulunamadı: {rel}"})
        sys.exit(1)
    model = load_model(model_path)
    n_params = sum(p.numel() for p in model.parameters())
    forward_steps(model, torch.zeros(1, 1, 28, 28))  # ilk çağrının gecikmesini baştan öde
    n_text = f"{n_params:,}".replace(",", ".")
    print(f"Model hazır: CNN, {n_text} parametre · cihaz: CPU · PyTorch {torch.__version__}", flush=True)
    emit("READY", {"params": n_params, "torch": torch.__version__})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            img = Image.open(io.BytesIO(base64.b64decode(req["png"])))
            result = run(model, img, out=lambda s: print(s, flush=True), invert=True)  # tuval hep beyaz
            emit("RESULT", {"id": req_id, **result})
        except Exception as e:  # noqa: BLE001 — demo süreci her hatada ayakta kalmalı
            print(f"Hata: {e}", flush=True)
            emit("ERROR", {"id": req_id, "message": str(e)})


def main():
    p = argparse.ArgumentParser(description="Çizimden rakam tahmini")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="Tahmin edilecek görüntü dosyası (PNG/JPG)")
    g.add_argument("--serve", action="store_true", help="Electron demosu için stdin'den istek oku")
    p.add_argument("--model", default=str(MODEL_PATH), help="Model dosyası (varsayılan: models/mnist_cnn.pt)")
    args = p.parse_args()

    from pathlib import Path
    model_path = Path(args.model).resolve()
    if args.serve:
        serve(model_path)
    else:
        run(load_model(model_path), Image.open(args.image))


if __name__ == "__main__":
    main()
