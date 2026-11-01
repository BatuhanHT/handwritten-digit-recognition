# El Yazısı Rakam Tanıma

MNIST veri seti üzerinde PyTorch ile eğitilmiş bir evrişimli sinir ağı (CNN). El yazısıyla yazılmış rakamları (0–9) tanır.

**Sonuç:** 2 epoch eğitimle test setinde **%98,67** doğruluk.

## Kurulum

Python 3.10+ gerekir.

```bash
git clone https://github.com/BatuhanHT/handwritten-digit-recognition.git
cd handwritten-digit-recognition
./setup.sh
source .venv/bin/activate
```

## Eğitim

```bash
python src/train.py --epochs 5
```

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--epochs` | 5 | Epoch sayısı |
| `--batch-size` | 64 | Batch boyutu |
| `--lr` | 0.001 | Öğrenme oranı |

İlk çalıştırmada MNIST `data/` klasörüne otomatik iner. Eğitilen model `models/mnist_cnn.pt` olarak kaydedilir. Cihaz otomatik seçilir (CUDA, Apple MPS veya CPU).

## Model

| Katman | Detay |
|---|---|
| Conv2d + ReLU + MaxPool | 1 → 32 kanal, 3x3 |
| Conv2d + ReLU + MaxPool | 32 → 64 kanal, 3x3 |
| Dropout | 0.25 |
| Linear + ReLU | 3136 → 128 |
| Linear | 128 → 10 |

## Proje Yapısı

```
├── src/train.py       # Model ve eğitim kodu
├── setup.sh           # Sanal ortam ve bağımlılık kurulumu
├── requirements.txt
└── ÖZET.md            # Proje özeti ve yol haritası
```

## Yol Haritası

- [x] Ortam kurulumu
- [x] CNN modeli ve eğitim betiği
- [ ] Karışıklık matrisi ile değerlendirme
- [ ] Tek görüntü üzerinde tahmin betiği
- [ ] Çizim arayüzü ile canlı tahmin

Ayrıntılar için [ÖZET.md](ÖZET.md) dosyasına bakın.
