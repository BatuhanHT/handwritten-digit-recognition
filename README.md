# El Yazısı Rakam Tanıma

MNIST veri seti üzerinde PyTorch ile eğitilmiş bir evrişimli sinir ağı (CNN). 28×28 gri tonlamalı el yazısı rakam görüntülerini 0–9 arasındaki 10 sınıftan birine ayırır. Electron ile yazılmış canlı demo uygulamasında çizdiğiniz rakamı tanır ve rakamın nasıl belirlendiğini adım adım gösterir.

EBLG463 Yapay Zeka ve Makine Öğrenimi dersi grup projesi (Dr. Savaş Ünsal). Ekip: Aleyna Zengin, Batuhan Halit Temel, Muhammet Emre Öz, Deniz Ekiz.

## Sonuçlar

5 epoch eğitimin ardından 10.000 görüntülük test setinde:

| Metrik | Değer |
|---|---|
| Test doğruluğu | **%98,94** (hedef ≥ %98) |
| Makro F1 skoru | 0,9894 |
| Hatalı tahmin | 106 |
| En sık hata | 0 → 9 (gerçek 0, tahmin 9; 22 kez) |

| Epoch | Eğitim kaybı | Test doğruluğu |
|---|---|---|
| 1 | 0,1487 | %98,35 |
| 2 | 0,0508 | %98,97 |
| 3 | 0,0365 | %98,98 |
| 4 | 0,0288 | %99,10 (en iyi) |
| 5 | 0,0228 | %98,94 (kaydedilen model) |

Tablo, sunumdaki eğitim koşusuna aittir. Repodaki `models/mnist_cnn.pt` aynı ayarlarla yapılmış başka bir koşudan gelir: 5. epoch'ta test doğruluğu %99,10. Seed sabit olmadığı için koşular arasında küçük farklar olur.

## Kurulum

Python 3.10 veya üstü gerekir. macOS'taki sistem Python'u 3.9 ise önce `brew install python@3.12` ile yeni bir sürüm kurun; `setup.sh` uygun sürümü kendisi bulur.

```bash
git clone https://github.com/BatuhanHT/handwritten-digit-recognition.git
cd handwritten-digit-recognition
./setup.sh
source .venv/bin/activate
```

## Kullanım

```bash
python src/train.py --epochs 5          # Modeli eğitir → models/mnist_cnn.pt, models/history.json
python src/evaluate.py                  # Metrikler ve grafikler → models/degerlendirme/
python src/visualize.py                 # İlk 20 test görüntüsünde tahmin → models/tahminler.png
python src/predict.py --image rakam.png # Tek bir görüntüden tahmin, adımlar terminale yazılır
```

`train.py` parametreleri:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--epochs` | 5 | Epoch sayısı |
| `--batch-size` | 64 | Batch boyutu |
| `--lr` | 0.001 | Öğrenme oranı |

İlk çalıştırmada MNIST `data/` klasörüne otomatik iner. Cihaz otomatik seçilir (CUDA, Apple MPS veya CPU). `evaluate.py`, `visualize.py` ve `predict.py` kayıtlı modeli kullanır; önce `train.py` çalıştırılmalıdır.

`evaluate.py` şu dosyaları üretir: `metrikler.json` (sınıf bazında metrikler ve karışıklık matrisi), `karisiklik_matrisi.png`, `egitim_egrileri.png`, `hatali_tahminler.png`.

## Canlı Demo (Electron)

Çizdiğiniz rakamı eğitilmiş modelle tahmin eden masaüstü uygulaması. Arka planda `src/predict.py` çalışır. Terminal panelinde her adım için önce o adımı yapan **gerçek kod satırları** (dosyadaki satır numaralarıyla), hemen altında Python'un ürettiği çıktı görünür. Kod hücreleri, uygulama her açıldığında (ya da Cmd+R ile yenilendiğinde) `src/predict.py` içindeki `# [adım N]` … `# [/adım N]` işaretlerinden okunur; paketlenmiş sürümde paketin içindeki kopyadan okunur. Rakam şu 4 adımda belirlenir:

1. **Çizim:** 280×280 tuval gri tonlamaya çevrilir.
2. **Ters çevir + kırp:** MNIST'teki gibi siyah zemin ve beyaz rakam elde edilir, rakamın çevresindeki boşluk kırpılır.
3. **28×28'e getir:** Uzun kenar 20 piksele ölçeklenir, rakam kütle merkezine göre 28×28 alanın ortasına yerleştirilir, normalize edilir.
4. **CNN + softmax:** Görüntü katmanlardan geçer (32×14×14 → 64×7×7 → 3136 → 128 → 10). Her rakam için bir olasılık hesaplanır, en yükseği tahmindir.

Node.js 18+ ve eğitilmiş model (`models/mnist_cnn.pt`) gerekir. İnternet bağlantısı gerekmez.

```bash
cd app
npm install
npm start
```

Kısayollar: **Enter** tahmin eder, **Esc** tuvali temizler, **K** kod hücrelerini açar ya da kapatır (bir hücrenin başlığına tıklamak yalnızca onu açar). Uygulama `.venv/bin/python`'u kullanır; yoksa `python3` denenir. Başka bir yorumlayıcı için `PYTHON=/yol/python npm start` yazın.

### Hazır paketler (dmg / exe)

Python kurmadan denemek için [Releases](https://github.com/BatuhanHT/handwritten-digit-recognition/releases) sayfasından macOS 14+ (Apple Silicon) `.dmg` ya da Windows (x64) `.exe` dosyasını indirin. Python, PyTorch ve model paketin içinde gelir. Paketler imzasızdır; ilk açılışta çıkan uyarının nasıl geçileceği release notlarında anlatılıyor.

Paketler `v*` etiketi atılınca GitHub Actions ile derlenir (`.github/workflows/release.yml`). Yerelde derlemek için (yalnızca kendi işletim sisteminiz için):

```bash
pip install -r requirements-build.txt   # PyInstaller
scripts/build_backend.sh                # src/predict.py → build/backend/predict-backend/
cd app && npm run dist                  # → app/dist/*.dmg ya da *.exe
```

## Testler

```bash
python -m unittest discover -s tests -v   # Python: ön işleme, ileri geçiş, --serve protokolü
cd app && npm test                        # Arayüz: predict.py'deki kod hücresi işaretleri
```

Python testleri ön işlemeyi (ters çevirme, kırpma, 20 piksel ölçekleme, kütle merkezine göre ortalama) ve adım adım ileri geçişin `CNN.forward` ile aynı sonucu verdiğini doğrular. Model dosyası ya da MNIST indirmesi gerektirmez.

## Model

| Katman | Çıktı boyutu |
|---|---|
| Girdi | 1×28×28 |
| Conv2d 3×3 + ReLU + MaxPool | 32×14×14 |
| Conv2d 3×3 + ReLU + MaxPool | 64×7×7 |
| Düzleştirme + Dropout 0,25 | 3136 |
| Linear + ReLU | 128 |
| Linear | 10 |

Toplam 421.642 eğitilebilir parametre. Optimizasyon: Adam (öğrenme oranı 0,001). Kayıp fonksiyonu: çapraz entropi. Batch boyutu: 64.

Ön işleme: piksel değerleri 0–1 aralığına ölçeklenir, ardından eğitim setinin ortalaması (0,1307) ve standart sapması (0,3081) ile normalize edilir.

## Proje Yapısı

```
├── src/
│   ├── train.py       # CNN modeli ve eğitim döngüsü
│   ├── evaluate.py    # Test metrikleri, karışıklık matrisi, eğitim eğrileri, hatalı tahminler
│   ├── visualize.py   # Örnek tahminlerin görselleştirilmesi
│   └── predict.py     # Tek görüntüden tahmin; Electron demosunun arka ucu
├── app/               # Electron canlı demo arayüzü
├── tests/             # predict.py testleri
├── scripts/           # build_backend.sh: Python arka ucunu paketler (PyInstaller)
├── .github/workflows/ # release.yml: dmg ve exe derleyip GitHub Release'e yükler
├── data/              # MNIST (ilk çalıştırmada otomatik iner, repoda yok)
├── models/            # Eğitilen model ve çıktılar (tahminler.png örnek çıktı)
├── setup.sh           # Sanal ortam ve bağımlılık kurulumu
├── requirements.txt
├── README.md
└── ÖZET.md            # Proje önerisi ve yol haritası
```

## Sınırlılıklar

- Ayrı bir doğrulama seti yok; eğitim, test seti üzerinden izlendi.
- En iyi epoch'un değil, son epoch'un modeli kaydediliyor.
- Seed sabit olmadığı için sonuçlar çalıştırmadan çalıştırmaya biraz değişiyor.

## Yol Haritası

- [x] Ortam kurulumu
- [x] CNN modeli ve eğitim betiği
- [x] Karışıklık matrisi ile değerlendirme
- [x] Örnek tahminlerin görselleştirilmesi
- [x] Tek görüntüden tahmin betiği
- [x] Çizim arayüzü (Electron) ile canlı tahmin
- [ ] Doğrulama seti ve en iyi modeli kaydetme
- [ ] Veri artırma (döndürme, kaydırma)
- [ ] MLP temel modeliyle karşılaştırma

## Kaynaklar

- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning applied to document recognition," *Proc. IEEE*, 86(11), 1998. (MNIST)
- A. Paszke vd., "PyTorch: An imperative style, high-performance deep learning library," *NeurIPS*, 2019.
- D. P. Kingma, J. Ba, "Adam: A method for stochastic optimization," *ICLR*, 2015.
- torchvision, scikit-learn, NumPy, Matplotlib ve Electron açık kaynak kütüphaneleri.

Proje önerisi ve ayrıntılar için [ÖZET.md](ÖZET.md) dosyasına bakın.
