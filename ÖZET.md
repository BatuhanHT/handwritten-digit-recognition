# ÖZET — El Yazısı Rakam Tanıma

## Proje Özeti
El yazısıyla yazılmış rakamları (0–9) görüntülerden otomatik olarak tanıyan bir yapay zeka modeli geliştirmek.

## Amaç
- 28×28 gri tonlamalı rakam görüntülerini 10 sınıfa (0–9) ayıran bir model eğitmek
- Doğruluk hedefi: test setinde ≥ %98 — **ulaşıldı: %98,94**

## Veri Seti
- **MNIST**: 60.000 eğitim, 10.000 test görüntüsü
- Görüntü boyutu: 28×28 piksel, gri tonlama
- Sınıflar dengeli: test setinde her rakamdan 892 ile 1135 arasında örnek var

## Yöntem
1. Veri yükleme ve ön işleme: pikseller 0–1 aralığına ölçeklenir, ardından eğitim setinin ortalaması (0,1307) ve standart sapması (0,3081) ile normalize edilir
2. Model: Evrişimli Sinir Ağı (CNN) — iki evrişim bloğu (3×3 filtre, ReLU, max pooling), dropout ve iki tam bağlantılı katman
3. Eğitim: Adam optimizasyonu, öğrenme oranı 0,001, batch 64, 5 epoch, çapraz entropi kaybı
4. Değerlendirme: doğruluk, sınıf bazında metrikler, karışıklık matrisi (confusion matrix), hatalı tahmin örnekleri
5. Canlı demo: kullanıcının çizdiği rakam MNIST formatına getirilir (ters çevirme, kırpma, 20 piksele ölçekleme, kütle merkezine göre ortalama) ve modelle tahmin edilir

Başlangıçta planlanan basit tam bağlantılı (MLP) temel model henüz eğitilmedi; CNN ile karşılaştırma sonraki adımlar arasında.

## Kullanılan Teknolojiler
- Python 3.10+
- PyTorch ve torchvision
- scikit-learn (metrikler)
- NumPy, Matplotlib, Pillow
- Electron (canlı demo arayüzü)

Deneyler Jupyter Notebook yerine `src/` altındaki betiklerle yapıldı.

## Proje Yapısı
```
├── src/
│   ├── train.py       # CNN modeli ve eğitim döngüsü
│   ├── evaluate.py    # Test metrikleri ve grafikler
│   ├── visualize.py   # Örnek tahminlerin görselleştirilmesi
│   └── predict.py     # Tek görüntüden tahmin; Electron demosunun arka ucu
├── app/               # Electron canlı demo arayüzü
├── tests/             # predict.py testleri
├── data/              # MNIST (ilk çalıştırmada otomatik iner, repoda yok)
├── models/            # Eğitilen model ve çıktılar
├── setup.sh           # Sanal ortam ve bağımlılık kurulumu
├── requirements.txt
├── README.md
└── ÖZET.md            # Bu dosya
```

## Yol Haritası
- [x] Ortam kurulumu
- [x] Veri yükleme ve ön işleme
- [ ] Temel (MLP) model
- [x] CNN modeli
- [ ] Hiperparametre ayarı (şimdilik varsayılan değerler kullanıldı)
- [x] Değerlendirme ve raporlama
- [x] Çizim arayüzü ile canlı tahmin

## Notlar
- **Sonuçlar:** 5 epoch sonunda test doğruluğu %98,94, makro F1 0,9894, 10.000 görüntüde 106 hata. En iyi sonuç 4. epoch'ta (%99,10). En sık hata: 22 kez gerçek 0'ın 9 olarak tahmin edilmesi.
- **Sınırlılıklar:** Ayrı doğrulama seti yok, eğitim test setiyle izlendi; son epoch'un modeli kaydediliyor; seed sabit olmadığı için sonuçlar her çalıştırmada biraz değişiyor.
- **Sonraki adımlar:** Doğrulama seti ile en iyi modeli kaydetmek, veri artırma (döndürme, kaydırma) ile özellikle 0 ile 9 arasındaki karışıklığı azaltmak, MLP temel modeliyle karşılaştırma.
