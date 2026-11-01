# ÖZET — El Yazısı Rakam Tanıma

## Proje Özeti
El yazısıyla yazılmış rakamları (0–9) görüntülerden otomatik olarak tanıyan bir yapay zeka modeli geliştirmek.

## Amaç
- 28x28 gri tonlamalı rakam görüntülerini 10 sınıfa (0–9) ayıran bir model eğitmek
- Doğruluk hedefi: test setinde ≥ %98

## Veri Seti
- **MNIST**: 60.000 eğitim, 10.000 test görüntüsü
- Görüntü boyutu: 28x28 piksel, gri tonlama

## Yöntem
1. Veri yükleme ve ön işleme (normalizasyon: 0–1 aralığı)
2. Temel model: basit tam bağlantılı (MLP) ağ
3. Geliştirilmiş model: Evrişimli Sinir Ağı (CNN)
4. Eğitim, doğrulama ve test
5. Değerlendirme: doğruluk, karışıklık matrisi (confusion matrix)

## Önerilen Teknolojiler
- Python 3
- TensorFlow/Keras veya PyTorch
- NumPy, Matplotlib
- Jupyter Notebook

## Proje Yapısı (öneri)
```
ÖZET/
├── ÖZET.md          # Bu dosya
├── data/            # Veri seti
├── notebooks/       # Deneyler
├── src/             # Eğitim ve tahmin kodları
└── models/          # Kaydedilen modeller
```

## Yol Haritası
- [ ] Ortam kurulumu
- [ ] Veri keşfi ve ön işleme
- [ ] Temel (MLP) model
- [ ] CNN modeli
- [ ] Hiperparametre ayarı
- [ ] Değerlendirme ve raporlama
- [ ] (Opsiyonel) Çizim arayüzü ile canlı tahmin

## Notlar
_Proje ilerledikçe buraya eklenecek._
