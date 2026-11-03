El yazısı rakam tanıma canlı demo uygulaması. Bir rakam çizin; uygulama rakamın nasıl belirlendiğini (ters çevirme ve kırpma, 28×28'e getirme, CNN katmanları, softmax olasılıkları) adım adım gösterir ve tahmini verir. Python ve PyTorch paketin içinde gelir; ayrıca Python kurmanız ya da internete bağlanmanız gerekmez.

Paketteki model: 5 epoch eğitilmiş CNN, MNIST test setinde %99,10 doğruluk.

## İndirme

| Sistem | Dosya |
|---|---|
| macOS 14 (Sonoma) ve üstü, Apple Silicon (M1, M2, M3, M4…) | `RakamTanima-<sürüm>-mac-arm64.dmg` |
| Windows 10/11 (64 bit) | `RakamTanima-<sürüm>-win-x64.exe` |

## İlk açılış

Paketler imzasızdır (ders projesi; Apple veya Microsoft geliştirici sertifikası yok). Bu yüzden ilk açılışta sistem uyarı verir:

- **macOS:** `.dmg`'yi açın, uygulamayı Uygulamalar klasörüne sürükleyin. İlk açılışta uyarı çıkarsa **Sistem Ayarları → Gizlilik ve Güvenlik** bölümünde "Rakam Tanima Demo" için **Yine de Aç**'a basın. "Hasarlı" uyarısı çıkarsa ya da uygulama açılıp sağ üstte "Model yükleniyor…" yazısında takılı kalırsa Terminal'de şunu çalıştırıp uygulamayı yeniden açın:
  `xattr -dr com.apple.quarantine "/Applications/Rakam Tanima Demo.app"`
- **Windows:** SmartScreen uyarısında **Ek bilgi → Yine de çalıştır**'a basın, sonra kurulumu tamamlayın.

## Kullanım

Rakamı fareyle ya da touchpad ile tuvalin büyük kısmını kullanarak çizin. **Enter** tahmin eder, **Esc** temizler. Uygulama açıldıktan sonra modelin yüklenmesi birkaç saniye sürebilir; sağ üstte "Model hazır" yazınca başlayabilirsiniz.

Kaynak kod, model ve sonuçlar: https://github.com/BatuhanHT/handwritten-digit-recognition
