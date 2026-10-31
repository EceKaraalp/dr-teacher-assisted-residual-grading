# Metriklerin yeniden hesaplanması (CPU)

## Kullanılan girdiler

- **Eski deney:** `results\archived_final_verification_2026-10-16\seed_{42,52,62}_probabilities.npz` (`ids`, `labels` ve her yol için 367×5 olasılık). Bu olasılıklar, SHA-256 değerleri doğrulanmış eski checkpoint'lerin CPU üzerinde çalıştırılmasıyla üretilmiştir.
- **Yeni deney:** `student_diagnostic_20261023_175134_909746.zip` → `seed{42,52,62}_test_diagnostic.npz` (`ids`, `labels` ve 367×5 logit). Bunlar, hash'i doğrulanmış yeni checkpoint'lerin yalnızca çıkarımla elde edilen çıktılarıdır.
- **Kod:** `sklearn.metrics.cohen_kappa_score(y, pred, labels=range(5), weights='quadratic')` ve `accuracy_score`. `pred` değeri olasılık ya da logit üzerinden `argmax` ile bulunur. Softmax sıralamayı değiştirmediği için olasılık ve logit aynı sınıfı verir.
- Yeni logitler, eski kimlik sırasına göre kimlikle hizalandı (sıralar zaten aynıydı). Hizalanan etiketlerin eşitliği her seed için `assert` ile kontrol edildi.
- Çıktı dosyası: `recalculation_results.json`.

## Kontroller

| Kontrol | Sonuç |
|---|---|
| Gerçek etiketler | İki deneyde de `train.csv` ile aynı, sınıf dağılımı 181/37/100/19/30 |
| Sınıf sırası | 0..4 (`labels=range(5)` açıkça verildi) |
| QWK parametreleri | `weights='quadratic'` |
| Test örneği sayısı | Her seed için 367, eksik veya yinelenen yok |
| Olasılık ile sınıf karışıklığı | Yok: QWK sınıf etiketleriyle hesaplanıyor; AUC, NLL ve ECE olasılıklarla hesaplanıyor |
| Yeniden hesap = raporlanan değer (eski) | Üç seed için confusion matrix ve QWK birebir aynı |
| Yeniden hesap = raporlanan değer (yeni) | primary, equal, teacher ve student yollarının dördünde, üç seed için confusion matrix ve QWK birebir aynı (`new_reported_vs_diag`) |
| QWK'nin confusion matrix'ten bağımsız hesabı (yeni) | `per_seed.json` matrislerinden elle yazılmış QWK formülüyle hesaplanan değerler raporlanan değerlere eşit |

## Seed bazlı sonuçlar

| Yol | Seed 42 QWK / Acc | Seed 52 QWK / Acc | Seed 62 QWK / Acc | QWK ort. ± SD (ddof=1) | Acc ort. ± SD (ddof=1) |
|---|---|---|---|---|---|
| **Eski kapılı residual (makale)** | 0.9174 / 0.8529 | 0.9191 / 0.8583 | 0.9192 / 0.8529 | **0.9186 ± 0.0010** | **0.8547 ± 0.0031** |
| Eski equal `(z_t+z_s)/2` | 0.9174 / 0.8529 | 0.9191 / 0.8583 | 0.9157 / 0.8474 | 0.9174 ± 0.0017 | 0.8529 ± 0.0054 |
| Eski öğretmen yolu | 0.9158 / 0.8474 | 0.9122 / 0.8529 | 0.9141 / 0.8529 | 0.9141 ± 0.0018 | 0.8510 ± 0.0031 |
| Eski öğrenci dalı tek başına | −0.0853 / 0.0545 | 0.0000 / 0.0518 | 0.0000 / 0.0518 | −0.0284 ± 0.0493 | 0.0527 ± 0.0016 |
| Yeni kapılı residual (primary) | 0.9067 / 0.8529 | 0.9009 / 0.8392 | 0.8914 / 0.8392 | 0.8996 ± 0.0077 | 0.8438 ± 0.0079 |
| **Yeni equal (revizyon)** | 0.9099 / 0.8556 | 0.9001 / 0.8365 | 0.8974 / 0.8529 | **0.9025 ± 0.0066** | **0.8483 ± 0.0103** |
| Yeni öğretmen yolu | 0.8909 / 0.8447 | 0.9009 / 0.8392 | 0.8915 / 0.8392 | 0.8944 ± 0.0056 | 0.8411 ± 0.0031 |
| Yeni öğrenci dalı tek başına | 0.7901 / 0.6185 | 0.0000 / 0.0518 | 0.0000 / 0.1008 | 0.2634 ± 0.4561 | 0.2570 ± 0.3140 |

Popülasyon SD'si kullanılsaydı makaledeki değerler QWK için ±0.0008, accuracy için ±0.26 olurdu. Makaledeki ±0.0010 ve ±0.31 değerleri yalnızca örneklem SD'siyle tutarlıdır. Yeni sonuçlar da örneklem SD'si kullanır. **SD tanımları aynıdır.**

## Görüntü düzeyinde farklar (eski kapılı yol ile yeni equal yol)

| Seed | Farklı tahmin edilen görüntü | Yalnızca eskinin doğru bildiği | Yalnızca yeninin doğru bildiği |
|---|---|---|---|
| 42 | 39 | 15 | 16 |
| 52 | 41 | 21 | 13 |
| 62 | 41 | 17 | 17 |

## Eşleştirilmiş görüntü bootstrap'ı

Bu analiz, farkın test örneklemesine göre büyüklüğünü değerlendirmek için yapıldı. Ayrıntılar: `old_vs_new_paired_bootstrap.json`, 10,000 örnekleme, RNG 20261024; aynı indeks dizisi üç seed'e ve iki sisteme birlikte uygulandı.

| Karşılaştırma (eski − yeni, 3 seed ortalama QWK) | Nokta | %95 aralık |
|---|---|---|
| Eski kapılı − yeni equal | +0.0161 | [−0.0117, +0.0451] |
| Eski kapılı − yeni kapılı | +0.0189 | [−0.0087, +0.0480] |
| Eski equal − yeni equal | +0.0149 | [−0.0126, +0.0439] |
| Eski öğretmen − yeni öğretmen | +0.0196 | [−0.0107, +0.0503] |

Bu aralıklar yalnızca test görüntüsü örneklemesini yansıtır. Eğitim tekrarından kaynaklanan belirsizliği veya hasta kümelenmesini kapsamaz. Tüm aralıklar sıfırı içerir. Bu nedenle fark, bu test kümesinde istatistiksel olarak sağlam biçimde ayrışmış **değildir**.

## Sonuç

Her iki sayı da görüntü bazlı tahminlerden birebir yeniden üretilmiştir. Metrik kodu, sınıf sırası, etiketler, örnek sayısı ve SD türü aynıdır. Fark, metrik hesabından kaynaklanmamaktadır.
