# Eski sistem: Gate2 (kapılı) ve Equal çıkarım, üç seed (CPU, salt okunur)

Betik: `recompute_equal_vs_gated_3seeds.py`. Girdi: `D:\m1\Documents\seed 42|52|62\best_model.pth` (strict yükleme, tüm anahtarlar eşleşti), APTOS `cache_224.npz` + dondurulmuş manifestin 367 test görüntüsü. GPU, eğitim ve dosya değişikliği yok. Çıktılar: `recomputed_equal_vs_gated.json`, `recomputed_logits_seed*.npz`.

Doğrulama: Gated çıktı, sunucunun kaydettiği `test_metrics.json` değerlerini birebir yeniden üretti (QWK 0.917439 / 0.919075 / 0.919232; accuracy aynı). Yani model, ön işleme ve test kümesi doğru kuruldu.

## Aynı ağırlıklar, farklı çıkarım yolları (test, 367 görüntü)

| Çıktı | QWK seed 42 / 52 / 62 | QWK ortalama ± SD | Accuracy % | Macro-F1 |
|---|---|---|---|---|
| **Gate2 (kapılı residual)** `z_T + 0.70·g₂·(z_S − ort.)` | 0.9174 / 0.9191 / 0.9192 | **0.9186 ± 0.0010** | 85.47 ± 0.31 | 0.7034 ± 0.0094 |
| **Equal** `(z_T + z_S)/2` | 0.9174 / 0.9191 / 0.9157 | **0.9174 ± 0.0017** | 85.29 ± 0.54 | 0.7037 ± 0.0090 |
| Makale formülü `((z₁+z₂)/2 + z_S)/2` (ham öğretmen logitleri) | 0.9122 / 0.9051 / 0.9078 | 0.9083 ± 0.0036 | 84.11 ± 0.42 | 0.6872 ± 0.0093 |
| Yalnızca kapılı öğretmen `z_T` (checkpoint içindeki) | 0.9158 / 0.9122 / 0.9141 | 0.9141 ± 0.0018 | 85.10 ± 0.31 | 0.6974 ± 0.0089 |
| Yalnızca T2 (checkpoint içindeki) | 0.9158 / 0.9155 / 0.9141 | 0.9151 ± 0.0009 | 85.20 ± 0.42 | 0.6979 ± 0.0095 |
| Yalnızca T1 | 0.6589 / 0.6443 / 0.6612 | 0.6548 ± 0.0092 | 67.12 ± 0.42 | 0.4637 ± 0.0031 |
| Öğrenci dalı tek başına | −0.0853 / 0.0000 / 0.0000 | −0.0284 ± 0.0493 | 5.27 ± 0.16 | 0.0231 ± 0.0059 |

SD örnek standart sapmasıdır (n−1), üç seed üzerinden.

## Bulgular

1. Equal, üç seed ortalamasında 0.9174 (Gate2'ye göre −0.0012). Seed 42 ve 52'de etiketler aynı; seed 62'de 7 görüntüde (%1.9) farklı. Önceki denetimde yazdığım 0.9174 ortalaması doğrulandı.
2. Makaledeki 0.9186, Equal değil Gate2 sonucudur. Makalenin Equal formülü (ham iki öğretmen ortalaması) 0.9083 verir.
3. Gate2 ve Equal farkı (0.0012) seed SD'leri (0.0010–0.0017) içinde; üstünlük iddia edilemez. Eşit ağırlığa geçiş performans değil sadelik gerekçesiyle savunulabilir.
4. Öğrenci dalı tek başına 1–2 sınıf tahmin ediyor. Gate2 ortalama değeri 0.077–0.090, çatışma freni görüntülerin %88'inde devrede. Öğrencinin çıktıya katkısı bu nedenle görüntüye özgü sınıflandırma bilgisi olarak yorumlanmamalı; katkının ne olduğu bu analizde test edilmedi.
5. Kapılı öğretmen (0.9141) ve tam sistem (0.9186) arasındaki fark +0.0045. Bu fark için eşlenik güven aralığı hesaplanmadı.

## Dikkat

- Öğretmen satırları, öğrenci checkpoint'inin içindeki öğretmenlere aittir. Bu aşamada BN istatistikleri ve spectral-norm vektörleri güncellendiği için, daha önce raporlanan bağımsız öğretmen QWK'sinden (T2'li kapılı ensemble 0.907, T1 0.640) farklıdır. İkisini aynı tabloda karıştırmayın.
- Bu betiğin ROC-AUC değeri sunucudaki değerle aynı tanıma sahip değil (sunucu 0.9583, burada one-vs-rest macro 0.9276); makalede sunucu değerini kullanın veya tanımı belirtin.
- Seed başına SD yalnızca öğrenci eğitimi seed'ini yansıtır; öğretmenler üç seed'de de aynıdır ve öğretmen eğitiminde seed ayarlanmamıştır.
