# 0.9186 QWK sonucunun kaynağı

## Özet

Orijinal makalede raporlanan QWK 0.9186 ± 0.0010 ve accuracy %85.47 ± %0.31 değerleri, eski sunucu deneyindeki `TS_ConvNeXtTiny_Residual` modelinin final aşamasındaki üç seed'inin (42, 52, 62) ortalamasıdır. Bu sonuç modelin **kapılı residual çıktı yolundan** elde edilmiştir. Yerel checkpoint kopyaları CPU üzerinde yeniden çalıştırıldığında üç confusion matrix'in üçü de birebir aynı çıkmıştır. Sonucun hangi eğitim koşusuna ve hangi ağırlıklara ait olduğu bu nedenle doğrulanmıştır.

## Kanıtlar

| Bilgi | Değer | Kanıt |
|---|---|---|
| Makaledeki değer | QWK 0.9186 ± 0.0010; Acc %85.47 ± %0.31 | Orijinal DOCX (SHA-256 `6261ede6…`), Abstract/Results |
| Kayıt dosyası | `lowdata_seed_results.csv`, Stage=`final`, Model=`TS_ConvNeXtTiny_Residual` | `D:\m1\Documents\lowdata_seed_results.csv` (SHA-256 `fcbcac71…`); `results\lowdata_hybrid_distill\` |
| Seed bazlı QWK | 42: 0.917439 · 52: 0.919075 · 62: 0.919232 | CSV ve her seed'in `test_metrics.json` dosyası |
| Seed bazlı accuracy | 0.852861 · 0.858311 · 0.852861 | Aynı dosyalar |
| Ortalama ± SD | QWK 0.918582 ± 0.000993; Acc 0.854678 ± 0.003146 | Bu denetimde yeniden hesaplandı (`recalculation_results.json` → `old_csv_sd`) |
| SD türü | **Örneklem standart sapması (ddof=1)**. Popülasyon SD'si kullanılsaydı 0.00081 ve 0.00257 olurdu; makaledeki değerler bunlarla değil ddof=1 ile eşleşiyor. | Notebook'taki `summarize()` fonksiyonu pandas `agg(['mean','std'])` kullanıyor; pandas'ta varsayılan ddof=1 |
| Üç seed ortalaması mı? | Evet | CSV'de üç `final` satırı var |
| Deney adı | `results/lowdata_hybrid_distill`, "Screening + Top-2 Final (3 Seeds)" | `train_lowdata_hybrid_distill.ipynb` |
| Deney tarihi | Doğrulanamadı. Yerel dosya tarihleri (14–16.10.2026) yalnızca kopyalama zamanını gösteriyor. | Dosya sistemi |
| Mimari | `ResidualTeacherStudentModel`: dondurulmuş `HeavyTeacherEnsemble` (T1 = `Baseline_HybridCNNViT_Gate_Improved` ve T2 = `Improved_Anchored_SpectralNorm`, olasılık kapısıyla birleştiriliyor) ve ConvNeXt Tiny öğrenci. Ayarlar: `residual_scale=0.70`, `use_conflict_brake=True`, `conflict_threshold=0.82`, `conflict_gmax=0.66`. Parametreler: toplam 141,999,859; eğitilebilir 27,824,454. | Notebook `build_ts_convnext_residual`; `test_metrics.json` (`param_total`); 16 Ekim doğrulamasında `strict=True` yükleme ve parametre sayısı eşleşmesi |
| Checkpoint'ler | `D:\m1\Documents\seed 42\best_model.pth` `d3327f355a51ac810431094f2e79505f1d43e31f13988ea1abd78caa65bd4eee`; `seed 52` `c3a056f1d1557121d83fed26c5527eae86feab10fd4415b79d3eb03252aada43`; `seed 62` `8027a8aa6a39e298f9189ed4c2f9c591f95b24abb46524adfd2476e5ced8e2b7` (her biri 568,566,788 bayt) | 24 Ekim'de yeniden hesaplanan hash'ler 16 Ekim `provenance.json` kayıtlarıyla aynı. Sunucudaki orijinal dosyaların hash'i bilinmiyor. |
| Checkpoint'in rolü | Doğrulama QWK'sine göre seçilmiş en iyi epoch (`best_model.pth`) | `fit()` doğrulama QWK'si iyileştiğinde `best_path` dosyasını kaydediyor; test öncesinde bu dosya yükleniyor |
| En iyi epoch / toplam epoch | 10/20, 4/14, 11/21 | Seed başına `training_history.json` → `val_qwk` dizisi |
| En iyi doğrulama QWK'si | 0.8976 · 0.8859 · 0.8889 (ortalama 0.8908) | Aynı dosyalar ve CSV `Best_Val_QWK` sütunu |
| Veri bölünmesi | Stratified 80/10/10, `train_test_split(..., random_state=42, stratify=...)` iki kez; train/val/test = 2928/367/367. Bölünme **eğitim seed'inden bağımsız** olarak sabit. | Depodaki `dataset_loader.py` dosyası. Sunucudaki kopyanın hash'i doğrulanamadı. |
| Test kimlikleri | 367 kimlik, manifestle aynı küme ve aynı sıra | 16 Ekim'de yeniden oluşturulup manifestle eşleştirildi; bu denetimde tekrar kontrol edildi (`data_split_comparison.md`). Eğitim sırasında yazılmış bir kimlik logu yok. |
| Sınıf sırası | 0–4 (No DR, Mild, Moderate, Severe, Proliferative) | `test_metrics.json` sınıflandırma raporu ve `labels=range(5)` |
| Füzyon (raporlanan yol) | Öğretmen tarafında `p_t=(1-g_t)p1+g_t p2` ve `z_t=log p_t`. Son çıktı `z = z_t + 0.70·g·(z_s − mean z_s)`, burada g öğrenilmiş ve örneğe bağlı bir kapı (çatışma freni de içeriyor). | Notebook satır 284–418; `fusion_method_comparison.md` |
| Kayıplar | Kapılı çıktı üzerinde ağırlıklı CE; non-inferiority (λ 0.8→0.25, margin 0.010→0.003); distillation λ=0.30; ordinal λ=0.12; kapı entropisi terimi | Notebook ConvNeXt konfigürasyonu (satır 863–869) ve `train_epoch` |
| Değerlendirme | `evaluate_test`; argmax sınıfı; `cohen_kappa_score(..., weights='quadratic')` | Notebook; 16 Ekim yeniden üretimi |
| Model ve mimari seçimi | Aday mimariler seed 42 ile **test QWK ortalamasına göre** sıralandı (`sort_values('QWK_mean')`, QWK sütunu test metriği). İlk iki finalist üç seed ile eğitildi. Kaydedilen en iyi model de test QWK'sine göre seçildi. | Notebook satır 913–918, 1007, 1092; `finalists_top2.json`; `lowdata_screen_summary_mean_std.csv` |
| Öğretmen seçimi | Dört T2 varyantı eğitildi ve **test QWK'sine göre** sıralandı. Öğretmen notebook'unda seed ayarı yok. Hangi varyantın kullanılacağı low-data notebook'unda sabit yol olarak yazılmış (`Improved_Anchored_SpectralNorm`). | `train_improved_anchored_hybrids.ipynb` satır 622–670 |

## Yeniden üretim

16 Ekim tarihli `verify_archived_final.py` çalıştırması CPU'da, eğitim yapılmadan ve `weights_only=True`, `strict=True` ayarlarıyla gerçekleştirildi. Raporlanan yol olan `archived_residual` için QWK 0.91858190 ± 0.00099292 ve accuracy 0.85467757 ± 0.00314632 bulundu. Üç seed'in confusion matrix'leri birebir aynıdır. AUC farkları 2×10⁻⁵'ten küçüktür.

Bu denetimde aynı görüntü bazlı olasılıklardan değerler yeniden hesaplandı ve sonuç değişmedi (bkz. `metric_recalculation.md`).

## Doğrulanamayanlar

- Sunucudaki orijinal checkpoint'lerin hash'leri.
- Eğitim tarihleri ve eğitim ortamı.
- Öğretmen eğitiminin seed değeri.
- Eğitim sırasında yazılmış bir split logu.
