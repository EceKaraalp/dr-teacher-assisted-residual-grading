# Bilimsel Tutarlılık — Final Denetim (Aşama 3)

Durum etiketleri: **DOĞRULANDI / KISMEN DOĞRULANDI / DOĞRULANAMADI / DÜZELTİLMELİ**.
Makale taslağı bu aşamada değiştirilmedi. Yeni kod/deney çalıştırılmadı; yalnızca mevcut dosyalar okundu.
Yerel `*.pth` dosyası: **yok** (`results\` altında recursive tarandı) → hiçbir checkpoint tabanlı iddia yerelde doğrulanamaz.

## KRİTİK YENİ BULGULAR (Aşama 2 raporlarını düzeltir)

### K1 — Raporlanan nihai model (QWK 0.9186, 27.8M, 6.70 ms) "full-system"dir; makalenin Eq. (8) ile kodu uyuşmuyor — DÜZELTİLMELİ
Kanıt: `train_lowdata_hybrid_distill.ipynb`
- `build_ts_convnext_residual()` (hücre 7) → `ResidualTeacherStudentModel(teacher=HeavyTeacherEnsemble, student=ConvNeXtTinyStudent, residual_scale=0.70, use_conflict_brake=True, conflict_threshold=0.82, conflict_gmax=0.66)`.
- `ResidualTeacherStudentModel.forward`: `z_final = z_teacher + residual_scale * g * (z_student − mean(z_student))`, `g` = öğrenilmiş 13→32→1 sigmoid kapı + conflict brake.
- `evaluate_test` bu modelin `z_final` çıktısını değerlendirir; `ms_per_image` aynı forward (öğretmenler dahil) üzerinden ölçülür; `count_model_params(self.model)` tüm sistemi sayar (öğretmenler `requires_grad=False`).
- Sonuç: `param_total=141,999,859` = 27,824,454 (eğitilebilir; ConvNeXt-T + kapı) + 114,175,405 (donmuş öğretmen sistemi). **27.8M "student" değil, tam sistemdeki eğitilebilir parametredir; 6.70 ms tam sistem gecikmesidir (öğretmenler forward içinde çalışır).**
- Makale (v2 satır ~262-272) "öğrenilmiş kapı, residual mixer veya conflict brake kullanılmaz; `z_final = ½(z_t + z_s)`" diyor. Kod bunun tersini yapıyor. Aşama 2'de "6.70 ms student-only" ifadesi çelişkisiz sanılmıştı; kod kanıtı bunun **desteklenmediğini** gösteriyor.
- `Student_ConvNeXt_Tiny` satırı (`teacher_student_compare_val_test.csv`) acc 0.0518 / QWK 0 / AUC 0.59 — eğitilmemiş/bozuk student logit'i; yani **eğitilmiş student-only metrik yerelde yok**.
- Etki: R1.3, R1.13, R1.16, R2.4 yanıtları ve Tablo 5 yeniden yazılmalı; Eq. (8) koda uyarlanmalı **ya da** deneyler Eq. (8)'e göre yeniden çalıştırılmalı (yazar kararı).
- Çözmek için: sunucudaki nihai `best_model.pth` (3 seed) + teacher checkpoint'leri; student-only ve full-system çıkarım betiği.

### K2 — Teacher 1 QWK 0.640 köken eşleşmesi — KISMEN DOĞRULANDI
- `results\teacher_student_compare\...val_test.csv` Teacher_Baseline test (acc 0.6322, QWK 0.6397, AUC 0.9041) ile `results\improved_unique_anchored_hybrid\Baseline_HybridCNNViT_Gate_Improved` (aynı sayılar, bit düzeyinde eşit) aynı kayıttır.
- Nihai model aynı yolu (`baseline_ckpt = .../Baseline_HybridCNNViT_Gate_Improved/best_model.pth`) yükler. Dolayısıyla **0.640, nihai sistemde kullanılan Teacher 1'in kayıtlı test sonucudur** (kod+sonuç dosyası kanıtı).
- 0.717 (`teacher_student_compare_retrain`) ayrı bir yeniden eğitimdir; 0.696 yerel 2026-10-09 tekrarıdır. Bunlar nihai sistemde kullanılmadı. Seed/split'in 0.640 için belgelenmediği, `.pth` yerelde olmadığı için **checkpoint hash doğrulaması yok**. Tahmin yapılmadı; "0.640 = nihai sistem Teacher 1" ifadesi yalnızca yol/sonuç eşleşmesine dayanır. Makalede 0.717/0.696 geçiyorsa DÜZELTİLMELİ veya "ayrı yeniden eğitim" diye etiketlenmeli.

### K3 — Teacher 2 ve "Gated Ensemble" eşleşmesi — KISMEN DOĞRULANDI
- Teacher 2 test QWK 0.9007 / acc 0.8311 = `results\advanced_hybrid_cnn_vit\Adv_Idea_5_SpectralNorm\test_metrics.json` (bit eşit).
- Tablo 2 "Gated" test (acc 0.8447, QWK 0.9068, AUC 0.9564) = `Improved_Anchored_SpectralNorm` (bit eşit); kod bu checkpoint'i öğretmen toplulukları olarak yükler (`HeavyTeacherEnsemble`: donmuş Baseline + Adv_Idea_5 + kapı; `z_teacher = log((1−g)p_base + g p_adv)`).
- Yani kodda öğretmen birleşimi **olasılık-uzayı kapılı karışım**; makale Eq. (z_t) / "logit ortalaması" betimi buna uymayabilir (Eq. z_t satırı ~226 civarı kontrol edilmeli). DÜZELTİLMELİ (R1.7 ile bağlantılı).

### K4 — R1.1'in gerçek konusu (Aşama 2 düzeltmesi)
Hakem metni: Tablo 2'de equal-weight QWK 0.902 vs gated 0.907 iken metinde "tüm 367 örnekte aynı etiket" deniyor. Bu **öğretmen düzeyi** karşılaştırma. `Teachers_Student_Gate2` / `EqualWeight` (QWK 0.9174 her ikisi) **öğrenci-sistem düzeyi** ayrı karşılaştırma; `Teachers_Student_Gate2` test metrikleri nihai modelin seed 42 sonucuyla aynı (0.91744). Aşama 2 raporunda iki karşılaştırma karıştırılmıştı. Öğretmen düzeyinde: acc 0.8447 vs 0.8311 → etiketler **aynı olamaz** (v2 taslağındaki \sout/\hlcyan düzeltmesi doğru). Öğrenci düzeyinde: acc/F1/QWK/precision/recall bit eşit ama AUC 0.9583 vs 0.9453 → etiketler aynı olabilir, AUC olasılıklardan hesaplandığı için farklı olabilir; örnek bazlı olasılık olmadığı için **DOĞRULANAMADI**. Bootstrap [-0.002, 0.012] ve seed 52 "0.906/0.902" iddiaları için kod/çıktı dosyası bulunamadı → DOĞRULANAMADI (v2'de zaten kaldırıldı).

## Madde madde tablo

| # | Konu | Durum | Kanıt | Makaleye etkisi | Çözüm için gereken |
|---|---|---|---|---|---|
| 1 | 3 seed metrikleri | DOĞRULANDI | `no_gpu_analysis/outputs/per_seed_recompute_vs_recorded.csv`; QWK 0.91744/0.91907/0.91923, ort 0.9186±0.0010 (SD, n=3) | Yok (SD ≠ CI belirtilmeli) | — |
| 2 | Student-only vs full-system | DÜZELTİLMELİ | K1 | Eq. 8, Fig.1 başlığı, Tablo 5, §Efficiency, R1.3/13/16 | Sunucu checkpoint + yeniden değerlendirme |
| 3 | Teacher 1/2 deney kimliği | KISMEN DOĞRULANDI | K2, K3 | Tablo 2 kaynak dipnotu | Teacher checkpoint hash + eğitim logu/seed |
| 4 | Gate2/EqualWeight QWK–AUC | DOĞRULANAMADI | K4; örnek bazlı tahmin yok | v2 düzeltmesi yeterli; R1.1 yanıtı yeniden yazılmalı | Sunucudan `y_prob` veya checkpoint |
| 5 | Tablo 5 parametre | DÜZELTİLMELİ | Kod: toplam 141,999,859 = 27,824,454 + 114,175,405. 8.2M+113.8M=122.0M, 114.18M ile **uyuşmuyor** (fark ≈ −7.8M; hakemin 149.8M'si de kodla uyuşmuyor). Öğretmen bileşen sayıları doğrulanmadı | 142.0M→27.8M düzeltmesi "eğitilebilir" tanımı olarak doğru; öğretmen sayıları "doğrulanmadı" kalmalı | Sunucu checkpoint'ten bileşen bazlı sayım |
| 6 | Test seti ile model seçimi | DOĞRULANDI (sınırlama) | `MODEL_SELECTION_FORENSIC_AUDIT.md`; tablo 1 test QWK'e göre sıralı; `evaluate_test` + finalist seçimi `lowdata_summary` test_metrics.json üzerinden (nb hücre 7: `results_dir/.../test_metrics.json` ile `finalists_top2.json`) | v2'deki protokol paragrafı doğru; sonuçlar "bağımsız test" olarak sunulmamalı; Tablo 4 karşılaştırmasında da not | Yeni val-tabanlı seçim veya harici veri |
| 7 | Fusion CNN–ViT 0.923 > 0.9186 | DOĞRULANDI (çelişki gerçek) | Tablo 1 / `improved_unique_anchored_hybrid` / `unique_anchored_comparison.csv` | Üstünlük iddiası kaldırıldı; "karşılaştırılabilir" doğru. Tek seed/çok seed ayrımı yazılmalı | Fusion için çok-seed |
| 8 | Referable DR & sınıf bazlı | DOĞRULANDI | `referable_dr_mean_std.csv`: sens 0.964±0.004, spec 0.947±0.003 (SD, 3 seed, ≥2 eşiği); Severe recall 0.298 (ort), F1 0.399; seed 62: 5/19 = 0.263 ✓ | Güven aralığı yok; "SD" etiketi gerekli | — |
| 9 | Bootstrap 0.893–0.941 | KISMEN DOĞRULANDI | `cell_bootstrap_ci_per_seed.csv`: hücre-multinomial, 2000 tekrar, seed 42, **eşleştirilmemiş**, yalnız test örnekleme belirsizliği | Yalnız seed 42 için; "paired" gibi sunulmamalı | Örnek bazlı tahmin gerekir (paired) |
| 10 | Kalibrasyon / paired / Grad-CAM | DOĞRULANAMADI | Yerelde `y_prob` ve `.pth` yok | v2'de iddialar kaldırıldı; yanıt "beklemede" | Checkpoint (K1) |
| 11 | "Trainable" tanımı | DÜZELTİLMELİ | 27,824,454 = `requires_grad=True` (ConvNeXt-T + student kapısı, öğretmen donuk) | Tablo 5 başlığında "trainable in full system" yazılmalı; "student" demek yanıltıcı | Kod ile gösterim |

## Eğitim öncesi kapanması gereken metodoloji kararı
Eq. (8) (düz logit ortalaması) mı, kodlanan residual+kapı+brake mi esas alınacak? Hakem istedikleri ablation (R1.4, R1.6, R1.7, R2.3) bu karara bağlı. Yeni eğitim yapılmadan önce karar verin.
