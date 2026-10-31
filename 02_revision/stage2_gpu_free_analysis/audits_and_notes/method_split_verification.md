# Aşama 4A — Yöntem ve Veri Bölünmesi Kanıt Denetimi

Kapsam: yalnızca denetim. Eğitim, GPU, checkpoint yükleme/resume ve makale dosyası değişikliği yapılmadı. Önceki raporlar otomatik doğru kabul edilmedi; kod, sonuç dosyaları ve manifest ile yeniden eşleştirildi.

**Kanıt düzeyleri:** **K1** = kayıtlı deney çıktısı (JSON/CSV/notebook çıktısı) · **K2** = kod + sonuç dosyası uyumu (CPU'da yeniden hesaplama dahil) · **K3** = yalnızca kod çıkarımı · **K4** = kanıt yok.
**Durum etiketleri:** DOĞRULANDI / KISMEN DOĞRULANDI / DOĞRULANAMADI / DÜZELTİLMELİ.

Not (yan etki): CPU'da model sınıfları örneklenirken torchvision ConvNeXt-T ImageNet ağırlıkları (~109 MB) torch hub önbelleğine indi. Proje dosyalarında değişiklik yok.

---

## 1. `ResidualTeacherStudentModel` ↔ Denklem 8

Kaynak: `train_lowdata_hybrid_distill.ipynb`, hücre 5 (model tanımları); aynı sınıf `inference_only_teacher_student_comparison.ipynb` içinde de tanımlı (model tanım hücresi).

| Öğe | Makale (v2 tex ≈ l.262–272, Eq. 8) | Kod | Durum |
|---|---|---|---|
| Birleştirme | `z_final = ½(z_t + z_s)` | `z_final = z_teacher + residual_scale · g · (z_student − mean_c(z_student))` | **DÜZELTİLMELİ** (K2) |
| Öğrenilen kapı | "learned gate / mixer yok" | `self.gate = Linear(13,32)-ReLU-Linear(32,1)-Sigmoid` (481 parametre, eğitilebilir) | **DÜZELTİLMELİ** |
| Kapı girdisi | – | `[p_t(5), p_s(5), H_t, H_s, \|conf_t − conf_s\|]` = 13 | – |
| Conflict brake | yok | `teacher_conf > thr` ve `argmax` farklı ise `g = min(g, gmax·(1 − teacher_conf))` | **DÜZELTİLMELİ** |
| 0.7 katsayısı | yok | `residual_scale=0.70` (ölçek **kapı çıkışı ile çarpılır**, ağırlık ortalaması değil) | **DÜZELTİLMELİ** |
| Öğretmen logiti | `z_t = ½(z_base + z_adv)` (Sec. "Teacher Ensemble via Equal-Weight Fusion", ≈ l.209–214) | `HeavyTeacherEnsemble`: 13→64→32→1 sigmoid kapı, `p = (1−g)p_base + g p_adv`, `z_t = log(clamp(p,1e-8))` — **olasılık uzayında kapılı karışım** | **DÜZELTİLMELİ** |
| Student-only mod | "softmax(z_s) kompakt dağıtım modu" | Hiçbir yerde ayrı değerlendirilmedi; `Student_ConvNeXt_Tiny` satırları (acc 0.0518, QWK 0) eğitilmemiş/bozuk öğrenciye ait | **DOĞRULANAMADI** |

**Çağrı yerleri**
- Tanım: hücre 5. Kurulum: `build_ts_convnext_residual()` (hücre 5/7) → `residual_scale=0.70, use_conflict_brake=True, conflict_threshold=0.82, conflict_gmax=0.66`.
- Eğitim kaybı hücre 6 `train_epoch`: CE(**z_final**) + λ_ni·ReLU(CE_sistem − CE_öğretmen + margin) + λ_kd·KD(z_final→z_t.detach(), T=[2,4], w=[0.6,0.4]) + λ_ord·ordinal MSE − 0.01·gate entropisi. ConvNeXt: λ_kd=0.30, λ_ord=0.12, λ_ni 0.8→0.25, margin 0.010→0.003, AdamW 2e-4/1e-4, CosineWarmRestarts(10,2), 70 epoch, patience 10, klip 1.0.
- `validate` ve `evaluate_test` aynı tam modeli çağırır; en iyi checkpoint **validation QWK** ile seçilir.
- "Gate" ve "brake" terimleri makale metninde **parametresiz** olarak anlatılıyor; kodla çelişiyor.

**Yeni bulgu (hiperparametre uyuşmazlığı, K1+K2):** `inference_only_teacher_student_comparison.ipynb`, hücre 6, modeli `conflict_threshold=0.80, conflict_gmax=0.70` ile kuruyor; eğitim `0.82 / 0.66` kullandı. Bu parametreler state-dict'te yok (düz float). Tohum-42 sonucu (acc 0.852861, QWK 0.917439) iki yolda aynı çıktı; bu, brake'in test setinde sonucu değiştirmediğini düşündürür ama **nedeni doğrulanamaz** (örnek bazlı tahmin yok). Makale değerlerini yazarken hangi eşiğin raporlandığı belirtilmeli.

**Inference notebook'unda "EqualWeight"** (`eval_teacher_student_equal_weight`): `z_final = (z_t + z_s)/2`, aynı Gate2 checkpoint'indeki öğretmen+öğrenci ile, **kapı/residual yolu olmadan** (K1; çıktı satırı: Acc 0.8529, QWK 0.9174, AUC 0.9453). Bu, Eq. 8'in yazıldığı biçimin *yalnızca çıkarım zamanında, eğitimle uyumsuz* uygulamasıdır; eğitilen model bu formülle eğitilmedi. Gate2 ile Equal arasında acc/QWK birebir aynı, AUC farklı (0.9583 vs 0.9453) ⇒ aynı olasılık vektörleri **olamaz**; sadece argmax'lar (veya en azından confusion matrisi) aynı. Eşleşme tesadüf olabilir veya doğrulanamaz.

## 2. Nihai üç seed'in kökeni

| Soru | Bulgu | Düzey |
|---|---|---|
| Model sınıfı / config | `ResidualTeacherStudentModel` + `ConvNeXtTinyStudent` (torchvision `convnext_tiny`, IMAGENET1K_V1, son Linear→5), `TS_ConvNeXtTiny_Residual` | K1/K2 |
| Seed'ler | `FINAL_SEEDS=[42,52,62]`, `SCREEN_SEEDS=[42]`, `TOPK=2` (hücre 7) | K2 |
| Çıktı yolu | `results/lowdata_hybrid_distill/TS_ConvNeXtTiny_Residual/seed_{42,52,62}/` (`test_metrics.json`, `training_history.json`); checkpoint `best_model.pth` | K1 (JSON) / K4 (dosya) |
| Öğretmen ckpt yolları | `results/improved_unique_anchored_hybrid/Baseline_HybridCNNViT_Gate_Improved/best_model.pth` ve `.../Improved_Anchored_SpectralNorm/best_model.pth` (hücre 7) | K2 |
| Tahmin kodu | `evaluate_test` (hücre 6) — best_model.pth'yi yükler | K2 |
| Yerel `.pth` | **Hiçbiri yok** (`results\` özyinelemeli tarandı; `export\` yalnızca kod) | K4 |
| Örnek bazlı olasılık | **Yok**; yalnız `confusion_matrix` ve özet metrikler | K4 |
| Donanım | Notebook çıktısı: "GPU: NVIDIA RTX A5000", "Device: cuda" (sunucu); `batch_size = 16` | K1 |
| Final komut/log | Notebook hücre çıktıları dışında bağımsız log yok | K4 |

Kayıtlı sonuçlar (`lowdata_seed_results.csv`, Stage=final):

| Seed | Test QWK | Acc | Best Val QWK | ms/img | Eğitilen epoch (en iyi) |
|---|---|---|---|---|---|
| 42 | 0.917439 | 0.852861 | 0.8976 | 6.7067 | 20 (10) |
| 52 | 0.919075 | 0.858311 | 0.8859 | 6.6886 | 14 (4) |
| 62 | 0.919232 | 0.852861 | 0.8889 | 6.7029 | 21 (11) |

Ortalama QWK = **0.91858** (≈ 0.9186) → **DOĞRULANDI (K1)** — yalnızca bu kayıtlı sonuçlara göre; seed 52'de en iyi epoch 4 olması ve val QWK < test QWK (~0.886–0.898 vs ~0.917–0.919) belirgin; val/test farkı yorumlanmalı.

**Seed-42 satırı** `teacher_student_inference_comparison.csv` içindeki `Teachers_Student_Gate2` ile aynı (acc 0.8529, QWK 0.9174) ⇒ aynı tam sistem (K1+K2).

## 3. 0.9186 QWK ve 6.70 ms kapsamı

- **QWK 0.9186:** öğretmen + öğrenci + kapı tam sistemi (`z_final` çıktısı üzerinden). **Öğrenci-tek QWK değildir.** Düzey: K2 (kod + `param_total=141,999,859` öğretmen dahil).
- **6.70 ms:** `evaluate_test`'te test yükleyicisi (batch 16) üzerinde toplam geçen süre / N. Tam sistemi (iki öğretmen + öğrenci), host→device kopyasını, veri yükleyici süresini ve ilk batch ısınmasını içerir; `torch.cuda.synchronize()` var (hücre 6, l.~850/869). Cihaz = RTX A5000 (K1). Makaledeki donanım ifadesiyle (ve R1/R2'nin "student 6.7 ms vs ensemble ×3" karşılaştırmasıyla) **uyumsuz olabilir**: 6.70 ms tam sistem değeridir, öğrenci-tek gecikme ölçülmedi. **DÜZELTİLMELİ.** Öğrenci-tek ve öğretmen-tek gecikme için yeni ölçüm gerekir (GPU, sunucu).
- Makalenin "student only 27.8M / hızlı" anlatımı tam-sistem ölçümüyle çelişir.

## 4. Parametre sayıları

Kaynak: `param_*` anahtarları üç seed'in `test_metrics.json` dosyalarında (K1) + notebook sınıflarının CPU'da örneklenmesi, **ağırlık yüklemeden** (K2). Her iki yöntem birebir uyuştu.

| Bileşen | Parametre | Durum |
|---|---|---|
| `HybridCNNViTBase` (Teacher 1) | 4,851,623 | donmuş |
| `AdvancedHybridModel` (Adv_Idea_5_SpectralNorm, Teacher 2) | 109,320,773 | donmuş |
| Öğretmen kapısı (13→64→32→1) | 3,009 | donmuş (`requires_grad=False` tüm `teacher_model.parameters()` için) |
| **Öğretmen toplam** | **114,175,405** | donmuş |
| ConvNeXt-T öğrenci | 27,823,973 | eğitilebilir |
| Karışım kapısı (13→32→1) | 481 | eğitilebilir |
| **Eğitilebilir** | **27,824,454** | = 27,823,973 + 481 |
| **Toplam (tam sistem)** | **141,999,859** | |

- Tablo 5'teki "8.2M + 113.8M (+27.8M = 149.8M)": kodla **uyuşmuyor** (gerçek: 4.85M + 109.32M + 0.003M; toplam 142.0M). **DÜZELTİLMELİ.** 8.2M / 113.8M / 149.8M için kaynak bulunamadı (**DOĞRULANAMADI**).
- "27.8M" = öğrenci + karışım kapısı; **öğretmen içeren bir modelin eğitilebilir parametre sayısı**, "öğrenci modelinin tek başına büyüklüğü" olarak anılabilir ancak (27.82M) öğrenci-only sayımı bu şekilde ayrı verilmeli.
- Sınırlama: state-dict anahtarları/boyutları doğrulanmadı (ckpt yok). Sayımlar kayıtlı JSON ile uyumlu CPU örneklemesidir.

**Kod çıkarımı (K3):** `ResidualTeacherStudentModel` eğitimde `self.model.train()` çağrılınca öğretmen alt modülleri de train moduna geçer (dropout açık, BatchNorm istatistikleri güncellenir; `requires_grad=False` yalnız gradyanı keser). `HeavyTeacherEnsemble.__init__` içindeki `.eval()` bunu kalıcı korumaz. Üstelik `best_model.pth` tüm `state_dict`'i (öğretmen dahil) kaydeder. Checkpoint olmadığından doğrulanamadı; sonuçları değiştirdiği kanıtlanmadı, ama "donmuş öğretmen" ifadesi ve distilasyon hedefi (`z_t.detach()` train-mode öğretmenden) etkilenebilir. Sunucuda checkpoint'lerde BN `running_mean` karşılaştırması gerekir.

## 5. Veri bölünmesi

Kaynaklar: `dataset_loader.py` (≈ l.96–150), `results/split_manifest.json`, `APTOS2019/train.csv`, `APTOS2019/cache_224.npz`.

- Kod: `train_test_split(test_size=0.1, random_state=42, stratify=y)` ardından `test_size=0.1/0.9, random_state=42, stratify`; `train.csv` satır sırası; 224×224 bilinear, normalizasyon ve augmentation **yok**.
- CPU'da aynı bölünme yeniden çalıştırıldı: **train 2928 / val 367 / test 367**; `id_code` kümeleri `split_manifest.json` ile **birebir eşit**; ikili kesişim = 0; birleşim = 3662 (K2). Notebook çıktısı da "Train 2928 / Val 367 / Test 367" (K1).

| Sınıf (0/1/2/3/4) | train.csv | train | val | test |
|---|---|---|---|---|
| Sayı | 1805/370/999/193/295 | 1443/296/799/155/235 | 181/37/100/19/30 | 181/37/100/19/30 |

- Notebook'taki sınıf ağırlıkları {0.4058, 1.978, 0.7329, 3.778, 2.492} train sayılarından hesaplananlarla eşleşiyor (DOĞRULANDI).
- Word makalesindeki **2929/366/367** bu koddan üretilmiyor; kod/manifest 2928/367/367. **DÜZELTİLMELİ** (makale metni).
- Manifest eğitimden sonra `revision_experiments\analyze_existing_dual_predictions.py` ile yazıldı; bölünmenin eğitimde kullanıldığı kanıtı = aynı kod + aynı seed ile yeniden üretilebilirlik + notebook çıktısı sayıları (KISMEN DOĞRULANDI; eğitim anındaki id listesi saklanmamış).

### Yinelenen görüntü bulgusu (yeni; önceki raporlarda yok)
224 px önbellekte birebir piksel-eşit görüntüler (MD5):
- 251 görüntü, 123 grup; train 197 / val 26 / test 28.
- **45 grup farklı bölmelere yayılıyor**; **30 grup çelişkili etiket** içeriyor.
- Test görüntülerinden **22'si train'de, 26'sı train veya val'de** birebir kopyaya sahip (val için 20 train kopyası). Test–train kopya çiftlerinde etiket uyumu 17 uyuşan / 6 çelişen.
- Sonuç: kısmi veri sızıntısı riski (test ≈ %7 oranında) ve etiket gürültüsü. Bu çalışma 224 px yeniden boyutlanmış görüntüde yapıldı; orijinal çözünürlükte kopya olmayabilir/ek yakın kopyalar olabilir. Etkisi ölçülmedi. **DÜZELTİLMELİ / açıklanmalı** (Limitations + gerekirse tekilleştirilmiş test alt kümesi analizi; checkpoint+örnek tahmini gerekir).

## 6. Test setinin model seçimindeki rolü

- Tarama (seed 42) ve finalist seçimi **test QWK** ile yapıldı (`screen_summary_df` `QWK_mean` sıralı; `finalists_top2.json`: ConvNeXtTiny_Residual, LiteCNNViT_Conflict_OrdinalDistill); `best_overall_model_info.json` test QWK'ya göre seed 62'yi (0.91923) seçip `best_overall_model.pth`'e kopyalıyor (hücre 7–8).
- Run-içi checkpoint (epoch) seçimi validation QWK ile.
- Sonuç: raporlanan test QWK model ailesi seçimi açısından **bağımsız test sonucu değildir** (iyimser yanlılık). **DOĞRULANDI (K2)**; makalede protokol ve yorum bölümlerinde belirtilmeli.

## 7. Öğretmen kimliği (özet yeniden eşleme)

- Teacher 1 test (acc 0.6322, QWK 0.6397, AUC 0.9041) = `improved_unique_anchored_hybrid/Baseline_HybridCNNViT_Gate_Improved` = final modelin yüklediği baseline (K1+K2). 0.717 → `teacher_student_compare_retrain`; 0.696 → yerel 2026-10-09 yeniden üretimi. Inference notebook ise baseline olarak `teacher_student_compare_retrain/Teacher1_Baseline` yüklüyor, yani **final eğitimdeki öğretmenle aynı checkpoint değil** (sonra tam state_dict Gate2 ckpt'siyle üzerine yazıldığı için sonuç etkilenmez, çünkü öğretmen ağırlıkları `gate2_model.load_state_dict` ile değişir).
- Teacher 2: `advanced_hybrid_cnn_vit/Adv_Idea_5_SpectralNorm` QWK 0.9007; "Gated" `Improved_Anchored_SpectralNorm` acc 0.8447, QWK 0.9068 (K1). R1.1'in 0.902 vs 0.907 karşılaştırması öğretmen düzeyindedir.

## 8. Çözülemeyen sorular

1. Üç final seed hangi komut/oturumla çalıştı (notebook dışı log yok)? Brake eşikleri 0.82/0.66 mı (eğitim) yoksa 0.80/0.70 mi (çıkarım) raporlanacak?
2. Eq. 8 koda mı uyarlanacak, yoksa kod Eq. 8'e göre yeniden mi eğitilecek? (Bilimsel karar — yazar.)
3. Checkpoint SHA-256'ları ve state-dict anahtar/boyut doğrulaması (sunucu).
4. Öğretmen BN/eval durumu final checkpoint'lerde (sunucu).
5. Örnek bazlı olasılıklar (kalibrasyon, eşli test, Grad-CAM, yinelenen-örnek analizi için).
6. 6.70 ms: öğrenci-tek ve öğretmen-tek gecikme ölçümü yok.
7. Gate2 ↔ EqualWeight aynı acc/QWK, farklı AUC: aynı argmax tahminleri mi?
8. Yinelenen görüntülerin sonuçlara etkisi (22 test görüntüsü).
9. Word'deki 2929/366/367 sayılarının kaynağı.
10. Tablo 5'teki 8.2M / 113.8M / 149.8M'nin kaynağı.

## 9. Bu aşamada değiştirilen dosyalar
Yalnızca bu dosya oluşturuldu. `0_main.tex`, Word, v1/v2 taslakları, notebook'lar ve sonuç dosyaları değişmedi.
