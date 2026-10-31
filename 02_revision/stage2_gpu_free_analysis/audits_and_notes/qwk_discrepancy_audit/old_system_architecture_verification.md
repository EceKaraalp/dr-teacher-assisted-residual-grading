# Eski (sunucu) sisteminin yapı doğrulaması — `D:\m1`

Kapsam: salt okunur inceleme. Checkpoint'ler `torch.load(mmap=True)` ile açıldı; GPU kullanılmadı, eğitim yapılmadı ve hiçbir dosya değiştirilmedi.

Kısaltmalar:
- **NB-S**: `D:\m1\Documents\train_lowdata_hybrid_distill.ipynb` (öğrenci eğitimi). Satır numaraları, kod hücreleri birleştirildikten sonraki satırlardır.
- **NB-T**: `<repo_root>\train_improved_anchored_hybrids.ipynb` (öğretmen eğitimi). `D:\m1` içinde bulunmuyor.

## 1. Kesinleşen sonuçlar

| # | Soru | Cevap | Kanıt | Durum |
|---|---|---|---|---|
| 1 | T1 nasıl eğitildi? | Tek başına, sınıf ağırlıklı CE ile, `HybridCNNViTBase`, AdamW 1e-3. 64 epoch koştu, en iyi val QWK 49. epokta (0.7439), test QWK 0.6397 | NB-T `BaselineTrainer` (190–240) ve 585–603; T1 `checkpoint_current.pth` (optimizer 134 tensör = T1'in tamamı, lr 1e-3); `training_history.json` (anahtarlar train_loss, val_loss, val_acc, val_qwk) | Kesin |
| 2 | T2 tek başına eğitildi mi? | **Hayır.** T2, T1 dondurulmuşken öğrenilmiş öğretmen kapısıyla birlikte **ortak** eğitildi. Kayıp, karışık çıktı `log p_final` üzerindedir | NB-T `ImprovedBaselineAnchoredEnsemble` (129–186) ve `ImprovedAnchoredTrainer` (342–500, kayıp 478. satır); ensemble geçmişinde `gate_mean`, `lambda_noninf` ve `margin_noninf` anahtarları var | Kesin |
| 3 | T1 ile T2 nasıl birleşiyor? | Örneğe bağlı **öğrenilmiş olasılık kapısı**: `p_T = (1−g₁)·p₁ + g₁·p₂`, `z_T = log(max(p_T, 1e-8))`. Eşit ağırlık değil | NB-T 160–186, NB-S 285–324; checkpoint'te `gate` 13→64→32→1 | Kesin |
| 4 | Kullanılan ensemble checkpoint'i hangisi? | `Documents\best_model.pth` = `Improved_Anchored_SpectralNorm`: 34 epoch, en iyi val 16. epokta (0.8780), test QWK 0.9068, acc 0.8447. Kapı ortalaması 0.51'den 0.87'ye çıkıyor, yani T2 baskın | `test_metrics.json` ve `training_history.json` SHA-256 değerleri Anchored klasöründekiyle aynı. Ensemble içindeki T1 parametreleri tek başına T1 checkpoint'iyle birebir aynı (134/134), dolayısıyla bu ensemble yan deneydeki yeniden eğitilmiş T1'i değil, orijinal T1'i içeriyor | Kesin (.pth ile JSON'ların aynı koşudan geldiği ortak klasör ve T1 eşleşmesiyle destekleniyor) |
| 5 | Öğrenci tek başına mı eğitildi? | **Hayır.** Öğrenci (ConvNeXt Tiny) ve dış kapı, birleşik residual çıktı `z_final` üzerinden eğitildi. Öğrencinin kendi logitine doğrudan kayıp uygulanmadı | NB-S `ResidualTeacherStudentModel` (348–419) ve eğitim kaybı (533–565); seed 42 optimizer'ı 186 tensör içeriyor, toplam 27,824,454 = öğrenci 27,823,973 + dış kapı 481 | Kesin |
| 6 | Öğrenci eğitiminde öğretmen eğitildi mi? | Ağırlıkları **hayır**: `requires_grad=False`, T1/T2/kapı parametreleri öğretmen ensemble'ı ile birebir aynı. Ancak `model.train()` (513) öğretmeni de train moduna aldığı için **BN istatistikleri ve spektral-norm u/v vektörleri güncellendi** | Ağırlık karşılaştırması: T2'de 311 tensör aynı, 161 farklı; farklı olanların tamamı BN buffer'ı veya `weight_u`/`weight_v`. Öğretmen kapısının 6 tensörü aynı | Kesin |
| 7 | Makaledeki 0.9186 hangi çıktı? | Kapılı residual `z_final` (NB-S val/test: `self.model(images)`). Seed başına 0.9174 / 0.9191 / 0.9192 | `seed */test_metrics.json` ve önceki yeniden hesaplama (`metric_recalculation.md`) | Kesin |
| 8 | "EqualWeight" ne? | **Yalnızca çıkarımda** `(z_T + z_S)/2`, aynı kapılı eğitilmiş ağırlıklarla. Bu yolla hiç eğitim yapılmadı. Seed 42 sonucu 0.917439 (kapılıyla aynı etiketler) | `inference_only_teacher_student_comparison.ipynb` hücre 7–8 ve çıktıları | Kesin |

## 2. Bir görüntünün çıktıya kadar izlediği yol (eski sistem, test zamanı)

1. **Girdi:** APTOS PNG → RGB → 224×224 bilinear → uint8 → float tensör, **0–255 aralığında**. Normalizasyon ve augmentation yok (`dataset_loader.py` 36–69; `D:\m1` içindeki `.py` dosyalarında `Normalize`/`/255` yok).
2. **T1 (`HybridCNNViTBase`):** CNN backbone, çok ölçekli dikkat ve `cnn_fc` ile patch embedding, 6 transformer bloğu ve `vit_fc` → iç **güven kapılı CNN–ViT füzyonu** (`fusion.cnn_conf`, `vit_conf`, `gate`) → `output_head` (512→5) → `z₁` → `p₁ = softmax(z₁)`.
3. **T2 (`AdvancedHybridModel`, Adv_Idea_5_SpectralNorm):** ResNet50 özellikleri (2048) ile ViT-B/16 CLS'si (768) birleşiyor → 2816 → spektral normlu doğrusal katman → `z₂` → `p₂`.
4. **Öğretmen kapısı (Gate1):** girdi `[p₁, p₂, H(p₁), H(p₂), |max p₁ − max p₂|]` (13) → MLP 13→64→32→1 → sigmoid → `g₁`. `p_T = (1−g₁)p₁ + g₁p₂` ve `z_T = log p_T`.
5. **Öğrenci (ConvNeXt Tiny, ImageNet ön eğitimli, son katman 768→5):** `z_S`, `p_S = softmax(z_S)`.
6. **Dış kapı (Gate2):** girdi `[p_T, p_S, H(p_T), H(p_S), güven farkı]` (13) → 32 → 1 → sigmoid → `g₂`.
7. **Çatışma freni:** `conf_T = max p_T` ve `conf_T > 0.82` ise ve öğretmenle öğrencinin tahmini farklıysa `g₂ ← min(g₂, 0.66·(1 − conf_T))` (NB-S 397–404, ayarlar 784–786).
8. **Residual karışım:** `z_final = z_T + 0.70 · g₂ · (z_S − mean(z_S))` (NB-S 407–408).
9. **Karar:** `argmax softmax(z_final)` → derece 0–4. QWK `cohen_kappa_score(weights='quadratic')` ile hesaplanıyor.

## 3. Eğitim aşamaları

| Aşama | Eğitilen | Dondurulan | Kaybın uygulandığı çıktı | Kayıp | Seçim |
|---|---|---|---|---|---|
| 1 | T1 | – | `z₁` | Sınıf ağırlıklı CE | En iyi val QWK |
| 2 | T2 ve öğretmen kapısı | T1 (eval değil; BN istatistikleri güncelleniyor) | `log p_T` (karışım) | Ağırlıklı NLL + λ·relu(NLL_karışım − NLL_T1 + margin) (λ 1.5→0.6, margin 0.02→0.005) + ek kayıplar − 0.02·H(kapı) | Val QWK ile en iyi epoch. **4 T2 varyantı test QWK'sine göre sıralanmış** (NB-T 671, 681) |
| 3 | Öğrenci ve dış kapı | T1, T2, öğretmen kapısı (ağırlıklar donuk, BN ve u/v güncelleniyor) | `z_final` (residual karışım) | Ağırlıklı CE + λ·relu(CE_final − CE_T + margin) (λ 0.8→0.25, margin 0.010→0.003) + 0.30·çoklu-T KL(`z_final` ∥ `z_T`), T=2 ve 4, ağırlıklar 0.6/0.4 + 0.12·ordinal CDF-MSE − 0.01·H(kapı). Gradyan kırpma 1.0, AdamW 2e-4, wd 1e-4 | Val QWK (`z_final`). Öğrenci mimarisi seçimi test QWK'sine göre yapılmış (önceki denetim) |

## 4. Makale metniyle karşılaştırma

| Konu | Makale | Eski kod ve checkpoint'ler |
|---|---|---|
| Öğretmen eğitimi | İki bağımsız öğretmen | T1 bağımsız; **T2, T1 donukken karışım içinde ortak** eğitilmiş |
| Öğretmen birleşimi | Eşit ağırlıklı logit ortalaması | **Öğrenilmiş, örneğe bağlı olasılık kapısı** |
| Öğrenci eğitimi | Yumuşak hedeflerle bağımsız damıtma, 4 kayıp öğrenci logitinde | Öğrenci **residual düzeltme** olarak eğitilmiş; kayıplar birleşik `z_final` üzerinde |
| Çıkarım | `(z_T + z_S)/2` | `z_T + 0.70·g₂·(z_S − mean)` ve çatışma freni |
| Sayılar | 0.9186 | Kapılı çıktı. Eşit çıkarım aynı ağırlıklarla 0.9174; makaledeki formülle (ham öğretmen logit ortalaması) 0.9083 |
| Öğrenci tek başına | – | QWK yaklaşık 0 (tek sınıfa çöküyor). Öğrenci tek başına sınıflandırıcı değil |

## 5. Yan deney: bağımsız öğretmenler (öğrenciye bağlı değil)

`compare_teachers_student_equal_weight.ipynb`, öğretmenleri `results/teacher_student_compare_retrain/` altında **ayrı ayrı** yeniden eğitmiş:
- T1 tek başına.
- T2 tek başına (100 epoch, 1e-4).
- Donuk iki öğretmen üzerinde yalnızca kapı.
- Eşit ağırlıklı öğretmen (`(p₁ + p₂)/2`).

Bu deneyin sonuçları (test acc: T1 0.6948, T2 0.8120, eşit 0.8147, kapılı 0.8120) `inference_only...` hücre 11'de elle yazılmış.

**Bu öğretmenler, raporlanan öğrenci sistemine girmedi.** `inference_only...` defteri yeniden eğitilmiş öğretmenleri yüklüyor, ancak ardından Gate2 checkpoint'ini aynı modül nesnesine `load_state_dict` ile yüklüyor. Gate2 checkpoint'i `teacher_model.*` anahtarlarını da içerdiği için öğretmen ağırlıkları orijinal anchored öğretmenlerle değiştiriliyor (hücre 6). Ayrıca:
- Yeniden eğitilmiş checkpoint'ler `D:\m1` içinde yok.
- Bu defterde çatışma freni 0.80/0.70 ile kurulmuş; eğitim 0.82/0.66 ile yapılmış. Seed 42 test sonucu yine 0.917439 çıkıyor.

## 6. Yeni 99 işlik Colab deneyi ile eşleşme

Karşılaştırılan kod: `colab_reviewer_complete\run.py` (`build_model`, `compute_loss`, eğitim döngüsü), `dr_pipeline.py` (`HeavyTeacherEnsemble`, `ResidualTeacherStudentModel`, `STUDENT_*`) ve `plan.py`.

| Bileşen | Eski | Yeni (raw, %100) | Eşleşme |
|---|---|---|---|
| T1 | Tek başına, ağırlıklı CE, 110/15/1e-3 | Aynı (`teacher1`) | Aynı |
| T2 | T1 donuk, kapıyla ortak, NLL + non-inf (1.5→0.6, 0.02→0.005) − 0.02·H; SpectralNorm config'inde ek kayıp yok | Aynı (`teacher2`, run.py 177–183) | Aynı |
| Öğretmen birleşimi | Öğrenilmiş kapı `(1−g)p₁ + g·p₂` | `fusion_mode="gate"` | Aynı |
| Öğrenci | Residual kapılı, 0.70, fren 0.82/0.66; kayıplar `z_final` üzerinde (CE, non-inf, KD T=2,4, ordinal, −H) | `mix="gated"`, `STUDENT_MODEL_KW`, `joint_full` kaybı | Aynı |
| Öğretmen BN | Train modunda güncelleniyor | Aynı (yalnızca `teacher_eval` kontrol kolunda eval) | Aynı |
| Girdi | 224, 0–255, normalizasyon ve augmentation yok | `raw` | Aynı |
| Split | 2928/367/367 | Aynı manifest | Aynı |
| Optimizer ve scheduler | AdamW wd 1e-4, CosineWR(10,2), clip 1.0, 2e-4/70/10 | Aynı | Aynı |
| Öğretmen tekrarı | Tek öğretmen seti, seed yok; 3 seed'in hepsinde aynı öğretmen | Her seed için öğretmenler yeniden eğitildi | **Farklı** (bilerek) |
| Seçim | T2 varyantı ve öğrenci mimarisi test QWK'sine göre seçildi | Yalnızca val QWK; mimari sabit | **Farklı** (bilerek, test sızıntısı yok) |
| Birincil çıktı | `z_final` (kapılı) → 0.9186 | `primary` = `z_final` → 0.8996 ± 0.0077; `equal` → 0.9025 ± 0.0066 | Aynı tanım |

Sonuç: yeni deney, eski modelin mimarisini ve eğitim tarifini birebir tekrarlıyor. Farklar yalnızca öğretmenlerin her seed için yeniden eğitilmesi ve test kümesine göre seçim yapılmaması.

## 7. Doğrulanamayanlar

- NB-T'nin sunucuda tam olarak hangi sürümle çalıştırıldığı (defter `D:\m1` içinde değil; yerel kopya inceledi).
- Öğretmen eğitiminde seed ayarı (NB-T'de seed belirlenmemiş).
- `Documents\best_model.pth` ile Anchored klasöründeki JSON'ların aynı koşuya ait olduğu doğrudan hash ile değil; ortak klasör, bire bir T1 eşleşmesi ve anahtar yapısıyla destekleniyor.
