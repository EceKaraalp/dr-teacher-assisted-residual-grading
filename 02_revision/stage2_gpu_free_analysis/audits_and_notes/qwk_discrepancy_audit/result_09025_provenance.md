# 0.9025 QWK sonucunun kaynağı

## Özet

QWK 0.9025 ± 0.0066 ve accuracy %84.83 ± %1.03, `reviewer_complete_v1` adlı yeni 99 işlik yeniden eğitim matrisinden gelir. Değerler `raw_f100_s{42,52,62}/student/joint_full` koşularının **"equal" çıkarım yolunun** üç seed ortalamasıdır. Bu yolda, kapılı eğitimle elde edilmiş ağırlıklar kullanılır ve çıkarım `(z_t + z_s)/2` ile yapılır. Yani bu yol ayrı bir eşit-füzyon eğitimi değildir. Seed bazlı görüntü logitleri raporlanan confusion matrix'leri birebir yeniden üretir.

## Kanıtlar

| Bilgi | Değer | Kanıt |
|---|---|---|
| Sonuç dosyası | `reviewer_complete_v1/analysis/per_seed.json` ve `mean_sd.json` | `C:\Users\user\Desktop\reviewer_complete_v1.zip` (SHA-256 `b05488ec…`) |
| Deney adı | `RUN_ID = reviewer_complete_v1`, kol `joint_full`, ön işleme `raw`, veri oranı 1.0 | `protocol.json` (SHA-256 `4f846a27…`); `status.json` (`training_jobs: 99, complete: true`) |
| Deney tarihi | Doğrulanamadı. Yerel kopyalar 23.10.2026 tarihli. | Dosya sistemi |
| Seed bazlı QWK (equal) | 42: 0.909939 · 52: 0.900073 · 62: 0.897400 | `per_seed.json` |
| Seed bazlı accuracy (equal) | 0.855586 · 0.836512 · 0.852861 | `per_seed.json` |
| Ortalama ± SD | QWK 0.902471 ± 0.006589; Acc 0.848320 ± 0.010330 | Bu denetimde yeniden hesaplandı |
| SD türü | **Örneklem SD'si (ddof=1)**. Popülasyon SD'si 0.0054 olurdu. | `analyze.py` satır 188, 256 (`np.std(..., ddof=1)`) |
| Üç seed ortalaması mı? | Evet | |
| Mimari | Aynı `ResidualTeacherStudentModel` sınıfı (`mix="gated"`, `teacher_mode="gate"`) ve aynı ayarlar: `residual_scale=0.70`, conflict brake 0.82/0.66. T1 = `HybridCNNViTBase`, T2 = `AdvancedHybridModel(Adv_Idea_5_SpectralNorm)`, öğrenci ConvNeXt Tiny. | `colab_reviewer_complete\run.py` `build_model`, `teacher_modules`; `dr_pipeline.py` `STUDENT_MODEL_KW` |
| Öğretmenler | **Her seed için yeniden eğitildi.** Önce `teacher1` (seed s), ardından T1 bağımlılığıyla `teacher2` (seed s). Ardından öğrenci eğitildi. | `run.py` satır 112–158 (`P.set_seed(run.seed)`); `protocol.json` → `runs` |
| Öğrenci checkpoint'leri (yerelde yok) | 42: `67252f6bf3b52a6b18f4d1c2f216df48be0f2463d95d4a1c7ff098f95fad75d0` · 52: `1d7db802299314a49ca884f4bf9a71171adab942bcf212c8d4446ffb8ed76e0e` · 62: `16789cbbbb7f6ed941aa9f327cff47a3c69a9bbe172051357c3d0b22feca3f22` | `diagnostic_report.json` → `complete.best_sha256`. Dosyaların kendisi bu denetimde hash'lenemedi. |
| Checkpoint'in rolü | Doğrulama QWK'si (primary/kapılı yol) en iyi olan epoch | `run.py` satır 298–304; protokol `selection: validation QWK` |
| En iyi epoch / tamamlanan epoch | 30/40, 7/17, 4/14 (1 tabanlı) | `diagnostic_report.json` (`checkpoint_epoch_zero_based` 29/6/3) |
| En iyi doğrulama QWK'si | 0.8970 · 0.8909 · 0.8821 (ortalama 0.8900) | Aynı dosya |
| Veri bölünmesi | Dondurulmuş manifest, 2928/367/367. CSV ve cache hash kontrolleri ile kimlik çakışması kontrolü yapılıyor. | `data.py` `load_data`; manifest SHA-256 `c185dc43…` |
| Test kimlikleri | 367 kimlik, manifestle aynı küme ve aynı sıra | `seedXX_test_diagnostic.npz` → `ids` |
| Sınıf sırası | 0–4, `labels=range(5)` | `analyze.py` |
| Kayıplar | Kapılı birleşik skor `u` üzerinde: ağırlıklı CE; çoklu sıcaklıklı KD; ordinal CDF; non-inferiority (0.8→0.25, 0.010→0.003); kapı entropisi | `run.py` satır 160–196 |
| Değerlendirme ve QWK | Ayrı test aşaması; `argmax` sonrası `cohen_kappa_score(y, pred, labels=range(5), weights='quadratic')` | `run.py` `evaluate`; `dr_common.compute_full_metrics`; `analyze.py` |
| Ortam | NVIDIA A100-SXM4-40GB, torch 2.11.0+cu130 (tanı çıkarımı) | `diagnostic_report.json` |

## Aynı koşudan diğer yollar (3 seed, ortalama ± örneklem SD)

| Yol | QWK | Accuracy |
|---|---|---|
| primary (kapılı residual; eski 0.9186 ile aynı tür çıktı) | 0.8996 ± 0.0077 | 0.8438 ± 0.0079 |
| equal `(z_t+z_s)/2` | **0.9025 ± 0.0066** | **0.8483 ± 0.0103** |
| teacher (kapılı öğretmen) | 0.8944 ± 0.0056 | 0.8411 ± 0.0031 |
| student (öğrenci dalı tek başına) | 0.2634 ± 0.4561 | 0.2570 ± 0.3140 |
| ayrı T1 / ayrı T2 eğitim aşamaları | 0.6814 ± 0.0232 / 0.8969 ± 0.0046 | 0.6912 / 0.8365 |

## Doğrulanamayanlar

- Checkpoint dosyalarının kendi hash'leri (dosyalar Drive'da).
- Yeni öğretmen checkpoint hash'leri.
- Gerçek eğitim tarihleri.
- Eğitim GPU'sunun tanı GPU'suyla aynı olup olmadığı.
