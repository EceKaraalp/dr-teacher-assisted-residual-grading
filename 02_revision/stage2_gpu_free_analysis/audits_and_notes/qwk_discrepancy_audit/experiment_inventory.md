# Deney ve kanıt envanteri

Denetim tarihi: 2026-10-24 (yerel). Bu denetim salt okunurdur: eğitim yapılmadı, GPU kullanılmadı, checkpoint, makale veya sonuç dosyası değiştirilmedi. Bu klasörde yalnızca yeni rapor ve analiz dosyaları oluşturuldu.

Dosya adları tek başına kanıt olarak kabul edilmedi. Aşağıdaki "içerik doğrulaması" sütunu, dosyanın içeriğinin nasıl kontrol edildiğini gösterir.

## 1. İstenen dosyalar

| İstenen kalem | Bulunan dosya | İçerik doğrulaması |
|---|---|---|
| Completed_Experiments_Evidence.docx | `<repo_root>\revision_submission_package\04_evidence_and_audits\Completed_Experiments_Evidence.docx` (234,351 bayt). Aynı boyutta kopyalar: `Desktop\04_evidence_and_audits\` ve `Round1\2020440_completed_experiments_evidence.docx` | Revizyon paketinden üretilen ikincil özettir. Bu denetimde birincil kaynak olarak kullanılmadı. |
| Received_Results_Audit.json | `...\revision_submission_package\04_evidence_and_audits\Received_Results_Audit.json` (18,398 bayt). Aynı boyutta kopyalar: oturum `files\received_results_audit.json` ve `Desktop\04_evidence_and_audits\` | İkincil denetim kaydıdır. Sayılar bu denetimde birincil kaynaklardan yeniden hesaplandı. |
| Orijinal makale | `C:\Users\user\Desktop\Round1\2020440-SECRETARY-R0-V1-20260828233914.docx`, SHA-256 `6261ede63cfb487f69755b7188c43034340ec98554ab4f8b4af1da0cb9c6aefb` | Özette "0.9186 ± 0.0010" ve "85.47% ± 0.31%" ifadeleri yer alıyor. |
| Eski deney sonuçları (0.9186) | `D:\m1\Documents\lowdata_seed_results.csv` (SHA-256 `fcbcac714bc56764…`). Depo kopyası: `results\lowdata_hybrid_distill\` (seed CSV, özet CSV'leri, `finalists_top2.json`, her seed için `test_metrics.json` ve `training_history.json`) | Satırlar ve confusion matrix'ler okundu, ortalama ve standart sapma yeniden hesaplandı. |
| Eski checkpoint'ler | `D:\m1\Documents\seed 42|52|62\best_model.pth` ve `checkpoint_current.pth`; `D:\m1\Documents\best_model.pth` (öğretmen topluluğu); T1 `...Baseline_HybridCNNViT_Gate_Improved\best_model.pth` | SHA-256 değerleri yeniden hesaplandı (`archive_file_hashes.json`). Ağırlık anahtarları ve şekilleri 16 Ekim doğrulamasında `strict=True` yüklemeyle kontrol edildi. |
| Eski eğitim kodu | `D:\m1\Documents\train_lowdata_hybrid_distill.ipynb` (SHA-256 `3c124978…`); öğretmen eğitimi `DR_Research\train_improved_anchored_hybrids.ipynb` (SHA-256 `ae5248b7…`); `dataset_loader.py` | İlgili kod satırları okundu. Bu dosyalar hiçbir zaman çalıştırılmadı. |
| Eski değerlendirmenin yeniden üretimi | `results\archived_final_verification_2026-10-16\` (`summary.json`, `provenance.json`, `seed_*_predictions.csv`, `seed_*_probabilities.npz`, `verification_report.txt`, `revision_additional_analyses.json`); script `revision_experiments\verify_archived_final.py` | Kayıtlı görüntü bazlı olasılıklar ve etiketler bu denetimde yeniden kullanıldı. |
| Yeni deney sonuçları (0.9025) | `C:\Users\user\Desktop\reviewer_complete_v1.zip`, SHA-256 `b05488ec0d91accf5dc2a7706a006625d5611d758ebd676b113715adf19d559f` (`analysis/per_seed.json`, `mean_sd.json`, `paired_qwk.json`, `status.json` → 99 eğitim işi tamamlanmış) | Confusion matrix'lerden QWK yeniden hesaplandı. |
| Yeni deneyde görüntü bazlı test çıktıları | `C:\Users\user\Downloads\student_diagnostic_20261023_175134_909746.zip`, SHA-256 `66450eb4…`: seed başına `seedXX_test_diagnostic.npz` (`ids`, `labels`, `primary`, `equal`, `teacher`, `student` logitleri) ve `diagnostic_report.json` | Yeni checkpoint'lerin A100 üzerinde yalnızca çıkarımla yeniden değerlendirilmesidir. Raporlanan confusion matrix'lerle birebir eşleşiyor (bkz. `metric_recalculation.md`). |
| Yeni deney protokolü ve kodu | `Downloads\protocol.json` (SHA-256 `4f846a27…`); paket `DR_Research\colab_reviewer_complete\` (`run.py`, `data.py`, `dr_pipeline.py`, `dr_common.py`, `analyze.py`, `MANIFEST.json`) | İlgili kod satırları okundu. |
| Split manifesti | `results\split_manifest.json` ve `colab_reviewer_complete\split_manifest.json`. İki dosyanın SHA-256 değeri aynı: `c185dc43…a23a` | Test kimlikleri, sırası ve etiketleri karşılaştırıldı. |
| Test tahminleri ve gerçek etiketler | Eski: `seed_*_probabilities.npz` ve `seed_*_predictions.csv`. Yeni: `seedXX_test_diagnostic.npz` | Kimlik, sıra ve etiket kontrolleri yapıldı. |
| Confusion matrix dosyaları | Eski: `test_metrics.json` ve `summary.json`. Yeni: `per_seed.json` (PNG dosyaları da var) | Matrisler sayısal olarak karşılaştırıldı. |
| Equal Weight / Gate2 karşılaştırmaları | `D:\m1\Documents\teacher_student_inference_comparison.csv`, `teacher_student_unified_comparison.csv`, `compare_teachers_student_equal_weight.ipynb`; yeni deneyde `paired_qwk.json` | Bkz. `fusion_method_comparison.md`. |
| Konfigürasyon ve seed kayıtları | Eski: notebook içindeki kodlanmış ayarlar. Yeni: `protocol.json` ve `diagnostic_report.json` içindeki `config` alanı | Ayarlar karşılaştırıldı. |

## 2. Bulunamayan veya erişilemeyen dosyalar

- Eski checkpoint'lerin **sunucudaki orijinal SHA-256 değerleri** bulunamadı. Yalnızca yerel kopyaların hash'leri mevcut.
- Eski eğitim sırasında yazılmış **split veya test kimliği logu** bulunamadı. Eski test kimlikleri sonradan yeniden oluşturuldu ve manifestle eşleştirildi.
- Eski öğretmen eğitiminin **seed değeri** bulunamadı. Öğretmen notebook'unda `manual_seed`, `np.random.seed` veya `random.seed` çağrısı yok.
- Eski eğitimin **donanım, PyTorch, timm ve CUDA sürümlerini** gösteren bir kayıt bulunamadı.
- Eski öğretmen eğitiminden kalan dört T2 varyantının metrik tablosu bulunamadı. Bu tablo, notebook'taki `comparison_df` çıktısı olmalıydı.
- Yeni deneyin **öğrenci ve öğretmen checkpoint dosyaları** yerelde yok; Drive'da duruyorlar. Öğrenci checkpoint'lerinin SHA-256 değerleri `diagnostic_report.json` içinde kayıtlı. Yeni öğretmen checkpoint hash'leri yerelde bulunamadı.
- Yeni deneyde iş başına üretilen `history` ve `complete` JSON dosyaları yerelde yok. ZIP içinde yalnızca analiz özetleri var. En iyi epoch ve doğrulama QWK değerleri `diagnostic_report.json` dosyasından alındı.
- Yeni deneyin gerçek eğitim tarih ve saatleri doğrulanamadı. Yerel dosya tarihleri yalnızca kopyalama zamanını gösteriyor (protocol.json: 23.10.2026).

## 3. Bu denetimde oluşturulan analiz dosyaları

- `archive_file_hashes.json`: `D:\m1\Documents` altındaki checkpoint, notebook, JSON ve CSV dosyalarının boyutları ve SHA-256 değerleri.
- `recalculation_results.json`: görüntü bazlı yeniden hesaplama, test kümesi karşılaştırması, seed bazlı ortalama ve standart sapma değerleri.
- `old_vs_new_paired_bootstrap.json`: eski ve yeni sonuçlar için eşleştirilmiş görüntü bootstrap'ı (10,000 örnekleme).
