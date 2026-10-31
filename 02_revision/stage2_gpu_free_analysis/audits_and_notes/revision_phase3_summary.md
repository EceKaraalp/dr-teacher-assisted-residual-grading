# Revizyon Aşama 3 Özeti

## 1. Tamamlanan GPU'suz işler
- Hakem 1 özgün metni (PDF) bulundu ve yanıtlarla karşılaştırıldı (`reviewer_response_verification.md`).
- Kodla sonuç dosyalarının bit düzeyinde eşleştirilmesi: Teacher 1 = `Baseline_HybridCNNViT_Gate_Improved` (QWK 0.6397); Teacher 2 = `Adv_Idea_5_SpectralNorm` (0.9007); Gated öğretmen = `Improved_Anchored_SpectralNorm` (0.9068).
- Notebook kod incelemesi: nihai model residual+kapı+brake sistemi; 6.70 ms ve 27.8M tam sistem değerleridir.
- Seed-62 Severe 5/19, referable DR, bootstrap yeniden doğrulandı.
- Tablo 4'ten Farag 2022 sayıları yerel PDF'ten doğrulandı.
- Raporlar: `scientific_consistency_final.md`, `reference_claim_audit.md`, `reviewer_response_verification.md`, `submission_readiness_checklist.md`, `change_log_v3.md`.

## 2. Açık bilimsel riskler
1. Eq. (8) (düz logit ortalaması) koddakiyle uyuşmuyor → makalenin ana yöntem tarifi yanlış olabilir.
2. Student-only metrik yok; "6.70 ms student-only" desteklenmiyor.
3. Test setiyle model seçimi (finalist sıralaması test QWK'ye göre).
4. Öğretmen parametre sayıları (8.2M+113.8M) 114.18M donmuş toplamla çelişiyor.
5. Teacher 1 için checkpoint/seed belgesiz.
6. Gate2/EqualWeight AUC farkı örnek bazlı olasılık olmadan açıklanamıyor.
7. Tablo 4 karşılaştırması protokol açısından adil değil/doğrulanmadı.

## 3. Yazar kararı gerektirenler
- Eq. (8)'i koda göre yeniden yazmak mı, yöntemi Eq. (8)'e göre yeniden deneylemek mi.
- AI-use beyanının gerçek içeriği; Acknowledgements; kod/manifest yayını; split sayıları (2929 vs 2928).
- Zorunlu deneylere ayrılacak GPU süresi (8 gün → 18 Ekim).

## 4. Sunucudan alınması gerekenler (bkz. `checkpoint_retrieval_plan.md`)
- Nihai model `TS_ConvNeXtTiny_Residual` seed 42/52/62 `best_model.pth` (+ `test_metrics.json`).
- Teacher 1: `improved_unique_anchored_hybrid\Baseline_HybridCNNViT_Gate_Improved\best_model.pth`; öğretmen topluluğu: `Improved_Anchored_SpectralNorm\best_model.pth`; Teacher 2: `advanced_hybrid_cnn_vit\Adv_Idea_5_SpectralNorm\best_model.pth`.
- Hash, `state_dict` anahtarları, `param` sayımı doğrulaması; örnek bazlı `y_prob` dışa aktarımı.

## 5. Kanıt doğrulandıktan sonra yapılabilecek deneyler
Student-only/full-system değerlendirme; kalibrasyon (ECE/Brier/NLL); paired bootstrap; Grad-CAM; T2-only vs T1+T2; ablation (R1.4, 1.7, 1.11); normalizasyon (R1.9); veri oranı (R1.10); augmentation (R2.2); harici veri (R2.1); aynı donanımda gecikme (R2.4).

## 6. Zorunlu / ertelenebilir
- **Zorunlu**: madde 1–4 riskleri çözmek, yukarıdaki deneyler, Yanıt belgesi düzeltmeleri (R1.1, 1.3, 1.13, 1.16, R2.4), AI beyanı kararı, derleme.
- **Ertelenebilir**: biçim temizliği, Tablo 4 protokol sütunu (PDF'ler gelince), ondalık tutarlılığı.
