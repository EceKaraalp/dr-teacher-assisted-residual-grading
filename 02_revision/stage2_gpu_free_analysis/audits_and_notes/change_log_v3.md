# change_log_v3

Bu aşamada makale taslağı (v1, v2), `0_main.tex`, Word dosyası **değiştirilmedi**. Silme/taşıma/yeniden adlandırma yok. GPU/eğitim yok. Yalnızca `pypdf` (D:\dr_env\venv) PDF metin okuma için aşama başında kuruldu.

Yeni dosyalar:
- `scientific_consistency_final.md` — K1–K4 kritik bulgular, 11 maddelik tablo.
- `reviewer_response_verification.md` — 21+5 yorum; özgün metin durumu.
- `reference_claim_audit.md` — Tablo 4 ve atıf-iddia denetimi.
- `submission_readiness_checklist.md`
- `revision_phase3_summary.md`

Aşama 2 raporlarına etkisi (dosyalar değişmedi, burada düzeltilir):
- `scientific_consistency_review.md` §3 ve `response_to_reviewers_draft_v2.md` R1.1: öğretmen düzeyi (Tablo 2) ile student-sistem düzeyi karıştırılmış.
- v2 taslak ve yanıtlardaki "6.70 ms student-only" / "27.8M student" ifadeleri kod kanıtıyla desteklenmiyor (v3 öneri: v2'ye sonraki aşamada düzelt).
- v2 Eq. (8) betimi koda uymuyor.

Kanıt dosyaları: `train_lowdata_hybrid_distill.ipynb` (hücre 4–7), `results\teacher_student_compare\*.csv`, `results\improved_unique_anchored_hybrid\*`, `results\advanced_hybrid_cnn_vit\Adv_Idea_5_SpectralNorm\test_metrics.json`, `no_gpu_analysis\outputs\*.csv`, hakem PDF.
