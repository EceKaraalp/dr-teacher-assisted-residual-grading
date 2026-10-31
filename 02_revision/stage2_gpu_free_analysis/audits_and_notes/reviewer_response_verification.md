# Hakem Yanıtı Doğrulama (Aşama 3)

**Özgün metin durumu**
- Hakem 1: **MEVCUT, birebir** — `C:\Users\user\Desktop\Round1\2020440-REVIEWER-V1-20260919093559.pdf` (çıkarım: `v2_work\reviewer_pdf_joined.txt`). 21 madde.
- Hakem 2: yalnızca `Round1\Yeni Metin Belgesi.txt` (Türkçe, kullanıcıdan gelen metin; resmi PDF'te değil). Resmî İngilizce özgün metin **bulunamadı**; yeniden yazılmadı.
- `response_to_reviewers_draft_v2.md` numaralaması R1.1–R1.21 ile hakem PDF'iyle **uyumlu** (kontrol edildi). Yanıt içeriği PDF'ten değil plan dosyasından özetlenmişti; aşağıdaki PDF'ten doğrulanmıştır.

Etiketler: DOĞRULANDI / KISMEN / DOĞRULANAMADI / DÜZELTİLMELİ (yanıt metni için).

| Yorum | Özgün metin | Mevcut yanıtın kanıtı | Eksik / düzeltme | Etiket |
|---|---|---|---|---|
| R1.1 QWK 0.902 vs 0.907, "aynı etiket" iddiası, bootstrap | Var | Taslakta çelişki kabul edildi, iddia kaldırıldı | Yanıt metni **student düzeyi Gate2/EqualWeight** ile karıştırılmış; Tablo 2 (öğretmen düzeyi) esas alınmalı. Bootstrap/metrik uygulaması örnek bazlı çıktı olmadan yeniden kontrol edilemiyor | DÜZELTİLMELİ |
| R1.2 Test setiyle model seçimi | Var | v2 "Model-Selection Protocol" paragrafı, Limitations | Seçim prosedürü (finalist = test QWK sıralı) tam yazılmalı; bağımsız test gibi sunulmamalı | KISMEN |
| R1.3 Student-only vs full | Var | Eq. 8 betimi, "student-only 6.70 ms" | **Kod: 6.70 ms, 27.8M ve 0.9186 full-system** (bkz. `scientific_consistency_final.md` K1). Yanıt şu an yanlış öncül taşıyor; student-only metrikler yok | DÜZELTİLMELİ |
| R1.4 Kontrollü ablation (aynı ConvNeXt-T) | Var | [PENDING] | Deney yapılmadı | DOĞRULANAMADI (beklemede) |
| R1.5 Fusion CNN–ViT üstünlüğü | Var | 0.923>0.9186 notu, iddia yumuşatıldı | Çok-seed Fusion sonucu yok | DOĞRULANDI (metin) / deney eksik |
| R1.6 Teacher 1 katkısı + çeşitlilik | Var | [PENDING] | T2-only distilasyon, disagreement/KL/hata korelasyonu yapılmadı | DOĞRULANAMADI |
| R1.7 Logit vs olasılık vs T-kalibre ortalama | Var | [PENDING] | Kodda öğretmen birleşimi olasılık-uzayı kapılı (K3); Eq. ile çelişebilir | DÜZELTİLMELİ |
| R1.8 Kalibrasyon (ECE/Brier/NLL) | Var | İddialar kaldırıldı | `y_prob`/checkpoint yok; analiz yapılamadı | DOĞRULANAMADI |
| R1.9 ImageNet normalizasyonu | Var | [PENDING], "ham tensör" notu | Kod incelenmeli; ablation yok | DOĞRULANAMADI |
| R1.10 Düşük veri (10/25/50/100%) | Var | [PENDING] | Deney yok; alternatif: iddia yumuşatma | DOĞRULANAMADI |
| R1.11 Non-inferiority kaybı (Eq. 5) | Var | [PENDING] | Notebook ile Eq. 5 kıyası, detach durumu, ablation yapılmadı | DOĞRULANAMADI |
| R1.12 Tekrarlanabilirlik | Var | Kısmen (split manifest, kod) | Optimizer/LR/batch/epoch, donanım, öğretmen seed'leri tablosu eksik; öğretmen eğitim logları yok | KISMEN |
| R1.13 Tablo 5 / gecikme | Var | 142.0M→27.8M düzeltmesi | Hakemin 149.8M'si de kodla uyuşmuyor; öğretmen sayıları doğrulanmadı; gecikme tam sistem ve donanım belgesiz | DÜZELTİLMELİ |
| R1.14 İstatistik | Var | Hücre-bootstrap (seed 42, eşleştirilmemiş) | Paired bootstrap ve örnekleme birimi/tekrar/seed tanımı; CI ≠ SD ayrımı | KISMEN |
| R1.15 Sınıf bazlı, referable DR | Var | CSV'ler: Severe 5/19 (seed 62), sens/spec | Specificity/precision tabloları ve CI (örnek bazlı değil, hücre bootstrap) | KISMEN → DOĞRULANDI (sayılar) |
| R1.16 Fig. 2 modu | Var | [PENDING] | Kod: Fig.2 tam sistem; student-only matris yok | DÜZELTİLMELİ |
| R1.17 Literatür karşılaştırması | Var | Tablo 4 isimleri düzeltildi, kappa notu | Kaynak protokolleri doğrulanamadı (`reference_claim_audit.md`) | KISMEN |
| R1.18 Kaynakça | Var | Crossref DOI düzeltmeleri | İddia–kaynak desteği yarısı doğrulanamadı | KISMEN |
| R1.19 Klinik iddialar | Var | Yumuşatıldı | Son okuma | KISMEN |
| R1.20 AI beyanı | Var | Eklenmedi (yazar kararı) | Hakemin önerdiği ifade yazarlarca **doğrulanmadan** kullanılamaz; revizyonda yapay zekâ desteği alındıysa beyan buna uygun olmalı | DOĞRULANAMADI (yazar) |
| R1.21 Sunum | Var | Statik kontroller | Derleme yapılamadı; Tablo 5 yeri | KISMEN |
| R2.1 Harici veri | Metin: yalnız Türkçe notta | Sınırlama | Harici veri yok | DOĞRULANAMADI |
| R2.2 Augmentation | aynı | Sınırlama + [PENDING] | Ablation yok | DOĞRULANAMADI |
| R2.3 Zayıf Teacher 1 | aynı | Yanıt T1 ~63% (0.640 doğru, kayıtlı) | "0.902 vs 0.901" ifadesi: 0.9016 (Equal) vs 0.9007 (T2) doğru ama gated 0.907; T2-only deneyi yok | KISMEN |
| R2.4 Hız/yük | aynı | Gecikme ifadesi | Kod: 6.70 ms tam sistem → yanıt yanlış öncüllü | DÜZELTİLMELİ |
| R2.5 Grad-CAM | aynı | Paragraf kaldırıldı | Checkpoint gerekli | DOĞRULANAMADI |

## Özet
- Düzeltilmeli yanıtlar: R1.1, R1.3, R1.7, R1.13, R1.16, R2.4.
- Yeni deney bekleyenler: R1.4, 1.6, 1.8–1.11, R2.1–2.3, 2.5.
- Yazar kararı: R1.20.
