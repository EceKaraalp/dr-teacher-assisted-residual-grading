ROUND-2 HAKEM DENEYLERI (reviewer_round2_v1)
=============================================

Amac: Revizyonda "kismen uygulandi" kalan hakem maddeleri icin ek deneyler.
Round-1 kodu (plan.py, run.py, data.py, analyze.py, external.py, dr_pipeline.py, models/) DEGISTIRILMEDEN
kopyalanmistir; yeni dosyalar: plan2.py, run2.py, fusion_baseline.py, analyze2.py, external2.py,
smoke2.py, test_round2.py.

Is matrisi (3 seed: 42/52/62; sabit 2928/367/367 bolunmesi):
  reference : round-1 raw_f100 teacher1, teacher2, joint_full, student_ce (12 is). Round-1 Drive klasorunden
              SHA-256 dogrulamasiyla SADECE OKUNARAK kopyalanir. Bulunamazsa round-1 koduyla yeniden egitilir
              ve reference_manifest.json'a "retrained" yazilir.
  t2indep   : Teacher 2 TEK BASINA agirlikli CE ile (6 is: ogretmen + joint_full ogrenci)   -> Hakem 1.6 / 2.3
  nosn      : Spectral norm OLMADAN Teacher 2, ayni kapili asama + joint_full ogrenci (6 is) -> Hakem 1.8
  aug       : TUM asamalarda augmentation: T1, T2, joint_full, student_ce (12 is)          -> Hakem 2.2
  fusion    : Tablo-1 Fusion CNN-ViT; tarihsel tarif ve student_ce ile eslenik tarif (6 is) -> Hakem 1.5
Toplam 30 yeni egitim (+ gerekirse 12 referans).

Analizler (analysis2/):
  - Gated (birincil) ve equal yollarin bagimsiz CE ogrenciye gore eslenik bootstrap araliklari (Hakem 1.4/1.9/2.4)
  - Round-1 etiket butceleri (%10/25/50/100) icin gated-CE ve equal-CE araliklari (round-1 test_logits.npz)
  - Her yeni kolun referansa gore farki, kalibrasyon (NLL/Brier/ECE, validasyonla sicaklik)
Dis degerlendirme (external2/):
  - IDRiD 103 resmi test (tum sistemler) + istege bagli Messidor-2 ve DDR
  - Gated birincil cikti icin Grad-CAM (round-1 ile ayni 10 vaka)                          -> Hakem 2.5

Kurallar: test seti egitimde/secimde kullanilmaz; secim validasyon QWK. Test degerlendirmesi ancak
istenen ailelerin tum egitimleri bittikten sonra yapilir. Biten islerde last.pth silinir (best.pth kalir).
Sonuclarin olumlu olacagi varsayilmaz; tum sayilar oldugu gibi raporlanmalidir.

Dis veri formatlari (--extra AD GORUNTULER ETIKETLER FORMAT):
  messidor2 : Krause ve ark. adjudicated CSV (image_id, adjudicated_dr_grade, adjudicated_gradable);
              derecelendirilemeyenler cikarilir.
  ddr       : DDR DR_grading/test.txt ("dosya derece"); derece 5 (ungradable) cikarilir.
  generic   : CSV sutunlari image, grade (istege bagli gradable).
GORUNTULER bir ZIP veya klasor olabilir; goruntu dosya adi (uzantisiz) etiket dosyasindaki ad ile eslesmelidir.
