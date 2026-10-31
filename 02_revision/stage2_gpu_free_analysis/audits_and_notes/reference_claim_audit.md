# Atıf–İddia Denetimi (Aşama 3)

Kapsam: v2 tex + Word kaynakça (`v2_work\refs.json`, `crossref.json`, `reference_audit.md`). Harici doğrulama: yalnızca Crossref DOI/üstveri (Aşama 2) ve **bir** yerel PDF (Farag 2022). Diğer kaynakların içeriği okunmadı → "DOĞRULANAMADI" (içerik düzeyinde).

## A. Tablo 4 karşılaştırma değerleri

| İddia (v2 Tablo 4) | Atıf | Kanıt | Sorun | İşlem | Etiket |
|---|---|---|---|---|---|
| Farag: acc 82.00, QWK 0.888 (APTOS, 5 sınıf) | farag2022 (IEEE Access 2022) | Yerel PDF `Automatic_Severity_Classification…DenseNet…CBAM.pdf` özeti: "82% accuracy – 0.888 (QWK)" severity grading | Bölme/protokol (tek split, kaç test) okunmadı | Protokol sütunu için bölüm II-IV okunmalı | DOĞRULANDI (sayılar) / protokol DOĞRULANAMADI |
| Kobat et al.: 84.90 acc, 0.769† (kappa), F1 0.708, 10-fold CV | oh2022 (anahtar adı) → Diagnostics 2022 | Crossref üstveri (yazar/dergi); sayılar okunmadı | Word [40] farklı kaynak (Oh et al., Sci. Rep. 2021); "†" kappa tanımı QWK olmayabilir | PDF okunmalı; kappa tipi doğrulanmalı; çapraz doğrulama ≠ tek split notu | DOĞRULANAMADI |
| Handoyo & Kusuma: 84.17 acc, F1 0.702 | minarno2022 | Crossref (yazar/sayfa 881–887) | Sayılar okunmadı; "split vs CV" bilinmiyor | PDF kontrolü | DOĞRULANAMADI |
| Zhang CvT-13: 84.31 / QWK 0.840 / AUC 0.97 | zhang… (tex) | Okunmadı | AUC/QWK protokolü | PDF kontrolü | DOĞRULANAMADI |
| Al Shafi: QWK 0.900‡ KD baseline | alshafi… | Okunmadı | ‡ notunun anlamı | PDF kontrolü | DOĞRULANAMADI |
| Karşılaştırmanın adilliği | — | Bu çalışmanın sayıları test-seti-etkili seçimle (bkz. consistency K/6) | Aynı bölme/aynı metrik değil; "üstünlük" dili uygun değil | Tablo altı not (R1.17) | DÜZELTİLMELİ |

## B. İddia–atıf eşleşmesi (R1.18'in özellikle istediği alanlar)

| İddia | Atıf (tex anahtarı) | Kanıt | Sorun | İşlem | Etiket |
|---|---|---|---|---|---|
| AdamW kullanıldı | loshchilov2019 | v2'de eklendi; Crossref/arXiv kimliği tanınan klasik makale | Dergi sürümü/DOI Word'de yok | Word listesine eklenmeli | KISMEN (kaynak mevcut, içerik doğrulaması yapılmadı) |
| Spektral norm kalibrasyonu/Lipschitz | miyato/ spectral norm kaynakları | İçerik okunmadı; hakem bu bağın zayıf olduğunu söylüyor | Spektral norm → kalibrasyon bağı kaynakla desteklenmiyor olabilir; kendi ECE ölçümü yok | Cümle "Lipschitz kısıtı" ile sınırla veya ölçümle destekle | DÜZELTİLMELİ |
| Ordinal öğrenme kaybı | ordinal kaynakları | Okunmadı | Kaynak sınıflandırma/ordinal kaybı mı, QWK-hizalı mı belirsiz | Okuma | DOĞRULANAMADI |
| Çoklu öğretmen çeşitliliği | zhou2012, kuncheva2014 (tex'e eklenmiş) | Word'de yok; kitap/üstveri | İlgili iddia "çeşitlilik" ölçümü yapılmadan kullanılıyor (R1.6) | Çeşitlilik ölçümü yapılana kadar yumuşat | DÜZELTİLMELİ |
| Kapı çökmesi/entropi kapısı | (öz bulgular) | Gate mean 0.09 sonuçları kayıt kaynaklı mı? | `gate collapse` sayıları (0.092/0.087/0.091) için çıktı dosyası bulunamadı | Kayıt çıktısı gerekli | DOĞRULANAMADI |

## C. Word ↔ LaTeX kaynakça farkları (otomatik eşleştirilmedi)
- Word [40] Oh et al. (Sci. Rep. 2021) ≠ tex `oh2022` (Kobat et al., Diagnostics 2022): **farklı kayıt**; hangisinin Tablo 4'te amaçlandığı yazarlarca doğrulanmalı. DÜZELTİLMELİ.
- Word [10], [11], [33] tex'te yok; tex'te loshchilov2019, zhou2012, kuncheva2014 Word'de yok. Alıntı yeri bazında kontrol gerekli.
- Word 46 numaralı, tex 49 bibitem: farklılık yukarıdaki üç + ek farklardan.
- Aşama 2'de Crossref ile düzeltilen kayıtlar: cao2020, liu2020, oh2022 yazarları, minarno2022, nabavi2024, yang2025, elassiouti2024 — üstveri düzeyinde DOĞRULANDI (Crossref), iddia desteği DOĞRULANAMADI.

## Genel sonuç
Sayısal olarak doğrulanan tek Tablo 4 kaynağı Farag 2022'dir. Diğer satırlar için PDF erişimi gerekir; erişilene kadar "doğrulanmadı" kalmalı. Kullanıcıdan istenen: yerelde bulunmayan 4 makalenin PDF'leri.
