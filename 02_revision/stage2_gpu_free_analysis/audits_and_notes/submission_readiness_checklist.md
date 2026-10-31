# Gönderim Hazırlık Kontrol Listesi (Aşama 3)

LaTeX motoru yerelde **yok**; kurulum yapılmadı, **derleme yapılmadı** ve başarılı derleme iddia edilmez.

## 1. Derleme bağımlılıkları
| Öğe | Durum | Not |
|---|---|---|
| `cas-dc.cls` / `cas-common.sty` | Eksik (yerelde bulunamadı) | Elsevier CAS şablonu gerekli |
| `.bst` (örn. `cas-model2-names.bst`) | Eksik | `formatting_audit.md`: v2 elle `thebibliography` kullanıyor |
| `figures\confusion_matrix.png` | Eksik | Şekil uydurulmadı |
| Mimari şekil `prism-uploads\*.pdf` | Bulunamadı | Gönderilen Word'den çıkarılabilir; yazar sağlamalı |
| LaTeX derleyici | Yok | Overleaf/MiKTeX gerekir |
| `\hlcyan`, `\sout` (soul/ulem) | Derleme doğrulanamadı | Revizyon işaretleri son sürümde kaldırılmalı |
| Çift yükleme (natbib/caption) | Şüpheli | Derlemede doğrulanmalı |
| Statik: süslü parantez, 27/27 `\begin/\end`, 8 `\ref` | Geçti (Aşama 2) | Derleme kanıtı değil |

## 2. Yazar onayı gerektirenler
| Madde | Durum |
|---|---|
| Generative-AI beyanı (R1.20) | Eklenmedi. Word'deki "Claude yalnız dil düzeltmesi" ifadesi revizyon sürecindeki yapay zekâ desteğiyle tutarlı olmayabilir; gerçek kullanım yazarlarca belirtilmeli |
| Acknowledgements | Yazar metni gerekli |
| Data availability | Taslakta eklendi (APTOS kamusal); **kod/split manifest yayını** yazar onayı |
| Kod ve `split_manifest.json` yayını | Karar bekliyor (repo/Zenodo) |
| Funding / Conflict of interest / CRediT | Bilgi yazarlardan |
| Informed consent / etik | APTOS kamusal veri; gerekiyorsa "yeni insan verisi toplanmadı" ifadesi yazar onayıyla |
| Split sayıları (Word 2929/366/367, tex 2928/367/367) | Yazar hangisinin gerçek olduğunu manifestle doğrulamalı (manifest 2928/367/367) |

## 3. Gönderim engelleyicileri (bilimsel)
1. Eq. (8) ↔ kod uyuşmazlığı ve student-only/full-system ayrımı (`scientific_consistency_final.md` K1).
2. Tablo 5 öğretmen parametre sayıları.
3. Yanıt belgesindeki R1.1, 1.3, 2.4 düzeltmeleri.
4. Hakemin zorunlu deneyleri (R1.4, 1.6–1.10, R2.1–2.3, 2.5) — sunucu checkpoint ve GPU gerekli.

## 4. Ertelenebilir
Biçim temizliği, `\hlcyan` kaldırma, ondalık tutarlılığı, Tablo 5 konumu (R1.21).
