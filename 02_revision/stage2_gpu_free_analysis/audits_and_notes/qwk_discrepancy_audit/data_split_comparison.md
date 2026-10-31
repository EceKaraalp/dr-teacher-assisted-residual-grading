# Veri bölünmesi ve test örneklerinin karşılaştırılması

Kaynak: `recalculation_results.json` → `test_set_comparison` (bu denetimde CPU'da hesaplandı).

## Bölünme tanımı

| | Eski (0.9186) | Yeni (0.9025) |
|---|---|---|
| Bölünme yöntemi | `dataset_loader.get_data_loaders`: stratified `train_test_split`, `random_state=42`, 80/10/10. Bölünme eğitim seed'inden bağımsızdır. | `data.load_data`: dondurulmuş `split_manifest.json`; kimlikler `train.csv` sırasına eşlenir |
| Manifest | Eski koşu eğitim sırasında bir manifest yazmadı. Test kimlikleri 16.10'da yeniden oluşturulup manifestle eşleştirildi. | `split_manifest.json`, SHA-256 `c185dc432225e2f74ab0621207acca2045cda3332a0c5c2d2d43df550b47a23a` |
| Manifest içeriği | `split_seed=42`; sayılar 2928/367/367 | Aynı dosya (paketteki kopyayla hash'ler aynı) |
| Veri kaynakları | `train.csv` `b2a34796…`, `cache_224.npz` `03ae18b2…` (16.10 provenance) | Aynı iki hash, `load_data` içinde zorunlu olarak kontrol ediliyor |
| Örnek sayıları | 2928 / 367 / 367 | 2928 / 367 / 367 (fraction=1.0) |

## Görüntü bazlı test kümesi kontrolleri (her seed için)

| Kontrol | Seed 42 | Seed 52 | Seed 62 |
|---|---|---|---|
| Eski n / benzersiz kimlik | 367 / 367 | 367 / 367 | 367 / 367 |
| Yeni n / benzersiz kimlik | 367 / 367 | 367 / 367 | 367 / 367 |
| Eski kimlik kümesi = manifest | ✔ | ✔ | ✔ |
| Yeni kimlik kümesi = manifest | ✔ | ✔ | ✔ |
| Eski sıra = manifest sırası | ✔ | ✔ | ✔ |
| Yeni sıra = manifest sırası | ✔ | ✔ | ✔ |
| Eski sıra = yeni sıra | ✔ | ✔ | ✔ |
| Eski etiketler = `train.csv` | ✔ | ✔ | ✔ |
| Yeni etiketler = `train.csv` | ✔ | ✔ | ✔ |
| Sınıf dağılımı (0–4) | 181/37/100/19/30 | aynı | aynı |

Eksik, fazladan veya yinelenen test kimliği yoktur.

## Yinelenen görüntüler

Kaynak: `revision_additional_analyses.json` → `duplicate_subset`. Yeniden boyutlandırılmış RGB piksellerin SHA-256 değerleriyle yapılan kontrolde, 367 test görüntüsünden **26'sının** pikselleri train veya validation kümesindeki bir görüntüyle aynıdır. Bu durum iki deney için de aynıdır, çünkü test kümesi aynıdır. Dolayısıyla iki sonuç arasındaki farkı açıklamaz.

## Eğitim kümesi

Yeni koşunun `diagnostic_report.json → config.train_ids` alanı 2928 eğitim kimliğini içerir. Fraction 1.0 olduğu için bu liste manifestin train kümesine eşittir; bu, `data.py` `nested_subset` kodundan çıkarılmıştır. Eski koşu için eğitim kimlikleri kaydedilmemiştir. Aynı bölünmenin kullanıldığı, aynı deterministik `train_test_split(random_state=42)` çağrısına ve test kimliklerinin manifestle eşleşmesine dayanır.

## Sonuç

İki sonuç **aynı 367 test görüntüsünde, aynı sırada ve aynı etiketlerle** hesaplanmıştır. Bu, örnek sayısı benzerliğine değil, kimlik, sıra ve etiket düzeyindeki kontrollere dayanır.

Kalan tek belirsizlik şudur: Eski koşuda eğitim sırasında yazılmış bir split logu yoktur. Test kümesi, sunucudaki `dataset_loader.py` dosyasının depodaki kopyayla aynı olduğu varsayımı altında yeniden oluşturulmuştur. Bununla birlikte, kayıtlı confusion matrix'lerin bu test kümesiyle birebir yeniden üretilmesi bu varsayımı güçlü biçimde destekler.
