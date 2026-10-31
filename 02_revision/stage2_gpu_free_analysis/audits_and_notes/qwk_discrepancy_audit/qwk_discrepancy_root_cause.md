# QWK farkının kök neden denetimi (0.9186 ile 0.9025)

Ayrıntılı kanıtlar şu dosyalardadır: `experiment_inventory.md`, `result_09186_provenance.md`, `result_09025_provenance.md`, `checkpoint_identity_comparison.md`, `data_split_comparison.md`, `metric_recalculation.md`, `fusion_method_comparison.md` ve JSON çıktıları.

## Kısa cevaplar

| Soru | Cevap | Kanıt düzeyi |
|---|---|---|
| İki sonuç hangi deneylerden geliyor? | **0.9186:** eski sunucu koşusu `results/lowdata_hybrid_distill`, `TS_ConvNeXtTiny_Residual`, final aşaması, seed 42/52/62, kapılı residual çıktı. **0.9025:** `reviewer_complete_v1`, `raw_f100_s{42,52,62}/student/joint_full`, equal `(z_t+z_s)/2` çıktı. | Kesin. İki sonuç da görüntü bazlı çıktılardan birebir yeniden üretildi. |
| Aynı model mi kullanılmış? | **Hayır.** Mimari tanımı aynıdır, fakat yeni deneyde öğretmenler (T1, T2) ve öğrenci her seed için yeniden eğitilmiştir. Eski deneyde üç seed tek bir öğretmen çiftini paylaşıyordu. Aynı görüntülerde 35–51 tahmin farkı vardır. | Kesin (işlevsel kanıt). Yeni dosyalar yerelde olmadığı için doğrudan tensör karşılaştırması yapılamadı. |
| Aynı test kümesi mi kullanılmış? | **Evet.** Aynı 367 kimlik, aynı sıra ve aynı etiketler; manifest hash'i aynı. | Kesin. Tek çekince: eski koşuda eğitim sırasında yazılmış bir kimlik logu yok. |
| Aynı füzyon yöntemi mi kullanılmış? | **Hayır.** 0.9186 kapılı residual yoldan, 0.9025 eşit ağırlıklı çıkarım yolundan gelir. Aynı yollar karşılaştırıldığında da açık sürer (kapılı −0.019, eşit −0.015). Bu nedenle füzyon farkı açığın nedeni değildir. | Kesin |
| Metrikler aynı yöntemle mi hesaplanmış? | **Evet.** Argmax, `cohen_kappa_score(weights='quadratic')`, 5 sınıf, n=367, üç seed ortalaması, örneklem SD'si (ddof=1). | Kesin |
| Farkın nedeni kanıtlandı mı? | **Kısmen.** Farkın *yeniden eğitilmiş ağırlıklardan* kaynaklandığı ve özellikle *öğretmen yolunun* yaklaşık 0.020 düşmesiyle izlendiği kanıtlandı. Test kümesi, etiket, metrik, split ve yuvarlama kaynaklı nedenler dışlandı. **Yeni öğretmenlerin neden daha düşük test başarısı verdiği ise kanıtlanamadı** (bkz. H). | Bkz. aşağıdaki sınıflandırma |
| Hangi ek dosyalar gerekiyor? | Bkz. bölüm 3 | — |
| Makalede ne kesinleştirilmemeli? | Bkz. bölüm 4 | — |

## 1. Ana sayısal bulgu

| Yol (aynı 367 görüntü, 3 seed) | Eski | Yeni | Eski − yeni |
|---|---|---|---|
| Kapılı residual | 0.9186 | 0.8996 | +0.0190 |
| Equal T+S | 0.9174 | 0.9025 | +0.0149 |
| Yalnızca öğretmen | 0.9141 | 0.8944 | +0.0197 |
| Seçim için kullanılan doğrulama QWK'si (kapılı) | 0.8908 | 0.8900 | +0.0008 |

- Sistem düzeyindeki düşüş, öğretmen yolundaki düşüşle neredeyse aynı büyüklüktedir. Her iki deneyde de tam sistem öğretmen yolundan yalnızca 0.003–0.008 yukarıdadır.
- Doğrulama QWK'leri neredeyse aynıdır. Fark **yalnızca test kümesinde** belirgindir.
- Eşleştirilmiş görüntü bootstrap'ında %95 aralıkların hepsi sıfırı içerir (örneğin kapılı − equal: +0.0161 [−0.0117, +0.0451]).

## 2. Olası nedenlerin sınıflandırılması

### A. Farklı model veya checkpoint

- **Destekleyen:**
  - Hash'ler farklı.
  - Yeni dosyalarda öğretmen parametreleri ayrı tutuluyor.
  - Aynı görüntülerde 35–46 kapılı-yol ve 39–51 öğretmen-yol tahmin farkı var.
  - Yeni öğretmenler seed başına eğitilmiş. Eski deneyde tek bir öğretmen çifti vardı ve gömülü öğretmenler yalnızca BatchNorm ve spektral tamponlarda farklıydı.
  - Öğretmen yolu farkı (0.0197), sistem farkının (0.0190) neredeyse tamamına karşılık geliyor.
- **Çelişen:** Yok.
- **Güven:** Yüksek. Farkın taşıyıcısı farklı eğitilmiş ağırlıklardır, en belirgin olarak da öğretmen kalitesidir.
- **Eksik:** Yeni öğretmen checkpoint hash'leri ve dosyaları; eski ve yeni ağırlıkların doğrudan karşılaştırması.

### B. Farklı veri bölünmesi

- **Destekleyen:** Yok.
- **Çelişen:** Aynı manifest hash'i, aynı test kimlik kümesi ve sırası. Train sayıları ve deterministik bölünme kodu aynı.
- **Güven:** Bu kategorinin neden olma olasılığı **düşük** (dışlandı).
- **Eksik:** Eski koşunun eğitim sırasında yazılmış split logu.

### C. Farklı test örnekleri veya etiketler

- **Destekleyen:** Yok.
- **Çelişen:** Kimlik, sıra ve etiket düzeyinde birebir eşitlik; sınıf dağılımı aynı.
- **Güven:** Dışlandı (yüksek güven).
- **Eksik:** Yok.

### D. Farklı füzyon yöntemi

- **Destekleyen:** Raporlanan iki sayı gerçekten farklı yollardan geliyor (kapılı ve eşit).
- **Çelişen:** Aynı yollar karşılaştırıldığında da açık 0.015–0.019. Eski deneyde equal ile kapılı arasındaki fark yalnızca −0.0012, yeni deneyde +0.0029. Füzyon farkı açığı büyütmüyor, yaklaşık 0.003 daraltıyor.
- **Güven:** Raporlama tutarsızlığı olarak **yüksek**; farkın nedeni olarak **düşük**.
- **Eksik:** Yok.

### E. Farklı değerlendirme kodu veya metrik hesabı

- **Destekleyen:** Kod dosyaları farklı (notebook ile `analyze.py`/`dr_common.py`).
- **Çelişen:** Aynı formül ve parametreler kullanılıyor. Bu denetimde iki sonuç aynı kodla yeniden hesaplandı ve raporlananlarla birebir eşleşti. SD türü ikisinde de ddof=1.
- **Güven:** Dışlandı (yüksek güven).
- **Eksik:** Yok.

### F. Farklı seed veya eğitim koşulları

- **Destekleyen:**
  - Eski öğretmen eğitiminde seed ayarı yok. Yeni öğretmenler `set_seed(s)` ile ve s ∈ {42, 52, 62} için yeniden eğitilmiş.
  - Ortamlar farklı: eski koşu sunucuda (RTX A5000 olduğu önceki kayıtlarda geçiyor, sürümler kayıtsız); yeni koşu Colab A100'de, torch 2.11.
  - En iyi epoch'lar farklı: eski 10/4/11, yeni 30/7/4.
  - Yeni sınıflar `dr_pipeline.py` içinde yeniden yazılmış.
- **Çelişen:**
  - Hiperparametreler aynı: öğretmenler 110/15/1e-3 ve 120/18/1e-4; öğrenci 70/10/2e-4; batch 16; AdamW 1e-4; CosineWarmRestarts 10/2; residual 0.70; frenler 0.82/0.66; kayıp katsayıları 0.8→0.25, 0.010→0.003, distillation 0.30, ordinal 0.12.
  - Doğrulama QWK'leri neredeyse aynı (0.8908 ve 0.8900).
- **Güven:** Orta. Farkın yeniden eğitim varyansıyla ortaya çıktığı tutarlı görünür; ancak hangi koşulun belirleyici olduğu ayrıştırılamaz.
- **Eksik:** Eski seed ve ortam kayıtları; eski öğretmenin farklı seed'lerle yeniden eğitildiği bir kontrol.

### G. Kayıt, yuvarlama veya raporlama hatası

- **Destekleyen:** Yok.
- **Çelişen:** 0.918582 ± 0.000993 → 0.9186 ± 0.0010 ve 0.854678 ± 0.003146 → %85.47 ± %0.31 doğru yuvarlanmış. 0.9025 ± 0.0066 ve %84.83 ± %1.03 de doğru.
- **Güven:** Sayısal hata olarak dışlandı. Ancak bir **yöntem raporlama tutarsızlığı** vardır: orijinal makale eşit ağırlıklı füzyon anlatırken 0.9186'yı kapılı yol üretmiştir (bkz. D).
- **Eksik:** Yok.

### H. Mevcut kanıtlarla belirlenemeyen neden

- **Belirlenemeyen:** Eski öğretmen çiftinin testte daha iyi (yaklaşık 0.914) ve yeni öğretmenlerin daha düşük (yaklaşık 0.894) olmasının nedeni. Mevcut kanıtlarla birbirinden ayrılamayan açıklamalar şunlardır:
  1. **Teste dayalı seçim yanlılığı:** Eski koşuda dört T2 varyantı test QWK'sine göre sıralandı (`train_improved_anchored_hybrids.ipynb` satır 670). Öğrenci mimarisi de test QWK ortalamasına göre seçildi (`summarize`, finalistler, `best_row`). Tek bir öğretmen çifti seçilip üç seed'de yeniden kullanıldığı için olumlu bir test sapması üç seed'e birlikte aktarılmış olabilir. Doğrulama skorlarının iki deneyde eşit, test skorlarının ise eski deneyde yüksek olması bu açıklamayla tutarlıdır, ancak bunu **kanıtlamaz**.
  2. **Tek öğretmen eğitiminin rastgele varyansı:** Öğretmen eğitiminin seed'i kayıtsızdır ve testte ±0.03 genişliğinde bir örnekleme belirsizliği vardır.
  3. **Ortam veya yazılım farkları** ve yeniden yazılmış kodun sayısal davranışı.
  4. **Gömülü öğretmenin BatchNorm tamponlarının öğrenci eğitimi sırasında güncellenmesi.** Ayrı kaydedilmiş öğretmen 0.9068, gömülü öğretmen yaklaşık 0.914 kaydetmiştir. Bu fark yeniden çalıştırılarak doğrulanmadı.
- **Güven:** Belirlenemedi.

## 3. Gereken ek dosyalar ve doğrulamalar

1. Sunucudaki eski checkpoint'lerin orijinal SHA-256 değerleri (`/mnt/veriler/Ece_DR/results/lowdata_hybrid_distill/...`), yerel kopyalarla eşleştirmek için.
2. Eski öğretmen eğitiminin çıktı logu veya `comparison_df` tablosu (dört T2 varyantının test ve doğrulama skorları); ortam ve sürüm bilgisi.
3. Yeni deneyin Drive'daki `teacher1/teacher2/student` `best.pth`, `complete.json` ve `history` dosyaları; özellikle yeni öğretmenlerin hash'leri ve doğrulama eğrileri.
4. Seçim yanlılığı açıklamasını test etmek için (yeni eğitim gerektirir; bu denetimin kapsamı dışındadır):
   - Eski öğretmen çiftini sabit tutup yalnızca öğrenciyi yeni kodla eğitmek, ya da
   - Yeni öğretmenleri eski checkpoint'lerle değiştirerek yalnızca çıkarım yapmak (eğitim gerektirmez; anahtar uyumluluğu önce test edilmelidir).
5. Ayrı kaydedilmiş eski öğretmenin CPU üzerinde yeniden değerlendirilmesi (0.9068 ile gömülü 0.914 arasındaki farkı doğrulamak için).

## 4. Makalede kesinleştirilmemesi gerekenler

- **0.9186 ± 0.0010 / %85.47 ana sonuç olarak kullanılmamalı.** Bu değer, test kümesine dayalı model ve öğretmen seçiminden geçmiş tarihsel bir checkpoint'in kapılı yolundan gelir ve yeniden eğitimde tekrarlanmamıştır.
- 0.9186, makalenin anlattığı **eşit ağırlıklı füzyonun sonucu olarak sunulmamalı.** Aynı ağırlıklarla anlatılan yöntem 0.9083, T+S equal yolu ise 0.9174 verir.
- 0.9186 ile 0.9025 arasındaki fark **yöntem değişikliğinin etkisi** olarak yorumlanmamalı. Aynı yollar karşılaştırıldığında da açık sürer.
- Farkın nedeni "teste dayalı seçim yanlılığı" olarak **kesin dille yazılmamalı**. Bu bir hipotezdir.
- Fark **istatistiksel olarak anlamlı bir düşüş** olarak da sunulmamalı. Bootstrap aralıkları sıfırı içerir.
- Revizyondaki 0.9025 ± 0.0066 değeri yeniden eğitimin doğru ve yeniden üretilmiş özetidir. Ancak bu değer de tarihsel olarak teste maruz kalmış aynı test kümesine aittir; bu sınırlılık belirtilmelidir.
- 0.9025 sonucunun **ayrı eğitilmiş bir eşit-füzyon modelinden değil**, kapılı eğitimle elde edilmiş ağırlıkların eşit çıkarımından geldiği açıkça yazılmalıdır.
- Eski sonuç metinde kalacaksa, "tarihsel, test-seçimli, yeniden üretilmemiş kapılı checkpoint sonucu" olarak etiketlenmelidir.

## 5. Kesin ve doğrulanmamış bulgular

**Kesin:**

- İki sayının kaynağı belirlendi.
- İki sayı da görüntü bazlı çıktılardan birebir yeniden üretildi.
- Test kümesi, etiketler ve sıra aynı.
- Metrik kodu ve SD türü aynı.
- Raporlanan füzyon yolları farklı, fakat bu açığı açıklamıyor.
- Modeller, öğretmenler dahil, farklı eğitilmiş ağırlıklardır.
- Açık, öğretmen yolundaki düşüşle aynı büyüklüktedir.

**Doğrulanmamış:**

- Yeni öğretmenlerin neden daha zayıf olduğu.
- Sunucu orijinal hash'leri.
- Yeni checkpoint dosyalarının içeriği.
- Eski eğitim ortamı ve seed bilgileri.
- Ayrı kaydedilmiş öğretmen ile gömülü öğretmen arasındaki farkın nedeni.
