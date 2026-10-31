# Checkpoint ve model kimliği karşılaştırması

## 1. Eski checkpoint'ler (0.9186)

| Dosya (yerel kopya) | Bayt | SHA-256 (24.10 yeniden hesap) | 16.10 `provenance.json` ile aynı mı? |
|---|---|---|---|
| `D:\m1\Documents\seed 42\best_model.pth` | 568,566,788 | `d3327f355a51ac810431094f2e79505f1d43e31f13988ea1abd78caa65bd4eee` | Evet |
| `D:\m1\Documents\seed 52\best_model.pth` | 568,566,788 | `c3a056f1d1557121d83fed26c5527eae86feab10fd4415b79d3eb03252aada43` | Evet |
| `D:\m1\Documents\seed 62\best_model.pth` | 568,566,788 | `8027a8aa6a39e298f9189ed4c2f9c591f95b24abb46524adfd2476e5ced8e2b7` | Evet |
| `seed 42/52/62\checkpoint_current.pth` | 791,335,203 / 791,334,627 / 791,335,267 | `b9cd1bff…` / `8c912523…` / `6c84f72e…` | 16.10'da kaydedilmemişti |
| `D:\m1\Documents\best_model.pth` (HeavyTeacherEnsemble) | 457,175,114 | `deec7b98…` | — |
| T1 `...Baseline_HybridCNNViT_Gate_Improved\best_model.pth` | 19,470,252 | `b986387d…` | — |

Tam hash listesi `archive_file_hashes.json` dosyasındadır.

### Yapı ve kimlik

- **Anahtarlar ve tensör şekilleri:** 16 Ekim doğrulaması üç `best_model.pth` dosyasını `weights_only=True` ile CPU'ya yükledi ve notebook'tan çıkarılan sınıf tanımlarına `strict=True` ile bağladı. Eksik, fazla veya şekli uyumsuz anahtar olsaydı yükleme hata verirdi. Toplam parametre sayısı 141,999,859 olarak bulundu ve kayıtlı `param_total` ile eşleşti. Bu denetimde büyük dosyalar gereksiz yere yeniden yüklenmedi; yalnızca hash'ler hesaplandı.
- **Nihai checkpoint mi?** `best_model.pth`, doğrulama QWK'sine göre seçilen epoch'tur (seed 42 için epoch 10/20, seed 52 için 4/14, seed 62 için 11/21). `checkpoint_current.pth` ise son epoch'a ait devam dosyasıdır. Makaledeki sayıyı birebir üreten dosya `best_model.pth`'dir: üç confusion matrix aynıdır.
- **Hangi deneyden geldiği:** `train_lowdata_hybrid_distill.ipynb` final aşamasından gelir. Kimlik dosya adından değil, confusion matrix ve parametre sayısının birebir eşleşmesinden çıkarıldı.
- **Seed'ler arasında öğretmen:** `revision_additional_analyses.json` → `teacher_state_comparison`. Her seed checkpoint'inin içindeki öğretmen, ayrı kaydedilmiş öğretmen checkpoint'iyle karşılaştırıldı. Farkların tamamı BatchNorm veya spektral normalizasyon tamponlarındadır: `running_mean`, `running_var`, `num_batches_tracked`, `classifier.weight_u` ve `weight_v`. Öğrenilebilir öğretmen ağırlıkları değişmemiştir.
  - **Sonuç:** Eski üç seed **aynı, bir kez eğitilmiş öğretmen çiftini** kullanmıştır. Seed yalnızca öğrenci ve kapı eğitimini değiştirmiştir.
- **Gömülü öğretmenin performansı:** Öğretmen yolu (`archived_teacher`) QWK 0.9158 / 0.9122 / 0.9141 verir. Ayrı kaydedilmiş öğretmen dosyası `Improved_Anchored_SpectralNorm/test_metrics.json` ise QWK 0.9068 kaydetmiştir.
  - Tamponlar öğrenci eğitimi sırasında güncellendiği için gömülü öğretmenin test çıktısı değişmiş olabilir.
  - Ayrı öğretmen checkpoint'i bu denetimde yeniden çalıştırılmadı. Bu nedenle 0.9068 ile 0.914 arasındaki farkın nedeni **doğrulanmadı**.

## 2. Yeni checkpoint'ler (0.9025)

| Seed | Öğrenci `best.pth` SHA-256 | Kaynak |
|---|---|---|
| 42 | `67252f6bf3b52a6b18f4d1c2f216df48be0f2463d95d4a1c7ff098f95fad75d0` | `diagnostic_report.json` |
| 52 | `1d7db802299314a49ca884f4bf9a71171adab942bcf212c8d4446ffb8ed76e0e` | aynı |
| 62 | `16789cbbbb7f6ed941aa9f327cff47a3c69a9bbe172051357c3d0b22feca3f22` | aynı |

- Dosyalar yerelde yok. Boyutları, anahtarları ve şekilleri bu denetimde **doğrulanamadı**.
- `run.py` `load_best`, yüklemeden önce hash'i `complete.json` ile karşılaştırır. `restore_state` ise eksik veya fazla anahtar olduğunda hata verir. Tanı çıkarımı bu yolla yapılmıştır.
- Öğrenci checkpoint'i öğretmenin öğrenilebilir parametrelerini içermez (`model_state`). Öğretmen, aynı seed'in `teacher2/best.pth` dosyasından yüklenir. Bu öğretmen dosyalarının hash'i yerelde bulunamadı.
- Yeni öğretmenler **her seed için ayrı eğitilmiştir.**

## 3. Ağırlıklar aynı mı?

Doğrudan tensör karşılaştırması **yapılamadı**, çünkü yeni checkpoint'ler yerelde değil. Buna karşılık şu işlevsel kanıtlar ağırlıkların farklı olduğunu kesin olarak gösterir:

- Aynı 367 görüntüde eski ve yeni kapılı yollar seed başına 35, 40 ve 46 görüntüde farklı sınıf tahmin eder.
- Eski ve yeni öğretmen yolları seed başına 39, 40 ve 51 görüntüde farklı tahmin eder.
- Yeni öğrenci hash'leri ile eski hash'ler farklıdır. Yeni dosya biçimi de farklıdır: öğretmen parametreleri ayrılmıştır.

Aynı girdiye farklı çıktı veren iki model aynı ağırlıklara sahip olamaz. **Sonuç: Model aynı değildir. Hem öğrenci hem öğretmen yeniden eğitilmiştir.**

## 4. Mimari uyumu

Kaynak kodda eski ve yeni sınıflar aynı ayarları kullanır: `ResidualTeacherStudentModel`, `residual_scale=0.70`, conflict brake 0.82/0.66, ConvNeXt Tiny öğrenci, T2 için `Adv_Idea_5_SpectralNorm`. Yeni sınıflar `dr_pipeline.py` içinde yeniden yazılmıştır.

Eski ve yeni sınıfların ağırlık anahtarlarının birebir uyumu bu denetimde test edilmedi. Yani eski checkpoint yeni sınıfa yüklenmedi.
