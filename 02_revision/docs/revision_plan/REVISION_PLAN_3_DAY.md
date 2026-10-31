# Hakem Revizyonu: 18 Ekim Teslim Planı

**Plan tarihi:** 5 Ekim 2026  
**Son teslim:** 18 Ekim 2026  
**Takvim:** 13 gün (5 Ekim başlangıcından 18 Ekim son teslimine kadar)  
**Durum:** Plan güncellendi; yeni eğitimler başlatılmadı.  
**İlke:** Hakemlerin istediği ek deneyler revizyonun zorunlu kapsamıdır; isteğe bağlı, yalnızca vakit kalırsa yapılacak işler değildir. Veri, checkpoint, izin veya GPU gereksinimleri ilk aşamada çözülecek; eğitim ve analizler mümkün olan en erken tarihte başlatılacak, metin ve yanıt mektubu deneylerle paralel yürütülecektir. Bir erişim engeli çıkarsa varsayılan çözüm deneyi kapsamdan çıkarmak değil, engeli gidermek, kaynak/işbirliği bulmak ve gerekiyorsa editörden süre uzatımı istemektir. Bilimsel güvenilirlik son tarihten önce gelir.

## Yönetici Özeti

13 günlük süre sıkışıktır; yine de aşağıdaki hakem maddelerinin tümü planlanan revizyon kapsamındadır ve ek deneyler yapılacaktır. Dış veri kümesi doğrulaması, çoklu-seed kontrollü ablation'lar, preprocessing/augmentation karşılaştırmaları, kalibrasyon, Grad-CAM ve tam sistem latency ölçümleri veri, checkpoint, izin ve hesaplama erişimine bağlı kritik işlerdir. Depoda APTOS görüntü/veri dosyaları, final öğretmen checkpoint'leri ve eşit-ağırlık/gated füzyon için görüntü başına tahmin dosyaları bulunmadığı için bunlar ilk günlerde temin edilecek veya ilgili yazarlardan/kurumlardan istenecektir. Takvimdeki ara karar kapıları deneyleri kapsamdan çıkarmak için değil; kaynak, paralel çalışma veya süre uzatımı gereksinimini erken belirlemek içindir. Yalnızca çözümü için tüm makul erişim ve süre seçenekleri tüketilen gerçek bir engel kalırsa, bu engel editöre/hakemlere şeffafça bildirilecek; doğrulanmamış deney sonucu hiçbir koşulda yazılmayacaktır.

Önerilen yaklaşım:

1. 6 Ekim sonuna kadar makale kaynağını, kritik sonuç artifacts'larını ve veri/checkpoint/GPU erişimini doğrula.
2. En geç 7 Ekim'de protokolü kilitle; erişim sağlanmışsa uzun eğitimleri başlat ve GPU kuyruğunu önceliğe göre planla.
3. Eğitim sürerken mevcut sonuçların yeniden hesabını, latency/class-wise analizlerini, metin düzeltmelerini ve yanıt mektubunu paralel yürüt.
4. 12 Ekim'de ara karar kapısı uygula: geciken deneyler için ek kaynak, paralel yürütme ve/veya editörden süre uzatımı kararı al; deneyleri sessizce kapsamdan çıkarma.
5. 15–17 Ekim'i yazım, LaTeX derleme, kaynak-sonuç çapraz kontrolü ve son teslim tamponu olarak koru.

## Öncelik Sırası

| Öncelik | Hakem konusu | Gerekli çıktı | Tür / tahmini süre | Engelleyici bağımlılık |
|---|---|---|---|---|
| P0 | Eşit-ağırlık ve gated QWK uyuşmazlığı; aynı tahmin iddiası | Eşleştirilmiş sınıf tahminleriyle etiket eşitliğini ve QWK'yi yeniden hesapla; Tablo 2 ve metni düzelt | Adli kontrol: 2-6 saat; tahmin üretimi gerekirse ek süre | İki model için aynı görüntü sırasındaki tahminler ve doğru ground-truth |
| P0 | Test setinin model ailesi seçiminde kullanılması | Gerçek seçim akışını doğru açıkla; test-bağımsızlık iddiasını kaldır; varsa bağımsız değerlendirme planını belirle | Kod/sonuç denetimi: 1-3 saat; bağımsız yeni test değerlendirmesi daha uzun | Kaydedilmiş notebook ve dış/önceden dokunulmamış test verisi |
| P0 | Raporlanan sonuçların ve deney kaynağının tutarlılığı | Her tablo/şekli kaynak CSV/JSON/checkpoint ve kullanılan inference modu ile eşleştir | 2-5 saat | İlgili artifacts; bazıları mevcut değil |
| P1 | Öğrenci-only ile tam sistem performansı | Aynı donanım/ölçüm protokolüyle ayrı metrik, parametre, latency ve mümkünse bellek raporla | Ölçüm 2-6 saat; model yükleme/bağımlılıklara bağlı | Teacher ve student checkpoint'leri, test/inference kodu, GPU |
| P1 | Teacher 1 katkısı | Teacher-2-only ile iki öğretmenli distillation karşılaştırması; disagreement/error overlap raporu | Eğitim: saatler-günler | Öğretmen checkpoint'leri, veri, GPU, sabit protokol |
| P1 | Kontrollü loss ablation | Aynı ConvNeXt Tiny ile CE, tek/çok sıcaklıklı KD, ordinal, non-inferiority ve tam loss karşılaştırması | En maliyetli deney grubu: çoklu seed ile günler+ | Veriler, öğretmenler, doğru deney kodu, GPU süresi |
| P1 | Sınıf-bazlı performans ve klinik hata | Per-class precision/recall/specificity/F1, support, referable-DR ölçüleri ve güven aralıkları | Mevcut tahminlerle 1-3 saat; tahmin yoksa inference eklenir | Tahminler/etiketler veya checkpoint ve veri |
| P1 | Logit/probability averaging ve kalibrasyon | Raw-logit, probability ve sıcaklık kalibrasyon karşılaştırması; ECE/Brier/NLL | 2-8 saat + gerekirse eğitim/kalibrasyon | Aynı örnekler için teacher logits/probabilities |
| P1 | Augmentation, düşük veri ve ImageNet normalization | Aynı backbone ve sabit protokolle augmentation, %10/%25/%50/%100 veri oranı ve giriş normalizasyonu karşılaştırmaları | Eğitim: saatler-günler; koşullar GPU kapasitesine göre kuyruklanır | Veri, GPU ve kilitli protokol |
| P1 | Non-inferiority terimi | Formül/marj/gradyan akışını koddaki uygulamayla eşleştir; adlandırmayı düzelt; ablation sonucu | Kod/metin: 1-3 saat; ablation ek süre | Notebook/source implementation |
| P1 | Dış veri kümesi/klinik geçerlilik | Farklı merkez/hastane kaynağından bağımsız harici doğrulama; veri erişimini temin et ve testte tuning yapma | En az günler; izin/etiket uyumu dahil | Lisanslı/izinli harici veri ve uyumlu preprocessing |
| P1 | Grad-CAM/açıklanabilirlik | Doğru ve hatalı örnekleri içeren, tanımlı inference modu için Grad-CAM görselleri ve sınırlı yorum | Checkpoint/veri hazırsa 2-8 saat | Çalışan model ve görüntüler |
| P2 | Tekrar üretilebilirlik, referanslar, biçim ve beyanlar | Eğitim ayrıntıları, kaynakça kontrolü, AI beyanı, şekil/tablo biçimi ve yanıt mektubu | 4-10 saat | Kaynak ve gerçek deney bilgileri |

## Hakem Yorumları İçin Eksiksiz Revizyon Kontrol Listesi

Bu bölüm hakem raporlarındaki her talebi ayrı bir takip maddesine dönüştürür; listelenen ek deneylerin tamamı planın taahhüt edilen kapsamındadır. Her madde teslimden önce **tamamlandı (kanıt/konum belirtilerek)** olarak kapatılmalıdır. Bir maddeye veri/izin/checkpoint/hesaplama engeli çıkarsa önce erişim sağlama, işbirliği/kaynak bulma, işleri paralelleştirme ve gerektiğinde süre uzatımı isteme adımları izlenecektir; engel giderilemediği istisnai durumda kısmi/yapılamadı durumu ve gerekçesi şeffafça kaydedilir. Bir deney yapılmadıysa sonuç varmış gibi yazılmayacaktır.

### Hakem 1

- [ ] **R1.1 — QWK ve aynı tahmin iddiasındaki çelişki:** Equal-weight ve entropy-gated/Gate2 için aynı 367 görüntünün aynı sıradaki ground-truth ve sınıf tahminlerini temin et. Örnek kimliklerini/sırasını doğrula; tahmin eşleşme sayısı ve oranını, her iki QWK'yi, accuracy/F1'i ve confusion matrix'i aynı metrik uygulamasıyla yeniden hesapla. QWK güven aralığı/paired bootstrap hesabının kaynağını da denetle. Tahmin dosyaları mevcut değilse checkpoint'lerden aynı split için tahminleri yeniden üret; eşit tahmin iddiasını ancak yeniden hesapla doğrulanırsa koru.
- [ ] **R1.2 — Test setinin model seçimine etkisi:** Notebook/kod, sonuç tabloları ve yazışma kayıtlarından mimari, loss ağırlığı, fusion ve hiperparametre seçim akışını yeniden kur. Test QWK'si model sıralama/seçiminde kullanıldıysa bunu açıkça bildir; testin hiçbir geliştirme kararında kullanılmadığı ifadesini kaldır. Validation ile checkpoint seçimi ve testle son değerlendirme arasındaki farkı doğru anlat; geçmişte test üzerinde karşılaştırma yapıldıysa aynı split'i sonradan bağımsız test diye adlandırma. Önceden dokunulmamış harici veriyle bağımsız değerlendirme R2.1 kapsamında yürütülecek.
- [ ] **R1.3 — Student-only ve tam sistem sonuçlarının ayrılması:** ConvNeXt Tiny öğrenci tek başına ve öğretmen(ler)+öğrenci nihai inference sistemi için ayrı sonuç satırları hazırla. Her ikisi için accuracy, macro-F1, QWK, AUC, parametre sayısı, peak memory ve latency raporla. Tablo, metin, denklem ve Figure 2 caption'ında inference modunu açıkça adlandır; öğrenci latency'sini tam sisteme atfetme.
- [ ] **R1.4 — Kontrollü bileşen ablation'ı:** Aynı ConvNeXt Tiny backbone, veri bölünmesi, eğitim reçetesi ve aynı önceden belirlenmiş en az üç seed altında şu koşulları karşılaştır: (a) weighted CE, (b) CE + single-temperature KD, (c) CE + multi-temperature KD, (d) CE + KD + ordinal CDF loss, (e) CE + KD + non-inferiority loss, (f) tam objective (CE + KD + ordinal + non-inferiority). Her bileşenin tek başına ve birlikte katkısını raporla; bütün koşullar aynı checkpoint-selection kuralını kullansın.
- [ ] **R1.5 — Standalone modeller karşısındaki üstünlük iddiası:** Tablo 1'deki Fusion CNN–ViT QWK=0.923 ile önerilen sistem için bildirilen ortalama QWK=0.9186'yı aynı protokol ve inference modu bakımından karşılaştır. Üstünlük iddiasını bu sayılar desteklemiyorsa Abstract/Results/Discussion/Conclusion'da “consistently improves/superior” gibi ifadeleri düzelt. Üstünlük savunulacaksa aynı split, seed sayısı ve değerlendirme protokolüyle adil, çoklu-seed karşılaştırma ve belirsizlik ölçüsü sağla.
- [ ] **R1.6 — Teacher 1'in faydası:** Teacher-2-only distillation ile Teacher-1+Teacher-2 distillation'ı aynı öğrenci ve eğitim protokolünde karşılaştır. Teacher 1'in tek başına QWK=0.640, Teacher 2'nin QWK=0.901 ve equal-weight birleşimin bildirilen küçük artışı (0.001) bağlamında katkıyı sayısal olarak açıkla. Mümkünse disagreement rate, hata örtüşmesi/error correlation, KL/diversity ölçüsü ve test/validation performansını raporla; katkı göstermiyorsa Teacher 1'in gerekliliğini iddia etme ve mimari/öğretmen seçimini tartış.
- [ ] **R1.7 — Ham logit ortalamasının gerekçesi:** Öğretmenlerin logit ölçeklerini ve çıktı dağılımlarını incele; aynı örneklerde (a) raw-logit averaging, (b) probability averaging ve (c) validation'da belirlenmiş sıcaklıkla kalibre edilmiş logit/probability averaging'i karşılaştır. QWK ve uygun sınıflandırma metriklerini raporla; sıcaklığı test setinde ayarlama. Ölçek uyumluluğunu sayısal bulgularla değerlendir.
- [ ] **R1.8 — Kalibrasyon iddiaları:** Öğretmen ve ilgili nihai modeller için aynı değerlendirme örnekleri üzerinde ECE, Brier score ve NLL hesapla; reliability diagram ekle. Spectral normalization'ın kalibrasyonu iyileştirdiği iddiasını uygun kontrol/ablation ile doğrula. Confidence değerlerini tek başına calibration kanıtı gibi sunma.
- [ ] **R1.9 — Pretrained model preprocessing'i:** ImageNet-pretrained ResNet50, ViT-B/16 ve ConvNeXt Tiny için kullanılan giriş ölçekleme/normalizasyonunu model bazında açıkla. Aynı sabit protokol altında raw [0,1] girdi, ImageNet mean/std normalization ve uygun retinal preprocessing (kullanılacaksa yöntemi sabitleyerek) karşılaştır. Test setinde preprocessing seçimi yapma. Deney yetişmezse tercihin gerekçesini ve bunun sınırlamasını Methods/Limitations'ta bildir.
- [ ] **R1.10 — Low-data iddiası:** Train verisinin stratified %10, %25, %50 ve %100 alt kümeleriyle (en azından uygulanabilir alt kümeler) kontrollü eğitim/evaluation yap; alt küme seed'lerini ve sınıf oranlarını sabitle, validation/test protokolünü değiştirme. Sonuçları veri miktarına göre tablo/şekille göster. Bu deney yapılamazsa “low-data” ifadesini yalnızca veri kümesinin ölçeğini tanımlayacak şekilde yumuşat; az veriyle üstünlük/sağlamlık iddiası kurma.
- [ ] **R1.11 — Non-inferiority loss tanımı:** Denklem (5)'i gerçek kod ve gradient akışıyla karşılaştır. Margin işareti, hinge yönü, teacher loss'un detach edilip edilmediği ve optimize edilen büyüklükleri açıkla. Formül geleneksel non-inferiority değil student-superiority margin ifade ediyorsa terimi düzelt veya terminolojiyi kaynakla gerekçelendir. R1.4'teki ablation'da bu terimin katkısını ayrıca göster.
- [ ] **R1.12 — Yeniden üretilebilirlik ayrıntıları:** Teacher 1, Teacher 2, gated/fusion model, ConvNeXt Tiny öğrenci ve standalone baseline'ların her biri için optimizer, learning rate/scheduler, batch size, epoch sınırı/early stopping, initialization/pretrained ağırlık, frozen/unfrozen katmanlar, checkpoint seçme kuralı, loss ağırlıkları/sıcaklıklar, seed ve split'i Methods veya ek tabloda ver. Yazılım/kütüphane sürümleri, donanım/GPU ve kullanılan precision bilgisini de kaydet; bilinmeyen ayarları tahmin ederek doldurma.
- [ ] **R1.13 — Hesaplama verimliliği ve parametre hesabı:** Parametre sayısını model bileşenlerinden yeniden say. Öğretmenlerin 113.8M ve 27.8M, öğrencinin 8.2M olduğu doğrulanırsa tam sistem toplamını yaklaşık 149.8M olarak düzelt; kullanılan inference'da aktif bileşenleri ayrıca belirt. Student-only ve full-system için aynı donanım, batch size, precision, warm-up sayısı, tekrar sayısı ve CUDA synchronization ile latency ölç; ölçüm birimini (ms/image ve throughput) ve bellek ölçüm yöntemini açıkla.
- [ ] **R1.14 — İstatistiksel belirsizlik:** Ana karşılaştırmalara güven aralıkları ve paired bootstrap testi ekle. Bootstrap sampling unit'i (hasta/görüntü; veri yapısına uygun), resample sayısını, stratification/paired eşleştirme yöntemini, random seed'i ve CI yöntemini belirt. Aynı test örneklerinde karşılaştırma yap; küçük örnek/çoklu seed sınırlılığını Discussion'da açıkla.
- [ ] **R1.15 — Klinik olarak anlamlı sınıf-bazlı analiz:** Her DR sınıfı için precision, recall, specificity, F1 ve support/sınıf örnek sayısını raporla; özellikle severe sınıfında bildirilen 5/19 doğru tahmini saklamadan tartış. Önceden tanımlanmış referable-DR eşik/sınıf birleştirmesi için sensitivity ve specificity ile güven aralıklarını ekle. Tanım ve hesaplama Methods'ta açıkça belirtilsin.
- [ ] **R1.16 — Figure 2'nin inference modu:** Mevcut confusion matrix'in student-only mi, yoksa öğretmen–öğrenci final logit averaging mi olduğunu doğrula ve caption'da açıkça yaz. Mümkünse iki inference modu için ayrı confusion matrix üret; sınıf sırası, support ve normalize edilip edilmediğini belirt. Kaynağı belirsiz şekli final model sonucu gibi kullanma.
- [ ] **R1.17 — Yayınlarla karşılaştırma:** İlgili çalışmaları kaynaklarından doğrula; karşılaştırma tablosunda mümkün olan her çalışma için veri kümesi/split, augmentation, validation/CV protokolü, kappa türü (QWK olup olmadığı), single-run veya çoklu-run/CV durumu ve raporlanan metrik bilgisini ekle. QWK olduğu doğrulanmayan kappa değerlerini QWK ile doğrudan karşılaştırma; protokol farklarını metinde belirt.
- [ ] **R1.18 — Kaynakça doğruluğu ve iddiaların dayanağı:** Her kaynağın bibliyografik bilgilerini (yazar, başlık, yıl, dergi/konferans, cilt/sayı, sayfa, DOI/URL ve gerekiyorsa erişim tarihi/yayın durumu) doğrula. Her atfın cümledeki iddiayı gerçekten desteklediğini, özellikle calibration, ordinal learning ve multi-teacher diversity iddialarını kontrol et; alakasız/yanlış atıfları düzelt.
- [ ] **R1.19 — Klinik kullanım/deployment iddiaları:** Tek APTOS 2019 değerlendirmesi; harici doğrulama, prospektif test, demografik adalet/fairness ve cihazlar arası değerlendirme yoksa screening deployment, regulated workflow ve on-device kullanımını kanıtlanmış kabiliyet gibi sunma. Bu ifadeleri olası gelecek uygulama olarak çerçevele; dış geçerlilik ve klinik çalışma gereksinimini Limitations'ta belirt.
- [ ] **R1.20 — AI-use beyanı:** “AI kullanılmadı” ifadesiyle Claude'un dilbilgisi/dil düzeltmede kullanıldığı ifadesi arasındaki çelişkiyi gider. Bilimsel içerik, deney analizi veya sonuç üretiminde generative AI kullanılmadıysa bunu açıkça; Claude'un yalnızca dil/grammar düzeltmede kullanımını derginin disclosure politikasına uygun biçimde beyan et. Kullanım kapsamını gerçeğe aykırı biçimde genişletme/daraltma.
- [ ] **R1.21 — Sunum ve biçim:** Table 5'i computational-efficiency alt bölümünün uygun yerine taşı; başlık/line number sıkışmalarını ve denklem tipografisini düzelt. Metrik gösterimi, ondalık hassasiyet, mean ± SD biçimi ve single-run ile multi-seed ayrımını tutarlı yap. Tüm tablo/şekil numaraları ve metin atıflarını son derlemede denetle.

### Hakem 2

- [ ] **R2.1 — Farklı veri setlerinde dış doğrulama:** APTOS 2019 dışındaki, tercihen farklı hastane/klinik kaynaklarından gelen etiketli retinal görüntülerde modeli değerlendir. Veri kaynağı, cihaz/merkez, etiket eşlemesi, preprocessing, sınıf dağılımı ve değerlendirme protokolünü açıkla; bu veri üzerinde tuning/model seçimi yapma. Veri erişimi, izin/lisans ve etiket eşlemesi ilk aşamada temin edilecek; erişim engeli çıkarsa alternatif uygun veri kaynağı/işbirliği aranacak ve teslim takvimi buna göre yeniden planlanacak.
- [ ] **R2.2 — Görsel veri artırma ve etkisi:** Küçük veri seti gerekçesiyle eğitim kümesinde kullanılan temel augmentation'ları açıkça tanımla; en azından random rotation ve horizontal/vertical flip seçeneklerinin klinik/anatomik uygunluğunu değerlendir. Aynı split, backbone, eğitim bütçesi ve seed'lerle augmentation yok/uygun augmentation karşılaştırması yap; hangi dönüşümlerin hangi olasılık/açı aralığında uygulandığını ve validation/test'e uygulanmadığını belirt. Sonuç farkını raporla. Deney tamamlanamazsa mevcut pipeline'da augmentation olup olmadığını doğru biçimde söyle; kullanılmamasının olası etkisini ve bu revizyonda ölçülemeyen katkıyı Limitations'ta açıkla.
- [ ] **R2.3 — Öğretmenler arasındaki başarı farkı ve zayıf öğretmenin gerekçesi:** Teacher 1'in bildirilen yaklaşık %63 başarımını (metrik adı ve hesaplamasıyla doğrula) Teacher 2 ile karşılaştır. R1.6'daki Teacher-2-only / two-teacher distillation ablation'ını, Teacher 1'in hata çeşitliliği ve tamamlayıcı bilgi ölçülerini kullanarak yanıtla. Teacher 1'in sisteme neden eklendiğini (tasarım gerekçesi) kanıtla; katkısı ölçülemiyor veya yoksa bunu kabul et, rolünü abartma ve çıkarılmasının olası etkisini sonuç/sınırlılıkta açıkla.
- [ ] **R2.4 — Hız, sistem yükü ve başarı artışı dengesi:** R1.3 ve R1.13'teki ortak benchmark'ı kullanarak öğrenci tek başına ile öğretmenler dahil tam inference sistemini aynı koşullarda ölç. Kullanıcının/hakemin belirttiği yaklaşık 3× yavaşlama iddiasını ölçümle doğrula veya düzelt; ms/image, throughput, parametre ve peak memory'yi ayrı raporla. Accuracy/QWK/macro-F1/AUC farkını aynı test örneklerinde karşılaştır; tartışmada ek başarının gecikme/bellek/hesaplama maliyetini haklı çıkarıp çıkarmadığını, hangi kullanım senaryosunda student-only'nin tercih edilebileceğini ve tam sistemin trade-off'unu açıkça değerlendir. Tam sistem ölçülemezse student-only latency'sini tüm sisteme mal etme.
- [ ] **R2.5 — Görsel açıklanabilirlik/Grad-CAM:** Çalışan ve checkpoint'i doğrulanmış model için Grad-CAM veya mimariye uygun eşdeğer saliency yöntemi uygula. Kullanılan model/inference modunu, hedef katmanı, hedef sınıfı ve görüntü örnekleme ölçütünü belirt; doğru ve hatalı tahminlerden, farklı şiddet sınıflarını kapsayacak temsilî örnekler göster. Isı haritalarını orijinal fundus görüntüsüyle üst üste ver ve caption'da bunların kararın nedensel açıklaması olmadığını, yalnızca model duyarlılığına ilişkin nitel görselleştirme olduğunu belirt.

### Hakem yanıtı ve kapanış ölçütü

- [ ] Yanıt mektubunda R1.1–R1.21 ve R2.1–R2.5 numaralarını kullan; her madde için hakemin talebini, yapılan değişikliği/deneyi, doğrulanmış sonucu ve revize makaledeki bölüm/tablo/şekil/sayfa-satır yerini belirt.
- [ ] Yapılmayan her talep için kibar ve doğrudan yanıt ver: hangi veri/izin/checkpoint/zaman kısıtının engel olduğunu söyle, ilgili iddiayı nasıl daralttığını göster ve gerekiyorsa future work olarak belirt. Gerçekleşmemiş deney için gelecek zamanla tamamlanmış gibi söz verme.
- [ ] Her sayısal değer, tablo ve görseli doğru deney ailesi, split, seed, örnek sayısı, inference modu ve artifact ile eşleştir. Student-only/full-system, teacher-only/teacher-student ve farklı deney ailelerinin sonuçlarını aynı satır veya iddiada karıştırma.
- [ ] Teslim öncesi bu listede hiçbir madde durumsuz kalmasın; kısmi/yapılamadı maddeleri hem yanıt mektubunda hem makalenin uygun sınırlılık/iddia düzenlemesinde görünür olsun.

## 5–18 Ekim Uygulama Takvimi

### 5 Ekim: Başlangıç ve Kaynak Dosyası Kapısı

- [x] Gönderilmiş makale ve hakem raporları incelendi.
- [x] Selection audit, low-data notebook ve mevcut aggregate sonuçlar incelendi.
- [x] `0_main.tex` içindeki test-bağımsızlık cümlesi bilinen test-metric ranking gerçeğine göre işaretli revizyona başlandı.
- [ ] Yetkili LaTeX kaynağını belirle: gönderilmiş PDF ile `0_main.tex`, `paper_latex/main.tex`, `paper_latex/main_prism.tex` arasındaki farkları karşılaştır. Yalnızca bir ana dosyayı revize et.
- [ ] Gönderim yönergesinden deadline time zone, marked/clean manuscript ve rebuttal biçimini doğrula.

### 6 Ekim: P0 Kanıt ve Erişim Kontrolü

- [ ] Equal-weight/gated tahminleri, teacher logits/probabilities, kullanılan etiket sırası ve checkpoint'leri iste/ara. Tam örnek bazlı çıktılar yoksa QWK eşitliği iddiasını doğrulanamaz olarak işaretle.
- [ ] Tablo 2 QWK ve confidence interval hesabının kaynağını bul; aynı truth, prediction order ve QWK implementasyonu ile yeniden hesapla. QWK sonuçları ancak karşılaştırma aynen tekrar üretilebiliyorsa düzelt.
- [ ] APTOS dosyaları, öğretmen ve student checkpoint'leri, eğitim ortamı/GPU, dış veri kümesi erişimi ve kullanılabilir hesaplama saatlerini doğrula.
- [ ] Mevcut test setinin model ranking'de kullanıldığını makalede ve yanıtta dürüstçe belirt; bu veri kümesini geriye dönük biçimde bağımsız test olarak nitelendirme.

### 7 Ekim: Protokolü Kilitle ve Uzun İşleri Başlat

- [ ] Deney matrisi, seed'ler, sabit train/validation bölünmesi, checkpoint seçim ölçütü, logit/probability tanımları, metrikler ve latency ölçüm yöntemini yazılı olarak kilitle.
- [ ] Kontrollü ConvNeXt Tiny loss ablation'ını, Teacher-2-only/two-teacher karşılaştırmasını, low-data oranlarını ve preprocessing/augmentation koşullarını başlat; işleri GPU kapasitesine göre paralel veya kuyruklu planla.
- [ ] Dış veri erişimi/izinleri için veri sağlayıcıları ve olası işbirlikçileriyle teyit al; veri temini gecikiyorsa aynı gün alternatif uygun harici veri kaynağı ve teslim süresi seçeneğini devreye sok.
- [ ] Yeni deneylerde test setini checkpoint seçimi, hiperparametre ayarı veya model ailesi sıralaması için kullanma. Mevcut APTOS test split'i geçmişte kullanılmış olduğundan yeni deney için bağımsızlık iddiası kurma; bağımsız tahmin için harici/önceden dokunulmamış bir set gerekir.
- [ ] Veri/checkpoint/GPU/izin erişimindeki engelleri aynı gün ilgili yazarlara/kuruma bildir; engel kalkana kadar erişim gerektirmeyen analizleri başlat ve süre uzatımı gerekiyorsa editöre erken başvur. Deney kapsamını varsayılan olarak daraltma.

### 8–10 Ekim: Deneyleri Çalıştır, Analizi Paralel Yürüt

- [ ] Eğitim job'larını log, disk alanı, checkpoint yazımı ve seed ile takip et; sonuçları mevcut manuscript artifacts üzerine yazma.
- [ ] Önce QWK çelişkisini çözecek görüntü başı tahminlerden eşleşme oranı, iki QWK, confusion matrix ve paired bootstrap CI üret; tahmin dosyası yoksa checkpoint'lerden tahminleri yeniden çıkar.
- [ ] Hızlı ve eğitim gerektirmeyen analizleri tamamla: sınıf-bazlı metrikler, referable-DR tanımı/ölçümü, parametre sayımı, student-only vs full-system inference modu.
- [ ] Latency ve peak memory'yi student-only ve tam sistem için aynı donanım, batch, precision, warm-up, tekrar sayısı ve CUDA synchronization ile ölç.
- [ ] Kaynak metindeki unsupported saliency, calibration, complementary attention ve test-independent seçim iddialarını gerçek kanıtla eşleştir; kanıt yoksa yumuşat/kaldır.
- [ ] Grad-CAM örneklerini üret; kalibrasyon, averaging, augmentation ve normalization deneylerini tamamlayıp her iş için log/çıktı ve seed bilgisini kaydet.

### 11–12 Ekim: Ara Sonuç ve Kapsam Kararı

- [ ] Her deney grubunda koşu/seed tamamlanma durumunu, convergence'ı, hata loglarını ve kalan süre tahminini kontrol et; başarısız işleri düzeltip yeniden kuyruğa al.
- [ ] Tam çoklu-seed ablation veya harici doğrulama 18 Ekim'e yetişmeme riski taşıyorsa, erişim/hesaplama darboğazını çözmek için ek GPU/işbirlikçi/uzatma seçeneğini derhal devreye al.
- [ ] Harici verinin preprocessing/etiket uyumu, merkez-cihaz bilgisi, izin/lisans ve hasta/görüntü çakışması kontrollerini tamamla; harici testte tuning yapılmadığını güvenceye al.
- [ ] Teacher contribution ablation, augmentation, low-data fractions, normalization, calibration, averaging ve Grad-CAM için sonuçların tamamlanmasını izle; bu ara kapı bir eskalasyon kararıdır, kapsamdan çıkarma kararı değildir.

### 13–14 Ekim: Sonuçları Dondur ve Tabloları Yeniden Kur

- [ ] Tüm yeni deney çıktıları ve harici doğrulama dosyalarını read-only kaynak kabul et; her sonucu run, seed, split, inference mode ve metrik fonksiyonuna bağla.
- [ ] Tablo 1–5 ve şekillerde student-only/full-system etiketlerini düzelt; total parameters sayısını instantiated model bileşenlerinden hesapla.
- [ ] Ablation tablosunu aynı ConvNeXt Tiny backbone ile izole et; bileşenler birlikte değişen tarihsel karşılaştırmayı kontrollü ablation diye sunma.
- [ ] Sınıf-bazlı confusion/report değerleri, sample support, macro-F1/QWK/AUC ve güven aralıklarını doğrula. Çok küçük sınıflar için belirsizliği açıkça göster.
- [ ] Tüm ek deney sonuçlarını ve Grad-CAM görsellerini ana metin/ek materyalde doğru kapsamda sun; tamamlanmamış analiz varsa teslimden önce engeli çöz veya uzatma al, yapılmış gibi raporlama.

### 15–16 Ekim: Makale Revizyonu ve Hakem Yanıtı

- [ ] Abstract, Methods, Results, Discussion, Conclusion, Limitations ve captions'ı yalnızca doğrulanmış sonuçlarla güncelle.
- [ ] Geçmiş test-set seçimi için şeffaf açıklama yap; test-bağımsızlığı iddiasını koruma. Bağımsız test yoksa iddiayı geri çek ve genellenebilirlik/deployment dilini sınırla.
- [ ] Reviewer 1 ve 2'nin her maddesi için Word yanıt mektubu hazırla: yorum, kısa ve kanıta dayalı yanıt, makaledeki bölüm/sayfa/line yeri. Teşekkür cümlesi yanıt mektubunda olsun; makalede hakeme hitap etme.
- [ ] Dergi talimatlarına göre strikeout/highlight revizyonu ve gerekiyorsa temiz kopyayı üret. Eski metni `\\sout{...}`, yeniyi `\\hlcyan{...}` ile göster.
- [ ] Kaynakça, DOI/yıl/yayın durumu, metin-içi referans ve AI-use beyanını kontrol et.

### 17 Ekim: Tam Kontrol ve Teslim Paketi

- [ ] LaTeX derlemesi, şekil/tablo numaraları, referanslar, sayfa/line atıfları ve highlight görünümünü kontrol et.
- [ ] Her hakem maddesini yanıt mektubuna karşı tek tek doğrula; yapılmamış deney/sonuç sözü olmadığını kontrol et.
- [ ] Sayısal değerleri sonuç dosyalarına son kez karşılaştır; test/gated QWK çelişkisi doğrulanmadıysa çözülmüş gibi yazma.
- [ ] Marked manuscript, clean manuscript (isteniyorsa), rebuttal, ek dosyalar ve yeniden üretim artifacts listesini hazırla.

### 18 Ekim: Son Gün ve Tampon

- [ ] Yalnızca son yazım/format düzeltmeleri yap; yeni, uzun eğitim başlatma.
- [ ] Teslim sisteminin dosya biçimi, beyan, cover letter ve yanıt mektubu gereksinimlerini doğrula.
- [ ] Son teslim öncesi gönderilen dosyaları açıp kontrol et; teslim onayı ve zaman damgasını sakla.

## 18 Ekim'e Yetişme Değerlendirmesi

**Planlanan kapsam;** metin revizyonu, artifact denetimi ve Hakem 1 ile Hakem 2'nin talep ettiği ek deneylerin tamamını içerir. 18 Ekim'e yetişmek için veri, checkpoint, izin ve hesaplama erişimi ilk iki günde çözülmeli; deney matrisi 7 Ekim'de kilitlenip koşular aynı gün başlatılmalıdır. Dış veri doğrulaması, çoklu-seed loss ablation, low-data oranları, augmentation/normalization, kalibrasyon/logit averaging ve tam sistem benchmark'ı kritik yol işleridir; ara tarihlerde ilerleme ölçülerek gerekli ek kaynak zamanında devreye alınacaktır.

**18 Ekim tesliminin tamamlanma ölçütü:** Hakem 1'in R1.1–R1.21 ve Hakem 2'nin R2.1–R2.5 maddelerinin her biri yanıt mektubunda kapatılmış; talep edilen tüm deneyler ve analizler ya makale/ek materyalde gerçek çıktılarıyla sunulmuş ya da çözülmesi mümkün olmayan belgeli bir engel için editörle süre/kapsam kararı alınmış olmalıdır. Hiçbir koşulda deney yapılmış gibi gösterilmeyecek. Revizyon için öngörülen ek deneyler “minimum paket” veya isteğe bağlı kapsam olarak değerlendirilmemelidir.

**Kritik yol ve risk yönetimi:** Harici veri kümesi erişimi/izinleri, tam multi-seed ablation matrisi, düşük veri oranları, augmentation ve ImageNet normalization koşulları, kalibrasyon/logit-probability averaging, Grad-CAM ve tekrarlı tam sistem benchmark'ı en yüksek süre riskine sahiptir. Bu riskler işleri GPU'ya göre paralelleştirerek, erişim taleplerini 6 Ekim'de başlatarak, 12 Ekim'de ek kaynak/uzatma kararı vererek yönetilecektir; bunları önceden kapsam dışı sayma veya yalnızca sınırlılık olarak bırakma planı yoktur.

## Kritik Bilinen Engeller

- Repoda APTOS dataset görüntüleri/CSV'leri yok; `.gitignore` veri klasörlerini hariç tutuyor. Bunların çalışma makinesinde repo dışı bulunup bulunmadığı ayrıca teyit edilmeli.
- Low-data sonuç klasöründe per-image equal-weight/gated prediction dosyaları görünmüyor; yalnızca aggregate sonuçlar ve bir seed için sınıf raporu/confusion matrix var.
- Referans verilen teacher/student `.pth` checkpoint'leri envanterde yok.
- Mevcut kayıtlar test QWK ile model ailesi sıralandığını gösteriyor; mevcut test kümesi sonradan bağımsız test kümesine dönüşmüş sayılamaz.
- Kullanıcı tarafından daha önce yapılmış çalışma ağacı değişikliklerine dokunulmayacak; bu plan yalnızca revizyon yürütme sırasını tanımlar.

## Başlangıç Durumu

- [x] Gönderilmiş makale ve hakem yorumları incelendi.
- [x] Repository selection audit ve low-data sonuç kayıtları incelendi.
- [x] `0_main.tex` içindeki test-set bağımsızlığı iddiası, bilinen test-metric ranking gerçeğine göre strikeout/highlight ile revize edilmeye başlandı.
- [ ] Yetkili LaTeX kaynak dosyası ve PDF eşleşmesi kesinleştirilecek.
- [ ] Equal-weight/gated prediction artifacts temin edilip QWK uyuşmazlığı doğrulanacak.
- [ ] 6 Ekim'e kadar veri/checkpoint/GPU erişimi doğrulanacak.
- [ ] 7 Ekim'de protokol kilitlenip ilk uzun deney kuyruğa alınacak (erişim kapısı geçilirse).
- [ ] 12 Ekim'de 18 Ekim kapsamı için ara go/no-go kararı verilecek.
- [ ] 17 Ekim'de son derleme ve submission QA tamamlanacak.

## Değişiklik İşaretleme

Makale revizyon sürümünde çıkarılan eski ifade `\sout{...}` ile üstü çizili, yeni ifade `\hlcyan{...}` ile highlight olarak tutulacaktır. Hakemlere teşekkür ve nokta-nokta yanıtlar makaleye değil ayrı Word yanıt mektubuna yazılacaktır. Dergi ayrıca temiz kopya isterse işaretli ve temiz sürümler ayrı teslim edilir; kaynak sonuç dosyaları değiştirilmez.
