# Füzyon yöntemlerinin karşılaştırılması

Gösterim: `p1` ve `p2` T1 ile T2'nin softmax olasılıklarıdır. `z_s` öğrenci logitleridir.

## Kodda bulunan yöntemler

| Yöntem | Matematiksel tanım | Ağırlıklar | Kod konumu | Hangi sayıda kullanıldı |
|---|---|---|---|---|
| Teacher Gate (öğretmen topluluğu) | `p_t = (1−g_t)·p1 + g_t·p2`, `z_t = log(clamp(p_t))` | g_t **öğrenilmiş** ve **örneğe bağlı**; T2 eğitim aşamasında öğreniliyor, öğrenci aşamasında donduruluyor | Eski: notebook `HeavyTeacherEnsemble` (satır 284–323). Yeni: `dr_pipeline.py` `HeavyTeacherEnsemble(fusion_mode="gate")` | Hem eski hem yeni sistemin öğretmen yolu |
| Teacher Equal (olasılık) | `z = log((p1+p2)/2)` | Sabit 0.5 | Eski: `revision_additional_analyses.json` → `teacher_equal_probability`. Yeni: `dr_pipeline.py` satır 518–519 (`z_equal_teacher`) | Yalnızca karşılaştırma |
| Teacher Equal (ham logit) | `z = (z1+z2)/2` | Sabit 0.5 | Eski doğrulama `teacher_equal_logit`; yeni `fusion_mode="equal_logits"` | Yalnızca karşılaştırma |
| **Teacher+Student Gate2 / kapılı residual** | `z = z_t + 0.70·g·(z_s − mean(z_s))`. Çatışma freni: öğretmen güveni 0.82'den büyükse ve tahminler çelişiyorsa `g ≤ 0.66·(1−conf)` | g **öğrenilmiş**, **örneğe bağlı** (girdi: p_t, p_s, entropiler, güven farkı) | Eski: notebook `ResidualTeacherStudentModel` (satır 347–418). Yeni: `dr_pipeline.py` (satır 119–167, `mix="gated"`) | **Eski 0.9186** (`archived_residual`). Yeni deneyde `primary` = 0.8996 |
| **Teacher+Student Equal** | `z = (z_t + z_s)/2` | Sabit 0.5; **yalnızca çıkarımda** uygulanıyor, aynı kapılı ağırlıklarla | Eski: `verify_archived_final.py` satır 237 (`equal_ts_archived_teacher`). Yeni: `run.py` satır 205 (`equal`) | **Yeni 0.9025**. Eski deneyde aynı yol 0.9174 |
| Makalede anlatılan eşit ortalama | `z = ((z1+z2)/2 + z_s)/2` (ham öğretmen logitlerinin ortalaması) | Sabit | `verify_archived_final.py` satır 238 (`manuscript_equal_logits`) | Eski checkpoint'lerde 0.9083; yeni deneyde hesaplanmadı |
| `mix="equal"` ile eşit eğitim | Eğitim kaybı `(z_t+z_s)/2` üzerinde | Sabit | `dr_pipeline.py` satır 119, 150 | Yeni 99 iş matrisinde **kullanılmadı**: protokolde `original joint gated output` yazıyor |

## "Gate2" ve "EqualWeight" adlandırması

Eski `teacher_student_inference_comparison.csv` dosyasında seed 42 için `Teachers_Student_Gate2` ve `Teachers_Student_EqualWeight` test sonuçlarının ikisi de QWK 0.917439'dur. Bu değer `archived_residual` ve `equal_ts_archived_teacher` değerleriyle aynıdır; seed 42'de iki yol arasında 0 etiket farkı vardır. AUC değerleri farklıdır (0.9583 ve 0.9453).

## Makaledeki anlatım ile kodun farkı

- Orijinal makale bağımsız öğretmenlerden, **eşit ağırlıklı logit ortalamasından** ve öğretmen ile öğrenci logitlerinin doğrudan ortalanmasından söz eder.
- Kodda ise öğretmen birleşimi **öğrenilmiş bir olasılık kapısıdır**. T2, T1 dondurulmuş haldeyken bu karışım içinde eğitilir.
- 0.9186 değerini üreten çıktı da **öğrenilmiş kapılı residual** çıktıdır. Makalede anlatılan eşit ortalama aynı ağırlıklarla 0.9083 verir.
- Bu durum, orijinal makaledeki yöntem anlatımının 0.9186 sayısını üreten yöntemle aynı olmadığını gösterir. Fark daha önceki denetimlerde de bulunmuştu ve burada kodla yeniden doğrulandı.

## İki sonucun füzyon açısından karşılaştırılması

| | Eski | Yeni | Fark |
|---|---|---|---|
| Kapılı residual | **0.9186** | 0.8996 | −0.0190 |
| Equal (T+S) | 0.9174 | **0.9025** | −0.0149 |
| Yalnızca öğretmen | 0.9141 | 0.8944 | −0.0197 |

0.9186 ile 0.9025 **farklı füzyon yollarıdır**: biri kapılı, diğeri eşit ağırlıklı.

Ancak aynı yollar birbiriyle karşılaştırıldığında da fark sürer: kapılı yolda −0.019, eşit yolda −0.015. Füzyon seçiminin bu iki deneydeki etkisi küçüktür (eski deneyde equal − kapılı = −0.0012, yeni deneyde +0.0029). Füzyon farkı, 0.0161'lik toplam açığı açıklamaz; aksine açığı yaklaşık 0.003 kadar daraltır.
