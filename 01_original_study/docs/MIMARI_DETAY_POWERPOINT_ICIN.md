# TS_ConvNeXtTiny_Residual — Mimari Detayları
## PowerPoint 3D Şema ve Makale Proposed Model Bölümü İçin Tam Kılavuz

> **Karar:** Gate2 kaldırıldı → **Equal-Weight Fusion** kullanılıyor  
> **Sonuç:** Test Accuracy **%85.29**, QWK **0.9174** (3 seed ort: %85.47, QWK 0.9186)  
> **Veri:** APTOS 2019 — 3,665 görüntü (5 sınıf)

---

## BÖLÜM 1: GENEL SİSTEM AKIŞI (En Üst Düzey Şema)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GİRDİ: Fundus Görüntüsü  224 × 224 × 3                │
└────────────────────┬────────────────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐   ┌──────────────────────┐
│   TEACHER 1     │   │     TEACHER 2         │
│ Baseline        │   │ Advanced              │
│ CNN-ViT Hybrid  │   │ ResNet50 + ViT-B/16  │
│                 │   │ + Spectral Norm        │
│ Params: ~15M    │   │ Params: ~108M          │
│ Val Acc: 66.76% │   │ Val Acc: 83.92%        │
└────────┬────────┘   └──────────┬────────────┘
         │                       │
         │  softmax(z_base)      │  softmax(z_adv)
         │  [5-dim prob vector]  │  [5-dim prob vector]
         └───────────┬───────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  TEACHER ENSEMBLE   │
          │  Equal-Weight Avg   │
          │                     │
          │  p_teacher =        │
          │  0.5×p_base +       │
          │  0.5×p_adv          │
          │                     │
          │  [5-dim soft label] │
          └──────────┬──────────┘
                     │
         ┌───────────┴───────────┐
         │   (aynı girdi resmi)  │
         ▼                       ▼
┌─────────────────┐   ┌──────────────────────┐
│   STUDENT       │   │  MULTI-COMPONENT     │
│ ConvNeXt-Tiny   │   │  TRAINING LOSS       │
│                 │   │                      │
│ Params: 28M     │   │  L = L_CE            │
│ Backbone:       │◄──│    + λ_ni · L_ni     │
│ ImageNet pretrain│   │    + λ_d · L_distill │
│ Head: FC(768→5) │   │    + λ_o · L_ord     │
└────────┬────────┘   └──────────────────────┘
         │
         │  softmax(z_student)
         │  [5-dim prob vector]
         │
         └───────────┬───────────┐
                     │           │
                     │  p_teacher│ (from above)
                     ▼
          ┌─────────────────────┐
          │  INFERENCE FUSION   │
          │  Equal-Weight Avg   │
          │                     │
          │  p_final =          │
          │  0.5×p_teacher +    │
          │  0.5×p_student      │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  SINIF TAHMİNİ     │
          │  ŷ = argmax(p_final)│
          │  Grade: 0, 1, 2,    │
          │         3 veya 4    │
          └─────────────────────┘
```

---

## BÖLÜM 2: TEACHER 1 — BASELINE CNN-ViT HYBRİD (Detaylı)

### 2.1 CNN Branch (Sol Dal)

```
GİRDİ: 224 × 224 × 3
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  STEM BLOCK                                          │
│  Conv2d(3→64, kernel=7, stride=2, padding=3)         │
│  → BatchNorm2d(64) → ReLU                            │
│  → MaxPool2d(kernel=3, stride=2, padding=1)          │
│  ÇIKTI: 56 × 56 × 64                                │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  LAYER 1 (2× ConvBlock)                              │
│  ConvBlock(64→64) + ConvBlock(64→64)                 │
│  Her ConvBlock:                                      │
│    Conv(3×3) → BN → ReLU → Conv(3×3) → BN           │
│    + Residual shortcut (skip connection)             │
│  ÇIKTI: 56 × 56 × 64                                │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  LAYER 2 (2× ConvBlock, stride=2)                    │
│  ConvBlock(64→128, stride=2) + ConvBlock(128→128)    │
│  ÇIKTI: 28 × 28 × 128                               │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  LAYER 3 (2× ConvBlock, stride=2)                    │
│  ConvBlock(128→256, stride=2) + ConvBlock(256→256)   │
│  ÇIKTI: 14 × 14 × 256                               │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Global Average Pooling (GAP)                        │
│  AdaptiveAvgPool2d(1) → 256-dim vektör               │
│  + Fully Connected: Linear(256→512) + ReLU           │
│  ÇIKTI: f_cnn ∈ ℝ^512                               │
└──────────────────────────────────────────────────────┘
```

### 2.2 ViT Branch (Sağ Dal)

```
GİRDİ: 224 × 224 × 3
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  PATCH EMBEDDING                                     │
│  Conv2d(3→256, kernel=16, stride=16)                 │
│  224/16 = 14 → 14×14 = 196 patch                    │
│  + [CLS] token eklenmez, mean pooling kullanılır     │
│  ÇIKTI: 196 token × 256 dim                         │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  POZİSYON EMBEDDING                                  │
│  Öğrenilebilir positional encoding eklenir           │
│  ÇIKTI: 196 × 256 (konumsal bilgi dahil)            │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  TRANSFORMER BLOCK × 6 (her biri aynı yapı)          │
│                                                      │
│  Her blok:                                           │
│  ┌────────────────────────────────────────────┐      │
│  │  LayerNorm(256)                             │      │
│  │  Multi-Head Self-Attention (8 kafa, d=256) │      │
│  │  → Q, K, V projections (256→256)           │      │
│  │  → Scaled dot-product attention            │      │
│  │  → Concat + Linear(256→256)               │      │
│  │  + Residual                                │      │
│  │                                            │      │
│  │  LayerNorm(256)                             │      │
│  │  MLP: Linear(256→512) → GELU              │      │
│  │       Linear(512→256)                      │      │
│  │  + Residual                                │      │
│  └────────────────────────────────────────────┘      │
│  ÇIKTI: 196 × 256                                    │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  TOKEN MEAN POOLING                                  │
│  196 token → dim boyutunda ortalama → 256-dim        │
│  + Fully Connected: Linear(256→512)                  │
│  ÇIKTI: f_vit ∈ ℝ^512                               │
└──────────────────────────────────────────────────────┘
```

### 2.3 Confidence-Gated Fusion (Her iki dal birleşimi)

```
f_cnn ∈ ℝ^512        f_vit ∈ ℝ^512
      │                      │
      ▼                      ▼
┌──────────┐          ┌──────────┐
│ Conf.    │          │ Conf.    │
│ Score 1  │          │ Score 2  │
│ s_c =    │          │ s_t =    │
│σ(W_c·f_cnn)│        │σ(W_t·f_vit)│
│ ∈ [0,1]  │          │ ∈ [0,1]  │
└────┬─────┘          └─────┬────┘
     │                      │
     └──────────┬───────────┘
                │
                ▼
     ┌──────────────────────┐
     │  GATE NETWORK (MLP)  │
     │  Input: [s_c, s_t]   │  (2-dim)
     │  → Linear(2→64)→ReLU │
     │  → Linear(64→1)→σ   │
     │                      │
     │  α ∈ (0,1)           │
     └──────────┬───────────┘
                │
                ▼
     ┌──────────────────────────────┐
     │  AĞIRLIKLI TOPLAM            │
     │  f_fused = α·f_cnn           │
     │          + (1-α)·f_vit       │
     │  ∈ ℝ^512                     │
     └──────────┬───────────────────┘
                │
                ▼
     ┌──────────────────────┐
     │  CLASS HEAD          │
     │  Linear(512→5)       │
     │  z_base ∈ ℝ^5        │
     └──────────────────────┘
```

---

## BÖLÜM 3: TEACHER 2 — ADVANCED CNN-ViT + SPECTRAL NORM (Detaylı)

```
GİRDİ: 224 × 224 × 3
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────────────────┐    ┌──────────────────────┐
│  ResNet50        │    │  ViT-B/16             │
│  (ImageNet       │    │  (ImageNet            │
│  pretrained)     │    │  pretrained)          │
│                  │    │                       │
│  4-stage:        │    │  Patch size: 16×16    │
│  64→256→512      │    │  196 patches          │
│  →1024→2048      │    │  768-dim embed        │
│                  │    │  12 transformer block │
│  Bottleneck      │    │  12 attention heads   │
│  residual blocks │    │                       │
│  GAP → 2048-dim  │    │  [CLS] token → 768-d  │
└────────┬─────────┘    └──────────┬────────────┘
         │                         │
         │ f_resnet ∈ ℝ^2048        │ f_vit ∈ ℝ^768
         └────────────┬────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  CONCAT                │
         │  [f_resnet ; f_vit]    │
         │  ∈ ℝ^(2048+768)=ℝ^2816 │
         └────────────┬───────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  SPECTRAL NORM FC      │
         │  SpectralNorm(         │
         │    Linear(2816→5)      │
         │  )                     │
         │  z_adv ∈ ℝ^5           │
         │                        │
         │  NOT: Spectral Norm    │
         │  → Lipschitz kısıtı    │
         │  → Daha iyi kalibrasyon│
         └────────────────────────┘
```

---

## BÖLÜM 4: TEACHER ENSEMBLE (Eşit Ağırlıklı Füzyon)

```
z_base ∈ ℝ^5         z_adv ∈ ℝ^5
      │                     │
      ▼                     ▼
 softmax(z_base)       softmax(z_adv)
 p_base ∈ Δ^4         p_adv ∈ Δ^4
   [5-dim prob]          [5-dim prob]
      │                     │
      └─────────┬───────────┘
                │
                ▼
    ┌─────────────────────────┐
    │  EQUAL-WEIGHT AVERAGE   │
    │                         │
    │  p_teacher =            │
    │   0.5 × p_base          │
    │ + 0.5 × p_adv           │
    │                         │
    │  p_teacher ∈ Δ^4        │
    │  (5-dim prob. simplex)  │
    └─────────────────────────┘
    
    Teacher Distillation Logit:
    z_t = log(p_teacher)  [log-probability]
    
    ─────────────────────────────────────────
    NOT: Gate2 neden kaldırıldı?
    ─────────────────────────────────────────
    Gate2 (Conflict-Aware Residual Fusion):
      z_final = z_teacher + α × g × (z_student - mean(z_student))
      Mean gate g = 0.09 (ÇOK DÜŞÜK)
      Student katkısı sadece %6.3
      Predictions: %100 equal weight ile AYNI
    → Gate2 hiçbir ek katkı sağlamadı, Equal-Weight yeterli
```

---

## BÖLÜM 5: STUDENT — CONVNEXT-TINY (Detaylı)

```
GİRDİ: 224 × 224 × 3
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  CONVNEXT-TINY BACKBONE                              │
│  (ImageNet pretrained, 28M parametre)                │
│                                                      │
│  STEM:                                               │
│  Conv2d(3→96, kernel=4, stride=4)                    │
│  → LayerNorm → 56×56×96                              │
│                                                      │
│  STAGE 1: 3× ConvNeXt Block (96 ch)                  │
│  ┌──────────────────────────────────────────────┐    │
│  │  Depthwise Conv2d(96→96, kernel=7, padding=3)│    │
│  │  LayerNorm → Linear(96→384) → GELU           │    │
│  │  Linear(384→96) + Residual                   │    │
│  └──────────────────────────────────────────────┘    │
│  ÇIKTI: 56×56×96                                     │
│                                                      │
│  DOWNSAMPLE: LayerNorm + Conv2d(96→192, k=2, s=2)    │
│                                                      │
│  STAGE 2: 3× ConvNeXt Block (192 ch)                 │
│  ÇIKTI: 28×28×192                                    │
│                                                      │
│  DOWNSAMPLE: LayerNorm + Conv2d(192→384, k=2, s=2)   │
│                                                      │
│  STAGE 3: 9× ConvNeXt Block (384 ch)                 │
│  ÇIKTI: 14×14×384                                    │
│                                                      │
│  DOWNSAMPLE: LayerNorm + Conv2d(384→768, k=2, s=2)   │
│                                                      │
│  STAGE 4: 3× ConvNeXt Block (768 ch)                 │
│  ÇIKTI: 7×7×768                                      │
│                                                      │
│  Global Average Pooling                              │
│  AdaptiveAvgPool2d(1) → 768-dim                      │
│  LayerNorm(768)                                      │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
         ┌────────────────────────┐
         │  CUSTOM CLASS HEAD     │
         │  (orijinal kaldırıldı) │
         │  Linear(768→5)         │
         │  z_student ∈ ℝ^5       │
         └────────────────────────┘
```

---

## BÖLÜM 6: EĞİTİM KAYIP FONKSİYONU (Loss Function)

### Formül

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{ni} \cdot \mathcal{L}_{ni} + \lambda_{distill} \cdot \mathcal{L}_{distill} + \lambda_{ord} \cdot \mathcal{L}_{ord}$$

### 6.1 Ağırlıklı Cross-Entropy (L_CE) — Ana kayıp

```
Amaç: Sınıf dengesizliğini gidermek
      (APTOS'ta Grade 0 >> Grade 4)

Formül:
  L_CE = -Σ_{c=0}^{4} w_c · y_c · log(p̂_c)

  Sınıf ağırlığı: w_c = N / (5 × n_c)
    N  = toplam örnek sayısı (2,930 train)
    n_c = sınıf c'deki örnek sayısı
    
  Sonuç: Nadir sınıflara (Grade 3, 4) daha yüksek ağırlık
```

### 6.2 Non-Inferiority Loss (L_ni) — Öğrenci-öğretmen parite kısıtı

```
Amaç: Student'ın teacher'dan KÖTÜ OLMAMASI
      (low-data rejiminde overfitting'e karşı koruma)

Formül:
  L_ni = (1/N) Σ_i max(0, ℓ_s^(i) - ℓ_t^(i) + m(epoch))

  ℓ_s^(i) = örnek i için student cross-entropy kaybı
  ℓ_t^(i) = örnek i için teacher ensemble cross-entropy kaybı
  m(epoch) = margin (zamanlanmış):
             Epoch 1 → 0.010 (katı kısıt)
             Epoch 70 → 0.003 (gevşek kısıt)
  
  λ_ni: 0.80 → 0.25 (lineer azalma)
  
  Yorumu: "Student her örnekte teacher'dan m kadar kötü olabilir,
           daha fazla değil."
```

### 6.3 Multi-Temperature Distillation Loss (L_distill)

```
Amaç: Teacher'ın "karanlık bilgisini" (dark knowledge) transfer et

Formül:
  L_distill = Σ_{k∈{1,2}} w_k · T_k² · KL(softmax(z_s/T_k) || softmax(z_t/T_k))

  T_1 = 2.0  → w_1 = 0.6  (ince-taneli, komşu sınıf ayrımı)
  T_2 = 4.0  → w_2 = 0.4  (kaba-anlamsal, uzak sınıf benzerliği)
  
  T_k² çarpanı: farklı sıcaklıklarda gradient büyüklüğünü dengeler
  
  λ_distill = 0.30
  
  Neden 2 sıcaklık?
  - T=2.0: Grade 1 vs Grade 2 sınırını korur (ince fark)
  - T=4.0: Grade 0 ile Grade 4 arasındaki semantik uzaklığı kodlar
```

### 6.4 Ordinal CDF Loss (L_ord) — DR sıralı yapısına uyum

```
Amaç: DR Grade'lerin sıralı yapısını (0<1<2<3<4) loss'a dahil et
      QWK metriği ile hizalanmak için

Formül:
  L_ord = (1/4) Σ_{k=0}^{3} (F̂_k - F_k*)²

  F̂_k = Σ_{c=0}^{k} p̂_c  (tahmin edilen kümülatif dağılım)
  F_k* = 1[y ≤ k]          (gerçek sınıf için hedef CDF)
  
  λ_ord = 0.12
  
  Örnek (y=2, yani Grade 2 için):
    k=0: F_0* = 0 (Grade 0 değil),  F̂_0 olmalı ≈ 0
    k=1: F_1* = 0 (Grade 1 değil),  F̂_1 olmalı ≈ 0
    k=2: F_2* = 1 (Grade 2 veya altı), F̂_2 olmalı ≈ 1
    k=3: F_3* = 1 (Grade 3 veya altı), F̂_3 olmalı ≈ 1
```

---

## BÖLÜM 7: INFERENCE (TAHMİN AŞAMASI)

```
                Test Fundus Görüntüsü
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
       [Teacher Ensemble]  [Student Network]
       p_teacher ∈ Δ^4    p_student = softmax(z_s) ∈ Δ^4
              │                 │
              └────────┬────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  EQUAL-WEIGHT FINAL     │
         │  p_final =              │
         │   0.5 × p_teacher       │
         │ + 0.5 × p_student       │
         └──────────┬──────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │  SINIF TAHMİNİ          │
         │  ŷ = argmax(p_final)    │
         │                         │
         │  0 → No DR              │
         │  1 → Mild NPDR          │
         │  2 → Moderate NPDR      │
         │  3 → Severe NPDR        │
         │  4 → Proliferative DR   │
         └─────────────────────────┘
```

---

## BÖLÜM 8: TÜM MODELLERİN PARAMETRE TABLOSU

| Bileşen | Mimari | Parametre Sayısı | Açıklama |
|---------|--------|-----------------|----------|
| Teacher 1 (Baseline) | Custom CNN-ViT | ~15M | Hafif teacher |
| Teacher 2 (Advanced) | ResNet50 + ViT-B/16 + SpNorm | ~108M | Güçlü teacher |
| Student | ConvNeXt-Tiny | 28M | Üretim modeli |
| Teacher Ensemble | — | 0 ek param | Sadece avg |
| Inference Fusion | — | 0 ek param | Sadece avg |

**NOT:** Gate2 kaldırılınca **2,049 parametre** tasarrufu (çok küçük, ama sıfır kazanım sağlamıştı).

---

## BÖLÜM 9: EĞİTİM HİPERPARAMETRELERİ (Tam Tablo)

| Hiperparametre | Değer |
|----------------|-------|
| Optimizer | AdamW |
| Öğrenme hızı | 2 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Batch size | 16 |
| Maks. epoch | 70 |
| LR scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| Early stopping (patience) | 10 epoch (val QWK izlenir) |
| Gradient clipping | max norm = 1.0 |
| λ_distill | 0.30 |
| λ_ordinal | 0.12 |
| λ_ni (zamanlanmış) | 0.80 → 0.25 (lineer azalma) |
| Non-inf. margin (zamanlanmış) | 0.010 → 0.003 (lineer azalma) |
| Distillation sıcaklıkları | T₁=2.0 (w=0.6), T₂=4.0 (w=0.4) |
| Veri bölümü | %80 train / %10 val / %10 test |
| Augmentation (student) | Yok (distillation katkısını izole etmek için) |

---

## BÖLÜM 10: DENEY SONUÇLARI (En iyi model karşılaştırması)

| Model | Test Acc. | F1-Macro | QWK | ROC-AUC |
|-------|----------|----------|-----|---------|
| Teacher 1 (Baseline) | 69.48% | — | — | — |
| Teacher 2 (Advanced) | 81.20% | — | — | — |
| Teacher Ensemble (Equal) | 81.47% | — | — | — |
| Student (tek başına) | ~79.89% | — | — | — |
| **TS_ConvNeXtTiny_Residual (önerilen)** | **85.29%** | **0.7046** | **0.9174** | **0.9453** |
| TS (Gate2, kaldırıldı) | 85.29% | 0.7046 | 0.9174 | 0.9583 |

**3 seed ortalaması:** Accuracy = **85.47%**, QWK = **0.9186**

---

## BÖLÜM 11: POWERPOINT 3D ŞEMA İÇİN BLOK TANIMLARI

### Şema Oluşturma Rehberi (PowerPoint'te 3D bloklar için)

#### KATMAN 1 (En üst — INPUT):
- **1 adet büyük dikdörtgen blok**
- Yazı: "Fundus Görüntüsü / 224×224×3 RGB"
- Renk: Açık mavi

#### KATMAN 2 (TEACHER'LAR — yan yana 2 blok):

**Sol Blok — Teacher 1:**
```
Başlık: "Teacher 1: Baseline CNN-ViT"
Alt bloklar (soldan sağa veya üstten alta):
  ① CNN Branch (ResNet-style)
     - Stem: 7×7 Conv
     - L1: 2× ConvBlock(64)
     - L2: 2× ConvBlock(128)
     - L3: 2× ConvBlock(256)
     - GAP + FC(256→512)
  ② ViT Branch
     - Patch Embed (16×16)
     - 6× Transformer Block
       (8-head MHSA + MLP)
     - Token Mean Pool + FC(256→512)
  ③ Confidence-Gated Fusion
     - s_c, s_t (confidence scores)
     - Gate MLP(2→64→1)
     - f_fused = α·f_cnn + (1-α)·f_vit
  ④ Classifier: Linear(512→5)
Renk: Turuncu tonları
```

**Sağ Blok — Teacher 2:**
```
Başlık: "Teacher 2: Advanced ResNet50+ViT-B/16"
Alt bloklar:
  ① ResNet50 Backbone (ImageNet pretrained)
     GAP → 2048-dim
  ② ViT-B/16 Encoder (ImageNet pretrained)
     [CLS] token → 768-dim
  ③ Concat(2048+768) = 2816-dim
  ④ Spectral Norm FC(2816→5)
Renk: Yeşil tonları
```

#### KATMAN 3 (TEACHER ENSEMBLE):
- **1 adet dikdörtgen, ortada**
- Yazı: "Teacher Ensemble\n p_teacher = 0.5·p₁ + 0.5·p₂"
- Renk: Sarı/Altın

#### KATMAN 4 (STUDENT):
- **1 adet orta boy dikdörtgen**
- Yazı: "Student: ConvNeXt-Tiny\n28M params"
- Alt bloklar:
  ```
  Stem (96ch) → Stage1(96) → DS → Stage2(192)
  → DS → Stage3(384) → DS → Stage4(768) → GAP
  → FC(768→5)
  ```
- Renk: Mor/Pembe tonları

#### KATMAN 5 (LOSS — sadece eğitimde):
- **4 adet küçük dikdörtgen, yatay**
- Bloklar: `L_CE | L_ni | L_distill | L_ord`
- Kesik çizgili kenarlık (inference'ta aktif değil)
- Renk: Kırmızı tonları

#### KATMAN 6 (INFERENCE FUSION):
- **1 adet orta dikdörtgen**
- Yazı: "Inference Fusion\n p_final = 0.5·p_teacher + 0.5·p_student"
- Renk: Sarı

#### KATMAN 7 (OUTPUT):
- **5 küçük kutu** (DR Grade 0'dan 4'e)
- Yazı: "Grade 0 / Grade 1 / Grade 2 / Grade 3 / Grade 4"
- Renk: Yeşilden kırmızıya gradient

---

## BÖLÜM 12: MAKALE PROPOSED MODEL BÖLÜMÜ İÇİN EKSTRA NOTLAR

### Gate2 Kaldırılmasının Gerekçesi (Literature'da desteklenmesi için)

**Bulgular:**
1. Mean gate değeri: g = 0.09 (çok düşük, student %6.3 etkili)
2. Teacher confidence mean: 0.88 (çok yüksek)
3. Gate dağılımı: örneklerin %86.4'ünde g < 0.20
4. Equal weight ile predictions: %100 identik (368/368)

**Akademik gerekçe (makalede yazılacak):**
> "An entropy-gated combination mechanism (Gate2) was investigated empirically; however, in the low-data setting where the teacher ensemble confidence remained consistently high (mean confidence ≈ 0.88), the gate collapsed to near-zero student weighting (mean gate value ≈ 0.09), producing predictions statistically indistinguishable from equal-weight averaging. This finding is consistent with prior observations that gating mechanisms require sufficient prediction uncertainty to be effective [ref]. Equal-weight fusion was therefore adopted as the production strategy, offering identical performance with substantially reduced complexity."

### Şekil Başlıkları (Figure Captions) için öneri

```
Figure 1. Overview of the proposed TS_ConvNeXtTiny_Residual framework. 
Two independently trained CNN-ViT hybrid teacher networks produce class 
probability distributions combined via equal-weight ensemble averaging 
(p_teacher). A ConvNeXt-Tiny student is optimized using a multi-component 
loss. At inference, teacher and student predictions are combined equally.

Figure 2. Teacher 1 architecture: Baseline CNN-ViT hybrid with confidence-
gated fusion. The CNN branch follows a ResNet-style residual design 
(Stem→L1→L2→L3→GAP), while the ViT branch uses 6-block transformer 
encoding with token mean pooling. A 2-layer MLP gate learns per-sample 
branch weights.

Figure 3. Teacher 2 architecture: Advanced hybrid combining pretrained 
ResNet50 (2048-dim) and ViT-B/16 ([CLS] token, 768-dim), concatenated 
and classified through a spectral-normalized linear head.

Figure 4. Multi-component training loss components: (a) Weighted cross-
entropy L_CE, (b) Non-inferiority loss L_ni with scheduled margin, 
(c) Multi-temperature distillation L_distill (T=2.0, T=4.0), 
(d) Ordinal CDF loss L_ord.
```

---

## BÖLÜM 13: SEMBOLLER VE NOTASYON ÖZETI

| Sembol | Anlam | Boyut |
|--------|-------|-------|
| x | Giriş fundus görüntüsü | ℝ^(B×3×224×224) |
| f_cnn | CNN branch feature vektörü (T1) | ℝ^512 |
| f_vit | ViT branch feature vektörü (T1) | ℝ^512 |
| α | Confidence gate değeri | ℝ (0,1) |
| f_fused | Gate-fused feature (T1) | ℝ^512 |
| z_base | Teacher 1 logits | ℝ^5 |
| f_resnet | ResNet50 GAP çıktısı (T2) | ℝ^2048 |
| f_vitb | ViT-B/16 CLS token (T2) | ℝ^768 |
| z_adv | Teacher 2 logits (spectral norm) | ℝ^5 |
| p_teacher | Teacher ensemble prob. | Δ^4 (5-simplex) |
| z_t | Teacher distill. logit = log(p_teacher) | ℝ^5 |
| z_s | Student logits | ℝ^5 |
| p_student | softmax(z_s) | Δ^4 |
| p_final | Inference final prob. | Δ^4 |
| ŷ | Tahmin edilen DR grade | {0,1,2,3,4} |
| T_k | Distillation sıcaklığı | {2.0, 4.0} |
| m(epoch) | Non-inf. margin | 0.010→0.003 |
| F̂_k | Tahmin edilen CDF at k | [0,1] |
| F_k* | Hedef CDF at k | {0,1} |

---

*Bu doküman PowerPoint 3D mimari şeması ve makale Proposed Model bölümü için hazırlanmıştır.*  
*Gate2 mekanizması hocaların geri bildirimi doğrultusunda kaldırılmıştır.*  
*Son güncelleme: 10 Mayıs 2026*
