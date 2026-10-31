IDRiD external evaluation and branch Grad-CAM
===========================================
Run from the repository with the existing CPU PyTorch environment:

python .\revision_experiments\evaluate_idrid.py --output .\results\idrid_external_2026-10-16

For the selected equal teacher-student inference rule, use a NEW output folder:
python .\revision_experiments\evaluate_idrid.py --inference equal --output .\results\idrid_equal_inference_2026-10-16
This retains learned T1/T2 probability fusion and directly averages z_teacher
and z_student. The outer gate, residual centering/scaling and brake are not
executed. All inference metrics, example selection and Grad-CAM targets follow
the selected rule; earlier residual results are not overwritten.

Defaults: local IDRiD Disease Grading ZIP in Downloads and archived checkpoints
in D:\m1\Documents. No training, downloading, test fitting or checkpoint selection.
Only the official 103-image test split is evaluated; training images are unused.
Use Retinopathy grade (0..4), not Risk of macular edema.

Preprocessing exactly follows archived APTOS evaluation: RGB, PIL bilinear
224x224, float32 pixels 0..255 without division or ImageNet normalization.
Each seed is evaluated with strict checkpoint loading. Predictions, probabilities,
class metrics, calibration metrics, confusion matrices, and source hashes persist.
10,000 shared image-bootstrap replicates estimate uncertainty of the mean metric
over three fixed seeds; they do not account for patient grouping or retraining.
Class-1 test support is only five images. Do not infer clinical validation.

Grad-CAM uses a fixed seed 42, with the first lexicographic image in each
true-grade/correctness stratum. It targets the full system's predicted-class logit.
Scoped gradient enabling removes only no_grad detachment during attribution;
original evaluation formulas and eval mode are retained. Original and gradient-
enabled logits must agree within atol=1e-4, rtol=1e-5 for every displayed case.
Teacher weights remain gradient-frozen. There is no optimizer or backward update.
Three maps describe CNN feature tensors: T1 cnn_backbone, T2 cnn_features,
and ConvNeXt features. They are not a full joint-system map or ViT explanation.
For residual inference, gates and the locally active conflict-brake path are
differentiated; hard class and threshold decisions remain locally fixed.
For equal inference, only the teacher-level gate is in the inference graph.
Positive maps are independently
normalized, so their color intensity cannot compare branch importance.
Zero-positive maps are explicitly flagged, not replaced with invented saliency.
No lesion annotations are used; maps do not establish lesion localization.

IDRiD: cite Porwal et al., Indian Diabetic Retinopathy Image Dataset (IDRiD):
A Database for Diabetic Retinopathy Screening Research, Data 2018, 3(3), 25.
https://doi.org/10.3390/data3030025
Honor the CC BY 4.0 terms included in the downloaded dataset.
