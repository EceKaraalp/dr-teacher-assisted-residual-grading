# Round-2 reviewer experiments (APTOS frozen test, 3 seeds)

Mean +/- sample SD across seeds. Paired bootstrap: 10,000 shared-image resamples, conditional on trained models.

## Reference provenance

- reference/raw_f100_s42/student/joint_full: imported
- reference/raw_f100_s42/student/student_ce: imported
- reference/raw_f100_s42/teacher1: imported
- reference/raw_f100_s42/teacher2: imported
- reference/raw_f100_s52/student/joint_full: imported
- reference/raw_f100_s52/student/student_ce: imported
- reference/raw_f100_s52/teacher1: imported
- reference/raw_f100_s52/teacher2: imported
- reference/raw_f100_s62/student/joint_full: imported
- reference/raw_f100_s62/student/student_ce: imported
- reference/raw_f100_s62/teacher1: imported
- reference/raw_f100_s62/teacher2: imported

## Test metrics

| Group | Path | Accuracy | Macro-F1 | QWK | NLL | ECE | Calibrated ECE |
|---|---|---|---|---|---|---|---|
| reference/teacher1 | primary | 0.6912 +/- 0.0069 | 0.4025 +/- 0.0294 | 0.6814 +/- 0.0232 | 0.9126 +/- 0.0080 | 0.1060 +/- 0.0239 | 0.1045 +/- 0.0089 |
| reference/teacher2 | primary | 0.8365 +/- 0.0047 | 0.6775 +/- 0.0162 | 0.8969 +/- 0.0046 | 0.6282 +/- 0.0834 | 0.0938 +/- 0.0570 | 0.0440 +/- 0.0240 |
| reference/teacher2 | teacher1 | 0.6912 +/- 0.0083 | 0.3905 +/- 0.0690 | 0.6731 +/- 0.0394 | 0.9100 +/- 0.0063 | 0.1102 +/- 0.0166 | 0.1040 +/- 0.0222 |
| reference/teacher2 | teacher2 | 0.8374 +/- 0.0057 | 0.6785 +/- 0.0176 | 0.8973 +/- 0.0053 | 1.0687 +/- 0.1048 | 0.1378 +/- 0.0136 | 0.0519 +/- 0.0070 |
| reference/teacher2 | teacher_raw_average | 0.8356 +/- 0.0031 | 0.6745 +/- 0.0059 | 0.8870 +/- 0.0094 | 0.6489 +/- 0.0457 | 0.1033 +/- 0.0144 | 0.0528 +/- 0.0238 |
| reference/teacher2 | teacher_probability_average | 0.8347 +/- 0.0042 | 0.6734 +/- 0.0111 | 0.8873 +/- 0.0065 | 0.5759 +/- 0.0140 | 0.0848 +/- 0.0187 | 0.0505 +/- 0.0319 |
| reference/student/joint_full | primary | 0.8438 +/- 0.0079 | 0.6905 +/- 0.0203 | 0.8996 +/- 0.0077 | 0.6165 +/- 0.0925 | 0.1028 +/- 0.0340 | 0.0400 +/- 0.0019 |
| reference/student/joint_full | student | 0.2570 +/- 0.3140 | 0.2120 +/- 0.3185 | 0.2634 +/- 0.4561 | 2.1824 +/- 0.4470 | 0.3627 +/- 0.1059 | 0.2185 +/- 0.1451 |
| reference/student/joint_full | teacher | 0.8411 +/- 0.0031 | 0.6810 +/- 0.0079 | 0.8944 +/- 0.0056 | 0.6203 +/- 0.0842 | 0.0912 +/- 0.0534 | 0.0429 +/- 0.0204 |
| reference/student/joint_full | equal | 0.8483 +/- 0.0103 | 0.7006 +/- 0.0301 | 0.9025 +/- 0.0066 | 0.5805 +/- 0.0533 | 0.1181 +/- 0.0376 | 0.0719 +/- 0.0251 |
| reference/student/student_ce | primary | 0.8392 +/- 0.0047 | 0.6781 +/- 0.0047 | 0.8994 +/- 0.0092 | 0.8228 +/- 0.3283 | 0.1273 +/- 0.0174 | 0.0617 +/- 0.0203 |
| t2indep/teacher2_indep | primary | 0.8320 +/- 0.0155 | 0.6786 +/- 0.0366 | 0.8900 +/- 0.0054 | 1.0619 +/- 0.1038 | 0.1312 +/- 0.0089 | 0.0563 +/- 0.0093 |
| t2indep/student/joint_full | primary | 0.8320 +/- 0.0201 | 0.6816 +/- 0.0351 | 0.8897 +/- 0.0100 | 1.0518 +/- 0.1095 | 0.1370 +/- 0.0065 | 0.0414 +/- 0.0069 |
| t2indep/student/joint_full | student | 0.0845 +/- 0.0283 | 0.0310 +/- 0.0098 | 0.0000 +/- 0.0000 | 2.3347 +/- 0.9935 | 0.4234 +/- 0.2969 | 0.1643 +/- 0.0218 |
| t2indep/student/joint_full | teacher | 0.8283 +/- 0.0263 | 0.6743 +/- 0.0499 | 0.8877 +/- 0.0112 | 1.0513 +/- 0.1061 | 0.1357 +/- 0.0128 | 0.0490 +/- 0.0134 |
| t2indep/student/joint_full | equal | 0.8338 +/- 0.0109 | 0.6888 +/- 0.0183 | 0.8887 +/- 0.0066 | 0.5954 +/- 0.0298 | 0.0899 +/- 0.0061 | 0.0503 +/- 0.0105 |
| nosn/teacher2 | primary | 0.8156 +/- 0.0079 | 0.6419 +/- 0.0139 | 0.8907 +/- 0.0100 | 0.7169 +/- 0.0443 | 0.1342 +/- 0.0083 | 0.0682 +/- 0.0079 |
| nosn/teacher2 | teacher1 | 0.6839 +/- 0.0152 | 0.3914 +/- 0.0538 | 0.6658 +/- 0.0438 | 0.9121 +/- 0.0021 | 0.1058 +/- 0.0149 | 0.0901 +/- 0.0243 |
| nosn/teacher2 | teacher2 | 0.8156 +/- 0.0079 | 0.6419 +/- 0.0139 | 0.8907 +/- 0.0100 | 1.0585 +/- 0.1696 | 0.1465 +/- 0.0049 | 0.0528 +/- 0.0042 |
| nosn/teacher2 | teacher_raw_average | 0.8084 +/- 0.0110 | 0.6340 +/- 0.0186 | 0.8846 +/- 0.0044 | 0.6559 +/- 0.0771 | 0.1131 +/- 0.0084 | 0.0336 +/- 0.0113 |
| nosn/teacher2 | teacher_probability_average | 0.8111 +/- 0.0110 | 0.6365 +/- 0.0237 | 0.8845 +/- 0.0062 | 0.6085 +/- 0.0215 | 0.0629 +/- 0.0124 | 0.0430 +/- 0.0114 |
| nosn/student/joint_full | primary | 0.8156 +/- 0.0088 | 0.6522 +/- 0.0170 | 0.8861 +/- 0.0142 | 0.7460 +/- 0.0500 | 0.1357 +/- 0.0119 | 0.0561 +/- 0.0097 |
| nosn/student/joint_full | student | 0.0945 +/- 0.0110 | 0.0345 +/- 0.0037 | 0.0000 +/- 0.0000 | 2.3936 +/- 0.3672 | 0.5178 +/- 0.0671 | 0.1225 +/- 0.0092 |
| nosn/student/joint_full | teacher | 0.8165 +/- 0.0042 | 0.6540 +/- 0.0131 | 0.8914 +/- 0.0114 | 0.7360 +/- 0.0492 | 0.1383 +/- 0.0117 | 0.0684 +/- 0.0065 |
| nosn/student/joint_full | equal | 0.8138 +/- 0.0129 | 0.6470 +/- 0.0210 | 0.8807 +/- 0.0208 | 0.6444 +/- 0.0238 | 0.0974 +/- 0.0152 | 0.0407 +/- 0.0133 |
| aug/teacher1 | primary | 0.6776 +/- 0.0394 | 0.4191 +/- 0.0623 | 0.7141 +/- 0.0350 | 0.9431 +/- 0.0966 | 0.1396 +/- 0.0992 | 0.0993 +/- 0.0351 |
| aug/teacher2 | primary | 0.8038 +/- 0.0357 | 0.6596 +/- 0.0550 | 0.8970 +/- 0.0142 | 0.6070 +/- 0.1030 | 0.1014 +/- 0.0509 | 0.0667 +/- 0.0201 |
| aug/teacher2 | teacher1 | 0.7039 +/- 0.0227 | 0.4505 +/- 0.0972 | 0.7286 +/- 0.0262 | 0.8734 +/- 0.0807 | 0.0973 +/- 0.0721 | 0.0841 +/- 0.0465 |
| aug/teacher2 | teacher2 | 0.7993 +/- 0.0422 | 0.6591 +/- 0.0618 | 0.8886 +/- 0.0289 | 0.6664 +/- 0.1380 | 0.0940 +/- 0.0431 | 0.0551 +/- 0.0172 |
| aug/teacher2 | teacher_raw_average | 0.8102 +/- 0.0363 | 0.6705 +/- 0.0584 | 0.9029 +/- 0.0069 | 0.5358 +/- 0.0540 | 0.0660 +/- 0.0296 | 0.0468 +/- 0.0110 |
| aug/teacher2 | teacher_probability_average | 0.8020 +/- 0.0363 | 0.6599 +/- 0.0536 | 0.8956 +/- 0.0070 | 0.5962 +/- 0.0807 | 0.0919 +/- 0.0206 | 0.0481 +/- 0.0060 |
| aug/student/joint_full | primary | 0.8247 +/- 0.0126 | 0.6901 +/- 0.0306 | 0.9055 +/- 0.0041 | 0.5306 +/- 0.0782 | 0.0638 +/- 0.0537 | 0.0496 +/- 0.0208 |
| aug/student/joint_full | student | 0.3070 +/- 0.4003 | 0.2372 +/- 0.3621 | 0.2811 +/- 0.4869 | 1.5415 +/- 0.6012 | 0.2194 +/- 0.0488 | 0.1608 +/- 0.0450 |
| aug/student/joint_full | teacher | 0.8056 +/- 0.0435 | 0.6577 +/- 0.0636 | 0.8896 +/- 0.0320 | 0.6046 +/- 0.1091 | 0.0889 +/- 0.0539 | 0.0440 +/- 0.0280 |
| aug/student/joint_full | equal | 0.8283 +/- 0.0166 | 0.6975 +/- 0.0318 | 0.9084 +/- 0.0018 | 0.5826 +/- 0.0802 | 0.1264 +/- 0.0914 | 0.0369 +/- 0.0109 |
| aug/student/student_ce | primary | 0.8483 +/- 0.0220 | 0.7247 +/- 0.0191 | 0.9229 +/- 0.0115 | 0.4939 +/- 0.1477 | 0.0644 +/- 0.0351 | 0.0293 +/- 0.0034 |
| fusion/fusion/historical | primary | 0.8147 +/- 0.0170 | 0.6448 +/- 0.0346 | 0.8949 +/- 0.0171 | 1.1352 +/- 0.3644 | 0.1602 +/- 0.0233 | 0.0569 +/- 0.0130 |
| fusion/fusion/matched | primary | 0.8238 +/- 0.0134 | 0.6540 +/- 0.0018 | 0.8810 +/- 0.0140 | 0.8930 +/- 0.2447 | 0.1268 +/- 0.0253 | 0.0479 +/- 0.0039 |

## Paired QWK differences (left - right)

| Contrast | Mean difference | 95% interval |
|---|---|---|
| reference/gated_vs_ce | +0.0003 | [-0.0188, +0.0195] |
| reference/equal_vs_ce | +0.0031 | [-0.0152, +0.0215] |
| reference/gated_vs_equal | -0.0028 | [-0.0093, +0.0037] |
| reference/teacher_gate_vs_ce | -0.0050 | [-0.0245, +0.0147] |
| teacher_level/gate_ensemble_vs_joint_t2_alone | -0.0004 | [-0.0036, +0.0024] |
| teacher_level/independent_t2_vs_gate_ensemble | -0.0068 | [-0.0245, +0.0097] |
| teacher_level/independent_t2_vs_joint_t2_alone | -0.0073 | [-0.0255, +0.0099] |
| t2indep/gated_vs_reference_gated | -0.0099 | [-0.0271, +0.0076] |
| t2indep/equal_vs_reference_equal | -0.0137 | [-0.0305, +0.0029] |
| t2indep/gated_vs_ce | -0.0097 | [-0.0293, +0.0103] |
| nosn/teacher_ensemble_vs_reference | -0.0062 | [-0.0211, +0.0086] |
| nosn/t2_alone_vs_reference | -0.0066 | [-0.0215, +0.0081] |
| nosn/gated_vs_reference_gated | -0.0135 | [-0.0268, -0.0004] |
| nosn/equal_vs_reference_equal | -0.0218 | [-0.0352, -0.0093] |
| aug/teacher1_vs_reference | +0.0328 | [-0.0047, +0.0694] |
| aug/teacher_ensemble_vs_reference | +0.0001 | [-0.0229, +0.0246] |
| aug/gated_vs_reference_gated | +0.0059 | [-0.0153, +0.0288] |
| aug/equal_vs_reference_equal | +0.0059 | [-0.0141, +0.0280] |
| aug/ce_vs_reference_ce | +0.0235 | [+0.0043, +0.0438] |
| aug/gated_vs_aug_ce | -0.0174 | [-0.0309, -0.0045] |
| fusion/historical_vs_reference_gated | -0.0047 | [-0.0228, +0.0128] |
| fusion/matched_vs_reference_gated | -0.0187 | [-0.0353, -0.0038] |
| fusion/matched_vs_reference_ce | -0.0184 | [-0.0395, +0.0026] |
| fusion/historical_vs_matched | +0.0139 | [-0.0067, +0.0353] |

## Round-1 label budgets (gated / equal minus CE)

- fraction_0.1: gated -0.0241 [-0.0567, +0.0067]; equal -0.0430 [-0.0735, -0.0136]
- fraction_0.25: gated -0.0042 [-0.0288, +0.0196]; equal -0.1713 [-0.2094, -0.1319]
- fraction_0.5: gated -0.0162 [-0.0345, +0.0007]; equal -0.0301 [-0.0544, -0.0080]
- fraction_1.0: gated +0.0003 [-0.0188, +0.0195]; equal +0.0031 [-0.0152, +0.0215]
