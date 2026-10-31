"""CPU-only recomputation of final-model (TS_ConvNeXtTiny_Residual) test metrics
from the confusion matrices stored in results/lowdata_hybrid_distill/*/seed_*/test_metrics.json.

No model, GPU or image is used. Per-sample predictions of the final model are NOT
available, so only confusion-matrix based metrics can be derived (no ECE/NLL/Brier,
no paired tests). The confidence intervals below are a multinomial resampling of the
confusion-matrix cells (a parametric bootstrap of the test set), not a paired bootstrap.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "lowdata_hybrid_distill" / "TS_ConvNeXtTiny_Residual"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
SEEDS = [42, 52, 62]
NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
N_BOOT = 2000
RNG = np.random.default_rng(20261010)


def qwk(cm):
    k = cm.shape[0]
    idx = np.arange(k)
    w = (idx[:, None] - idx[None, :]) ** 2 / (k - 1) ** 2
    n = cm.sum()
    expected = np.outer(cm.sum(1), cm.sum(0)) / n
    return 1.0 - (w * cm).sum() / (w * expected).sum()


def basic(cm):
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    pred = cm.sum(0).astype(float)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1, sup


def referable(cm):
    # referable DR = grade >= 2 (moderate or worse)
    tn = cm[:2, :2].sum()
    fp = cm[:2, 2:].sum()
    fn = cm[2:, :2].sum()
    tp = cm[2:, 2:].sum()
    return dict(sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                ppv=tp / (tp + fp), npv=tn / (tn + fn),
                TP=int(tp), FN=int(fn), TN=int(tn), FP=int(fp))


def summary(cm):
    prec, rec, f1, sup = basic(cm)
    r = referable(cm)
    return dict(accuracy=np.trace(cm) / cm.sum(), qwk=qwk(cm), macro_f1=f1.mean(),
                macro_recall=rec.mean(), ref_sensitivity=r["sensitivity"],
                ref_specificity=r["specificity"])


def main():
    rows, per_class, ref_rows, ci_rows = [], [], [], []
    for s in SEEDS:
        d = json.loads((SRC / f"seed_{s}" / "test_metrics.json").read_text())
        cm = np.array(d["confusion_matrix"], dtype=int)
        assert cm.sum() == 367, cm.sum()
        m = summary(cm)
        rows.append(dict(seed=s, n=int(cm.sum()),
                         acc_recomputed=m["accuracy"], acc_recorded=d["accuracy"],
                         qwk_recomputed=m["qwk"], qwk_recorded=d["qwk"],
                         f1_macro_recomputed=m["macro_f1"], f1_macro_recorded=d["f1_macro"],
                         max_abs_diff=max(abs(m["accuracy"] - d["accuracy"]),
                                          abs(m["qwk"] - d["qwk"]),
                                          abs(m["macro_f1"] - d["f1_macro"]))))
        prec, rec, f1, sup = basic(cm)
        for i, n in enumerate(NAMES):
            per_class.append(dict(seed=s, grade=i, name=n, support=int(sup[i]),
                                  precision=prec[i], recall=rec[i], f1=f1[i]))
        ref_rows.append(dict(seed=s, **referable(cm)))

        probs = (cm / cm.sum()).ravel()
        keys = ["accuracy", "qwk", "macro_f1", "ref_sensitivity", "ref_specificity"]
        boots = {k: [] for k in keys}
        for _ in range(N_BOOT):
            b = RNG.multinomial(cm.sum(), probs).reshape(cm.shape)
            sb = summary(b)
            for k in keys:
                boots[k].append(sb[k])
        for k in keys:
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
            ci_rows.append(dict(seed=s, metric=k, point=m[k], ci_low=lo, ci_high=hi))

    pd.DataFrame(rows).to_csv(OUT / "per_seed_recompute_vs_recorded.csv", index=False)
    pc = pd.DataFrame(per_class)
    pc.to_csv(OUT / "per_seed_per_class_metrics.csv", index=False)
    pc.groupby(["grade", "name"])[["precision", "recall", "f1"]].agg(["mean", "std"]).to_csv(
        OUT / "per_class_mean_std_over_seeds.csv")
    rf = pd.DataFrame(ref_rows)
    rf.to_csv(OUT / "referable_dr_per_seed.csv", index=False)
    rf.drop(columns="seed").agg(["mean", "std"]).to_csv(OUT / "referable_dr_mean_std.csv")
    pd.DataFrame(ci_rows).to_csv(OUT / "cell_bootstrap_ci_per_seed.csv", index=False)
    print(pd.DataFrame(rows).round(6).to_string())
    print(rf.round(4).to_string())
    print(rf.drop(columns="seed").agg(["mean", "std"]).round(4).to_string())
    print(pc.groupby("name")[["precision", "recall", "f1"]].agg(["mean", "std"]).round(3).to_string())
    print(pd.DataFrame(ci_rows).round(4).to_string())


if __name__ == "__main__":
    main()
