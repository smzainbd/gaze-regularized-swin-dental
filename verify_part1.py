"""
Verify Part 1 pipeline (imports, fold loaders, stats, syntax).
Run until 3 consecutive clean passes: python verify_part1.py
"""
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  OK  {name}")
    except Exception as e:
        FAILURES.append((name, str(e) or repr(e), traceback.format_exc()))
        print(f"  FAIL {name}: {e or repr(e)}")


def test_imports():
    import negative_result_stats  # noqa: F401
    import stage2_train_utils  # noqa: F401
    import stage2_cv  # noqa: F401
    import core_modules  # noqa: F401
    import dental_dataset  # noqa: F401


def test_stats():
    import numpy as np
    from negative_result_stats import (
        bootstrap_ci,
        aggregate_fold_metrics,
        power_analysis_auroc_sample_size,
        summarize_negative_detection_result,
        bootstrap_paired_diff_ci,
    )
    # Larger balanced set so bootstrap rarely hits single-class draws
    rng = np.random.default_rng(42)
    y = np.array([0] * 40 + [1] * 40)
    p1 = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.1, size=80), 0, 1)
    p2 = np.clip(p1 + 0.02, 0, 1)
    ci = bootstrap_ci(y, p1, n_boot=200, seed=42)
    assert "lower" in ci and ci["n"] == 80
    assert np.isfinite(ci["mean"]) and np.isfinite(ci["lower"]) and np.isfinite(ci["upper"])
    assert ci["lower"] <= ci["mean"] <= ci["upper"]

    agg = aggregate_fold_metrics([{"AUROC": 0.6}, {"AUROC": 0.7}])
    assert abs(agg["AUROC"]["mean"] - 0.65) < 1e-9

    pw = power_analysis_auroc_sample_size()
    assert pw["approx_n_per_group"] > 0

    diff = bootstrap_paired_diff_ci(y, p1, p2, n_boot=200, seed=42)
    assert np.isfinite(diff["mean_diff"])

    neg = summarize_negative_detection_result(
        {"AUROC": float(ci["point_estimate"])},
        {"AUROC": float(bootstrap_ci(y, p2, n_boot=50, seed=1)["point_estimate"])},
        y, p1, p2, n_boot=100,
    )
    assert "interpretation" in neg
    assert "power_analysis" in neg
    assert np.isfinite(neg["power_analysis"]["approx_n_per_group"])


def test_fold_loaders():
    from dental_dataset import build_dataloaders

    cfg = json.load(open("config.json"))
    assert int(cfg.get("num_folds", 0)) == 5
    project = ROOT
    ids_seen = set()
    for fold in range(int(cfg.get("num_folds", 5))):
        loaders = build_dataloaders(
            cfg, project, fold=fold, stage2_training=True, image_root=project,
        )
        n_train = len(loaders["tdd_train"].dataset)
        n_val = len(loaders["tdd_val"].dataset)
        n_test = len(loaders["tdd_test"].dataset)
        assert n_train + n_val == 900, f"fold {fold}: train+val != 900 ({n_train}+{n_val})"
        assert n_test == 100, f"fold {fold}: test != 100"
        assert 100 <= n_val <= 200, f"fold {fold}: val size {n_val} unexpected"
        # no overlap train/val
        train_ids = set(loaders["tdd_train"].dataset.df["id"].astype(str))
        val_ids = set(loaders["tdd_val"].dataset.df["id"].astype(str))
        assert train_ids.isdisjoint(val_ids), f"fold {fold}: train/val overlap"
        ids_seen |= val_ids
    assert len(ids_seen) == 900, f"CV val folds must cover all 900 pool images, got {len(ids_seen)}"


def test_subsample():
    import pandas as pd
    from dental_dataset import _subsample_stratified

    df = pd.read_csv("splits/train_split.csv")
    sub = _subsample_stratified(df, 0.5, seed=42)
    assert 0 < len(sub) < len(df)


def test_nb04_json():
    nb = json.load(open("Notebooks/NB04_stage2_train.ipynb", encoding="utf-8"))
    src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "run_part1_negative_pipeline" in src
    assert "num_folds', 5" in src or 'num_folds", 5' in src or "num_folds', 5)" in src
    assert "stage2_part1_negative_report.json" in src


def test_config_part1():
    cfg = json.load(open("config.json"))
    assert cfg["num_folds"] == 5
    assert cfg["stage2_mix_oral"] is False
    assert cfg["run_gaze_ablation"] is True
    assert cfg["eval_test_each_epoch"] is False
    assert "negative_result_bootstrap_n" in cfg


def test_stage2_cv_syntax():
    """Import pipeline entry without loading Swin weights (fast)."""
    import stage2_cv
    assert callable(stage2_cv.run_part1_negative_pipeline)
    assert callable(stage2_cv.train_stage2_fold)
    assert callable(stage2_cv.eval_stage1_baseline)


def main():
    global FAILURES
    FAILURES = []
    print("Part 1 verification loop")
    check("imports", test_imports)
    check("stats", test_stats)
    check("fold_loaders", test_fold_loaders)
    check("subsample", test_subsample)
    check("nb04_json", test_nb04_json)
    check("config_part1", test_config_part1)
    check("stage2_cv_syntax", test_stage2_cv_syntax)

    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for name, err, tb in FAILURES:
            print(f"  - {name}: {err}")
            print(tb)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
