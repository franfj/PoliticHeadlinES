"""v6: Borda-count ensemble of existing test predictions.

For each test article, each candidate title gets a position vote from
each predictor. Final ranking = sort by sum of Borda points.

Borda score for a title in a single ranking:
   pos 1 (best) -> 10 points, pos 2 -> 9, ..., pos 10 -> 1.

Predictors used (filtered to those that have a single-column 'y_pred'
or 'task_1'/'task_2' format):
  - test_predictions_gbm_fusion_a03.csv (best so far)
  - test_predictions_gbm.csv
  - test_predictions_gpt54.csv
  - test_predictions_ensemble_xlm-roberta-base.csv
  - test_predictions_v19_flip.csv

We weight each predictor uniformly. Future: weight by dev Kendall.
"""
import csv, sys
import pandas as pd
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("tasks/politicheadlines/results")

PREDICTORS = [
    "test_predictions_gbm_fusion_a03.csv",
    "test_predictions_gbm.csv",
    "test_predictions_gpt54.csv",
    "test_predictions_ensemble_xlm-roberta-base.csv",
    "test_predictions_v19_flip.csv",
]


def parse_ranking(s: str) -> list[str]:
    return s.strip().split()


def load_pred(path: Path) -> dict[str, list[str]]:
    """id -> ranking list (uses task_1 if present, else y_pred)."""
    df = pd.read_csv(path)
    if "task_1" in df.columns:
        col = "task_1"
    elif "y_pred" in df.columns:
        col = "y_pred"
    else:
        raise ValueError(f"{path}: no y_pred or task_1 column")
    return {row["id"]: parse_ranking(row[col]) for _, row in df.iterrows()}


def main():
    preds = {}
    for fname in PREDICTORS:
        p = RESULTS_DIR / fname
        if p.exists():
            preds[fname] = load_pred(p)
            print(f"loaded {fname}: {len(preds[fname])} rows")
    # Identify common ids
    common = set.intersection(*(set(d) for d in preds.values()))
    print(f"common ids: {len(common)}")

    out_rows = []
    for aid in sorted(common):
        # Borda points per title id
        score = defaultdict(int)
        for d in preds.values():
            ranking = d[aid]
            n = len(ranking)
            for pos, tid in enumerate(ranking):
                score[tid] += (n - pos)
        # Sort: highest borda first; tie-break by gbm_fusion order
        gbm_order = preds.get("test_predictions_gbm_fusion_a03.csv", {}).get(aid, [])
        gbm_pos = {t: i for i, t in enumerate(gbm_order)}
        merged = sorted(score.keys(), key=lambda t: (-score[t], gbm_pos.get(t, 999)))
        out_rows.append({"id": aid, "task_1": " ".join(merged), "task_2": " ".join(merged)})

    out_df = pd.DataFrame(out_rows)
    out_path = RESULTS_DIR / "test_predictions_v6_borda.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(out_df)} rows")

    # Build submission zip
    import zipfile, shutil
    sub_dir = Path("release/politic")
    sub_dir.mkdir(exist_ok=True)
    sub_csv = sub_dir / "results.csv"
    shutil.copy(out_path, sub_csv)
    sub_zip = sub_dir / "v6_borda.zip"
    with zipfile.ZipFile(sub_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_csv, "results.csv")
    print(f"Saved {sub_zip}")


if __name__ == "__main__":
    main()
