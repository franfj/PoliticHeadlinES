"""Politic post-hoc error analysis: produce a JSON of suspicious cases and
distill failure-mode statistics for the paper.

For each test article we emit:
  - article_body (truncated)
  - our top-3 ranking
  - candidate_titles (all 10)
  - propn_count per top-3 candidate (specificity proxy)
  - has_paraphrase_swap (does some pair of titles share most tokens but swap one entity)
  - confidence_proxy (gap between top-1 and top-2 specificity)

Also computes aggregate stats useful in the paper:
  - distribution of #candidates per article that share >=80% tokens with top-1
    (paraphrase-swap density)
  - fraction of articles where top-1 is the highest-specificity candidate
"""
import argparse, json, re
from pathlib import Path
from collections import Counter

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


PROPN = re.compile(r"\b[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+){0,2}\b")
SKIP = {"El", "La", "Los", "Las", "Un", "Una", "Unos", "Unas", "Y", "O", "U", "En", "Con", "Sin",
        "Por", "Para", "De", "Del", "A", "Al", "Su", "Sus"}


def count_propn(text: str) -> tuple[int, list[str]]:
    found = []
    for m in PROPN.findall(text or ""):
        head = m.split()[0]
        if head in SKIP:
            rest = m.split()[1:]
            if not rest:
                continue
            m = " ".join(rest)
            head = rest[0]
        if len(m) >= 4 and head not in SKIP:
            found.append(m)
    return len(found), found


def token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(RESULTS_DIR / "test_predictions_gbm_fusion_a03.csv"))
    ap.add_argument("--out_json", default=str(RESULTS_DIR / "error_analysis_candidates.json"))
    ap.add_argument("--out_summary", default=str(RESULTS_DIR / "error_analysis_summary.json"))
    ap.add_argument("--max_emit", type=int, default=200, help="Max suspicious cases to emit")
    args = ap.parse_args()

    test = pd.read_csv(DATA_DIR / "test_public" / "test_public.csv")
    preds = pd.read_csv(args.in_csv)
    pmap = {r["id"]: str(r["task_1"]).split() for _, r in preds.iterrows()}

    cases = []
    paraphrase_density_per_article = []
    top1_is_most_specific = 0
    n_processed = 0
    for _, r in test.iterrows():
        aid = r["id"]
        ranking = pmap.get(aid)
        if not ranking or len(ranking) < 3:
            continue
        n_processed += 1

        all_titles = {f"t{i}": str(r.get(f"title_{i}", "") or "") for i in range(1, 11)}

        # specificity per ranking position
        scored = []
        for tid in ranking:
            n_pn, pns = count_propn(all_titles[tid])
            scored.append((tid, n_pn, pns))

        # Most specific candidate
        max_specificity = max(scored, key=lambda x: x[1])
        if scored[0][1] == max_specificity[1]:
            top1_is_most_specific += 1

        # Paraphrase density: how many candidates share >= 0.65 token jaccard with top-1?
        top1_text = all_titles[ranking[0]]
        paraphrase_count = sum(1 for tid in ranking[1:]
                               if token_jaccard(top1_text, all_titles[tid]) >= 0.65)
        paraphrase_density_per_article.append(paraphrase_count)

        # Suspicious flag: top-1 has STRICTLY LESS proper-noun count than at least
        # one of the next two candidates AND that candidate is a paraphrase variant.
        top1_propn = scored[0][1]
        suspicious = False
        better_candidate = None
        for cand_tid, cand_propn, _ in scored[1:3]:
            if (cand_propn > top1_propn
                and token_jaccard(top1_text, all_titles[cand_tid]) >= 0.55):
                suspicious = True
                better_candidate = cand_tid
                break

        if suspicious and len(cases) < args.max_emit:
            cases.append({
                "id": aid,
                "body_preview": r["article_body"][:300],
                "image_hash": r["image_hash"],
                "our_top3": [tid for tid, *_ in scored[:3]],
                "our_top1_text": top1_text,
                "candidate_titles": all_titles,
                "propn_count": {tid: n for tid, n, _ in scored[:3]},
                "propn_lists": {tid: pns for tid, _, pns in scored[:3]},
                "paraphrase_count_top10": paraphrase_count,
                "suggested_alternative": better_candidate,
            })

    # Save the suspicious cases
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(cases)} suspicious cases to {args.out_json}")

    # Summary stats
    summary = {
        "n_test_articles": n_processed,
        "top1_is_most_specific_among_all_10": {
            "count": top1_is_most_specific,
            "fraction": top1_is_most_specific / max(1, n_processed),
        },
        "paraphrase_density_per_article": dict(Counter(paraphrase_density_per_article)),
        "paraphrase_density_mean": (
            sum(paraphrase_density_per_article) / max(1, len(paraphrase_density_per_article))
        ),
        "suspicious_case_count_emitted": len(cases),
    }
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary -> {args.out_summary}:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
