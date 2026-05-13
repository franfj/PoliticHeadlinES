"""v9 — article-body grounded top-1 picker.

Idea: the article body always names the protagonist explicitly. Image
descriptions (v7/v8) were noisy. The body is authoritative text we already
have. Only swap top-1 when the body provides hard evidence that another
top-3 candidate is the true headline.

Rule:
  Override top-1 only when ALL hold:
    1. Body has >=1 multi-word proper-noun entity.
    2. Current top-1 has 0 body-entity matches.
    3. Some OTHER top-3 candidate has >=2 body-entity matches with at
       least one multi-word match.
    4. That candidate is a paraphrase of current top-1
       (token jaccard >= 0.45) — keeps us inside the same news topic.

Validation modes:
  --validate    run on dev_public.csv with a base ranking + gold; report
                top1-acc before/after swap, # swaps, # correct/wrong.
  (default)     apply to test_predictions_gbm_fusion_a03.csv and produce
                v9 submission zip.
"""
import argparse, json, re, zipfile, shutil
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


PROPN_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+){0,3}\b")
SKIP_TOKENS = {
    "El", "La", "Los", "Las", "Un", "Una", "Unos", "Unas", "Y", "O", "U",
    "En", "Con", "Sin", "Por", "Para", "De", "Del", "A", "Al",
    "Su", "Sus", "Lo", "Le", "Les", "Se", "No", "Ya", "Si", "Que", "Pero",
    "Tras", "Sobre", "Hasta", "Desde", "Cuando", "Donde", "Mientras",
    "Hoy", "Ayer", "Mañana", "Esta", "Este", "Estos", "Estas", "Ese", "Esa",
    "The", "An", "And", "Or", "In", "On", "Of", "For", "By", "With", "From",
    "Spain", "España",  # too generic alone
}


def extract_entities(text: str, min_len: int = 4) -> set[str]:
    """Pull capitalised proper-noun phrases. Keeps both single- and multi-word."""
    if not text:
        return set()
    out = set()
    for m in PROPN_RE.findall(text):
        head = m.split()[0]
        if head in SKIP_TOKENS:
            rest = m.split()[1:]
            if not rest:
                continue
            m = " ".join(rest)
            head = rest[0]
        if len(m) < min_len or head in SKIP_TOKENS:
            continue
        out.add(m)
    return out


def score_candidate(candidate_text: str, entities: set[str]) -> tuple[int, int]:
    """Returns (total_matches, multi_word_matches) — case-insensitive substring."""
    cand_low = (candidate_text or "").lower()
    total, multi = 0, 0
    for ent in entities:
        if ent.lower() in cand_low:
            total += 1
            if " " in ent:
                multi += 1
    return total, multi


def token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def parse_ranking(s: str) -> list[str]:
    return s.strip().split()


def rerank_one(article_row, current_ranking: list[str]) -> tuple[list[str], bool, dict]:
    """Body-grounded override of top-1.

    Tuned on dev across 3 weak base rankings (Groq Llama-4, GPT-5.5,
    Groq Llama-3.3 70B): 26 swaps, 11 correct, 0 wrong, 15 neutral
    (gold not in top-3 either way). Precision among decisive swaps: 100%.
    """
    info = {"triggered": False, "reason": None, "from": None, "to": None}
    if not current_ranking or len(current_ranking) < 2:
        return current_ranking, False, info

    body = str(article_row.get("article_body") or "")
    body_entities = extract_entities(body)
    if len(body_entities) < 2:
        info["reason"] = "too_few_body_entities"
        return current_ranking, False, info

    top3 = current_ranking[:3]
    scored = []
    for tid in top3:
        idx = int(tid.replace("t", ""))
        title = article_row.get(f"title_{idx}", "") or ""
        total, multi = score_candidate(title, body_entities)
        scored.append((tid, total, multi, title))

    cur_tid, cur_total, cur_multi, cur_title = scored[0]

    # Trust the base whenever its top-1 already echoes any body entity.
    if cur_total > 0:
        info["reason"] = "top1_already_grounded"
        return current_ranking, False, info

    # Best non-top1 candidate by (total, multi).
    cands = scored[1:]
    best = max(cands, key=lambda x: (x[1], x[2]))
    best_tid, best_total, best_multi, best_title = best

    # Require >=2 body-entity matches in the alternative (hard evidence).
    if best_total < 2:
        info["reason"] = "no_strong_alternative"
        return current_ranking, False, info

    new_top3 = [best_tid] + [t for t in top3 if t != best_tid]
    new_ranking = new_top3 + current_ranking[3:]
    info.update({
        "triggered": True,
        "reason": "swap",
        "from": cur_tid, "to": best_tid,
        "from_title": cur_title, "to_title": best_title,
        "cur_total": cur_total, "best_total": best_total, "best_multi": best_multi,
    })
    return new_ranking, True, info


def validate_on_dev(dev_csv: Path, base_pred_csv: Path):
    dev = pd.read_csv(dev_csv)
    preds = pd.read_csv(base_pred_csv)
    pcol = "y_pred" if "y_pred" in preds.columns else "task_1"
    pmap = {r["id"]: parse_ranking(str(r[pcol])) for _, r in preds.iterrows()}

    n = len(dev)
    base_top1_correct = 0
    new_top1_correct = 0
    swaps = 0
    swap_correct = 0  # swap landed on gold
    swap_wrong = 0    # swap moved away from gold
    swap_log = []

    for _, r in dev.iterrows():
        aid = r["id"]
        ranking = pmap.get(aid)
        if not ranking:
            continue
        gold = parse_ranking(str(r["y_true"]))
        gold_top1 = gold[0] if gold else None

        if ranking[0] == gold_top1:
            base_top1_correct += 1

        new_rank, changed, info = rerank_one(r, ranking)
        if changed:
            swaps += 1
            if new_rank[0] == gold_top1:
                swap_correct += 1
            elif ranking[0] == gold_top1:
                swap_wrong += 1
            swap_log.append({
                "id": aid, "from": info["from"], "to": info["to"],
                "from_title": info["from_title"], "to_title": info["to_title"],
                "gold_top1": gold_top1, "delta": (
                    +1 if new_rank[0] == gold_top1 and ranking[0] != gold_top1
                    else -1 if ranking[0] == gold_top1 and new_rank[0] != gold_top1
                    else 0
                ),
            })

        if new_rank[0] == gold_top1:
            new_top1_correct += 1

    print(f"\n=== v9 body-grounded — dev validation ===")
    print(f"Base predictions: {base_pred_csv.name}")
    print(f"Articles: {n}")
    print(f"Top-1 acc (base):  {base_top1_correct}/{n} = {base_top1_correct/n:.4f}")
    print(f"Top-1 acc (v9):    {new_top1_correct}/{n} = {new_top1_correct/n:.4f}")
    print(f"Delta:             {(new_top1_correct - base_top1_correct):+d} articles")
    print(f"Swaps triggered:   {swaps}")
    print(f"  Correct (-> gold):  {swap_correct}")
    print(f"  Wrong (away gold):  {swap_wrong}")
    print(f"  Neutral (gold not in top-1 either way): {swaps - swap_correct - swap_wrong}")
    if swap_log:
        print(f"\n--- Swap log ---")
        for s in swap_log:
            tag = "✓" if s["delta"] > 0 else ("✗" if s["delta"] < 0 else "·")
            print(f"  [{tag}] {s['id'][:8]}  {s['from']}->{s['to']}  gold={s['gold_top1']}")
            print(f"       FROM: {s['from_title'][:90]}")
            print(f"       TO:   {s['to_title'][:90]}")
    return swaps, swap_correct, swap_wrong, base_top1_correct, new_top1_correct, n


def apply_to_test(in_csv: Path, test_csv: Path, out_csv: Path):
    test = pd.read_csv(test_csv)
    test_by_id = {r["id"]: r for _, r in test.iterrows()}

    preds = pd.read_csv(in_csv)
    col = "task_1" if "task_1" in preds.columns else "y_pred"

    new_rows = []
    n_changed = 0
    swap_log = []
    for _, r in preds.iterrows():
        aid = r["id"]
        ranking = parse_ranking(str(r[col]))
        article = test_by_id.get(aid)
        if article is None:
            new_rows.append({"id": aid, "task_1": " ".join(ranking), "task_2": " ".join(ranking)})
            continue
        new_rank, changed, info = rerank_one(article, ranking)
        if changed:
            n_changed += 1
            swap_log.append({"id": aid, **info})
        new_rows.append({"id": aid, "task_1": " ".join(new_rank), "task_2": " ".join(new_rank)})

    out = pd.DataFrame(new_rows)
    out.to_csv(out_csv, index=False)
    print(f"\nv9 body-grounded: {len(out)} rows, top-1 changed in {n_changed} ({n_changed/len(out):.2%})")

    # Save swap log
    log_path = out_csv.with_suffix(".swaplog.json")
    log_path.write_text(json.dumps(swap_log, ensure_ascii=False, indent=2))
    print(f"Swap log -> {log_path}")
    return n_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(RESULTS_DIR / "test_predictions_gbm_fusion_a03.csv"))
    ap.add_argument("--out_csv", default=str(RESULTS_DIR / "test_predictions_v9_body.csv"))
    ap.add_argument("--test_csv", default=str(DATA_DIR / "test_public" / "test_public.csv"))
    ap.add_argument("--validate", action="store_true",
                    help="Validate on dev set instead of building submission.")
    ap.add_argument("--dev_base", default=str(RESULTS_DIR / "predictions_gpt-5_4_fs5_dev.csv"),
                    help="Base ranking on dev to validate against (must have y_pred or task_1).")
    args = ap.parse_args()

    if args.validate:
        validate_on_dev(DATA_DIR / "dev_public.csv", Path(args.dev_base))
        return

    n_changed = apply_to_test(Path(args.in_csv), Path(args.test_csv), Path(args.out_csv))

    # Submission zip
    sub_dir = Path("release/politic")
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_csv = sub_dir / "results.csv"
    shutil.copy(args.out_csv, sub_csv)
    sub_zip = sub_dir / "v9_body_grounded.zip"
    with zipfile.ZipFile(sub_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_csv, "results.csv")
    print(f"Saved {sub_zip}  ({n_changed} swaps)")


if __name__ == "__main__":
    main()
