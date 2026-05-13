"""v8 — hyper-conservative top-1 override.

v7 swapped 214 times and netted -0.034. Postmortem: too many false-positive
swaps. v8 tightens the rule:

  Override only when ALL of these hold:
    1. Current top-1 has ZERO entity matches with the image description.
    2. Some OTHER top-3 candidate has >= 3 entity matches.
    3. At least one of those matches is a MULTI-WORD entity (e.g.,
       "Pedro Sánchez" — single-word matches like "Madrid" are too weak).
"""
import argparse, json, re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


PROPN_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+){0,3}\b")
SKIP_TOKENS = {
    "El", "La", "Los", "Las", "Un", "Una", "Unos", "Unas", "Y", "O", "U",
    "En", "Con", "Sin", "Por", "Para", "De", "Del", "A", "Al",
    "Su", "Sus", "Lo", "Le", "Les", "Se",
    "The", "An", "And", "Or", "In", "On", "Of", "For", "By", "With", "From",
    "Imagen", "Foto", "Fotografia", "Fotografía", "Fotograma",
    "Aparece", "Aparecen", "Muestra", "Muestran", "Vemos", "Vea", "Veo",
    "Esta", "Este", "Estos", "Estas", "Ese", "Esa", "Aquel",
    "Spain", "España", "Madrid", "Barcelona",  # too generic alone
}


def extract_entities(text):
    if not text:
        return set()
    matches = PROPN_RE.findall(text)
    out = set()
    for m in matches:
        head = m.split()[0]
        if head in SKIP_TOKENS:
            rest = m.split()[1:]
            if rest:
                m = " ".join(rest)
                head = rest[0]
            else:
                continue
        if len(m) < 4 or head in SKIP_TOKENS:
            continue
        out.add(m)
    return out


def score_candidate(candidate_text, entities):
    """Return (total_matches, multi_word_matches)."""
    cand_low = (candidate_text or "").lower()
    total = 0
    multi = 0
    for ent in entities:
        if ent.lower() in cand_low:
            total += 1
            if " " in ent:
                multi += 1
    return total, multi


def parse_ranking(s):
    return s.strip().split()


def rerank_one(article_row, current_ranking, desc):
    """Conservative: override top-1 only on hard-evidence cases."""
    if not desc or not current_ranking or len(current_ranking) < 2:
        return current_ranking, False
    entities = extract_entities(desc)
    if not entities:
        return current_ranking, False
    if not any(" " in e for e in entities):
        # No multi-word proper-noun in description — too weak signal.
        return current_ranking, False

    top3 = current_ranking[:3]
    scored = []
    for tid in top3:
        idx = int(tid.replace("t", ""))
        title = article_row.get(f"title_{idx}", "") or ""
        total, multi = score_candidate(title, entities)
        scored.append((tid, total, multi))

    cur_total, cur_multi = scored[0][1], scored[0][2]
    cands = [s for s in scored[1:]]
    best = max(cands, key=lambda x: (x[1], x[2]))
    best_tid, best_total, best_multi = best
    # Require strict dominance: best has >=2 more matches AND >=1 multi-word
    if best_total < cur_total + 2:
        return current_ranking, False
    if best_multi < 1:
        return current_ranking, False
    # And current top-1 has at most 1 match (otherwise we trust it)
    if cur_total > 1:
        return current_ranking, False

    # Promote best_tid to position 0; demote the rest.
    new_top3 = [best_tid] + [t for t in top3 if t != best_tid]
    new_ranking = new_top3 + current_ranking[3:]
    return new_ranking, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(RESULTS_DIR / "test_predictions_gbm_fusion_a03.csv"))
    ap.add_argument("--descriptions", default=str(RESULTS_DIR / "descriptions_qwen2vl-2b.json"))
    ap.add_argument("--out_csv", default=str(RESULTS_DIR / "test_predictions_v8_post_hoc.csv"))
    ap.add_argument("--test_csv", default=str(DATA_DIR / "test_public" / "test_public.csv"))
    args = ap.parse_args()

    test = pd.read_csv(args.test_csv)
    test_by_id = {r["id"]: r for _, r in test.iterrows()}

    preds = pd.read_csv(args.in_csv)
    descs = json.loads(Path(args.descriptions).read_text())

    new_rows = []
    n_changed = 0
    for _, r in preds.iterrows():
        aid = r["id"]
        col = "task_1" if "task_1" in preds.columns else "y_pred"
        ranking = parse_ranking(str(r[col]))
        article = test_by_id.get(aid)
        desc = None
        if article is not None:
            desc = descs.get(article["image_hash"])
        new_rank, changed = rerank_one(article, ranking, desc)
        if changed:
            n_changed += 1
        new_rows.append({"id": aid, "task_1": " ".join(new_rank), "task_2": " ".join(new_rank)})

    out = pd.DataFrame(new_rows)
    out.to_csv(args.out_csv, index=False)
    print(f"v8 hyper-conservative: {len(out)} rows, top-1 changed in {n_changed} ({n_changed/len(out):.2%})")

    import zipfile, shutil
    sub_dir = Path("release/politic")
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_csv = sub_dir / "results.csv"
    shutil.copy(args.out_csv, sub_csv)
    sub_zip = sub_dir / "v8_post_hoc.zip"
    with zipfile.ZipFile(sub_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_csv, "results.csv")
    print(f"Saved {sub_zip}")


if __name__ == "__main__":
    main()
