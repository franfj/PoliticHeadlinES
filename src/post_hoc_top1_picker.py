"""Post-hoc top-1 picker.

Given:
  - Existing GPT-5.4 ranking per article (gives us the top-3 candidates)
  - Image descriptions per article (from VLM run)
  - The 10 candidate titles per article

For each article:
  1. Take the GPT-5.4 ranking's top-3 candidates.
  2. From the image description, extract proper-noun entities
     (capitalised tokens, person/place names, institutions).
  3. Score each top-3 candidate by entity-overlap with the description.
  4. If one candidate has STRICTLY MORE entity matches than the others,
     it becomes the new top-1. The remaining two slide down by one.
  5. Otherwise keep GPT-5.4's order (no change).

Conservative: only changes when there is unambiguous image evidence.
"""
import argparse, json, re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


# Capitalised proper-noun extractor; very rough but no spaCy needed.
PROPN_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]+){0,3}\b")
# Stopwords / sentence-starters that get capitalised but are not proper nouns.
SKIP_TOKENS = {
    "El", "La", "Los", "Las", "Un", "Una", "Unos", "Unas", "Y", "O", "U",
    "En", "Con", "Sin", "Por", "Para", "De", "Del", "A", "Al",
    "Su", "Sus", "Lo", "Le", "Les", "Se",
    "The", "An", "And", "Or", "In", "On", "Of", "For", "By", "With", "From",
    "Imagen", "Foto", "Fotografia", "Fotografía", "Fotograma",
    "Aparece", "Aparecen", "Muestra", "Muestran", "Vemos", "Vea", "Veo",
    "Esta", "Este", "Estos", "Estas", "Ese", "Esa", "Aquel",
}


def extract_entities(text: str) -> set[str]:
    if not text:
        return set()
    matches = PROPN_RE.findall(text)
    out = set()
    for m in matches:
        # First token of a multi-word match
        head = m.split()[0]
        if head in SKIP_TOKENS:
            # Drop the leading filler, keep the rest
            rest = m.split()[1:]
            if rest:
                m = " ".join(rest)
                head = rest[0]
            else:
                continue
        # Skip very short entities and pure stopwords
        if len(m) < 4 or head in SKIP_TOKENS:
            continue
        out.add(m)
    return out


def title_text_lower(t: str) -> str:
    return (t or "").lower()


def score_candidate_against_entities(candidate_text: str, entities: set[str]) -> int:
    """Count how many entity strings appear in the candidate text."""
    cand_low = candidate_text.lower()
    score = 0
    for ent in entities:
        # Match as a whole word substring (case-insensitive)
        if ent.lower() in cand_low:
            score += 1
    return score


def parse_ranking(s: str) -> list[str]:
    return s.strip().split()


def rerank_one(article_row: pd.Series, current_ranking: list[str], desc: str | None) -> tuple[list[str], bool]:
    """Returns (new_ranking, changed_flag)."""
    if not desc or not current_ranking or len(current_ranking) < 2:
        return current_ranking, False
    entities = extract_entities(desc)
    if not entities:
        return current_ranking, False

    top3 = current_ranking[:3]
    scores = []
    for tid in top3:
        idx = int(tid.replace("t", ""))
        title = article_row.get(f"title_{idx}", "") or ""
        scores.append((tid, score_candidate_against_entities(title, entities)))

    # Find the best by entity overlap. Require strict majority over the current top-1.
    best_tid, best_score = max(scores, key=lambda x: x[1])
    cur_top1_score = scores[0][1]
    if best_tid == top3[0]:
        return current_ranking, False  # current top-1 already wins
    if best_score < cur_top1_score + 1:
        # Tie or only marginal — stay safe, don't override.
        return current_ranking, False

    # Promote best_tid to position 0; demote others one slot.
    new_top3 = [best_tid] + [t for t in top3 if t != best_tid]
    new_ranking = new_top3 + current_ranking[3:]
    return new_ranking, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=str(RESULTS_DIR / "test_predictions_gbm_fusion_a03.csv"))
    ap.add_argument("--descriptions", required=True,
                    help="JSON file mapping image_hash -> description")
    ap.add_argument("--out_csv", default=str(RESULTS_DIR / "test_predictions_v7_post_hoc.csv"))
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
    print(f"v7 post-hoc: {len(out)} rows, top-1 changed in {n_changed} ({n_changed/len(out):.1%})")
    print(f"Saved {args.out_csv}")

    # Build zip
    import zipfile, shutil
    sub_dir = Path("release/politic")
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_csv = sub_dir / "results.csv"
    shutil.copy(args.out_csv, sub_csv)
    zip_name = Path(args.out_csv).stem.replace("test_predictions_", "") + ".zip"
    sub_zip = sub_dir / zip_name
    with zipfile.ZipFile(sub_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_csv, "results.csv")
    print(f"Saved {sub_zip}")


if __name__ == "__main__":
    main()
