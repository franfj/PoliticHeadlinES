"""
PoliticHeadlinES 2026 — GPT-5.4 headline ranking baseline.

Task: Given an article body (+ optional image), rank 10 candidate titles
      from most to least relevant.

y_true format: "t9 t2 t7 t5 t6 t10 t3 t8 t4 t1"
  -> ordered list from BEST (rank 1) to WORST (rank 10).

Evaluation: Kendall's tau (rank correlation) — standard for ranking tasks.

Run ON SERVER:
  export OPENAI_API_KEY=...
  python3 gpt_ranker.py --split dev   # evaluate on dev
  python3 gpt_ranker.py --split test  # generate test predictions
"""
import os
import csv
import json
import time
import re
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.stats import kendalltau, spearmanr

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-5.4"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────
def parse_ranking(y_str: str) -> list[str]:
    """Parse 't9 t2 t7 ...' into ['t9', 't2', 't7', ...]."""
    return y_str.strip().split()


def ranking_to_positions(ranking: list[str]) -> dict[str, int]:
    """Convert ordered ranking list to {title_id: position}.
    Position 1 = best, 10 = worst."""
    return {tid: pos + 1 for pos, tid in enumerate(ranking)}


def positions_to_array(pos_dict: dict[str, int], n=10) -> np.ndarray:
    """Convert {t1: 5, t2: 3, ...} to array where index i = position of t(i+1)."""
    arr = np.zeros(n, dtype=int)
    for tid, pos in pos_dict.items():
        idx = int(tid.replace("t", "")) - 1
        arr[idx] = pos
    return arr


def load_data(split: str) -> pd.DataFrame:
    if split == "test":
        path = DATA_DIR / "test_public" / "test_public.csv"
    else:
        path = DATA_DIR / f"{split}_public.csv"
    df = pd.read_csv(path, encoding="utf-8")
    print(f"Loaded {len(df)} rows from {path}")
    return df


def build_few_shot_examples(train_df: pd.DataFrame, n=3) -> str:
    """Build few-shot examples from training data."""
    samples = train_df.sample(n=min(n, len(train_df)), random_state=42)
    examples = []
    for _, row in samples.iterrows():
        titles_block = ""
        for i in range(1, 11):
            titles_block += f"  t{i}: {row[f'title_{i}']}\n"
        # Truncate article body for prompt efficiency
        body = row["article_body"][:500] + "..." if len(row["article_body"]) > 500 else row["article_body"]
        ranking = row["y_true"]
        examples.append(
            f"Article (truncated): {body}\n\nCandidate titles:\n{titles_block}\nCorrect ranking (best to worst): {ranking}"
        )
    return "\n\n---\n\n".join(examples)


SYSTEM_PROMPT = """You are an expert Spanish political journalist. Your task is to rank 10 candidate headlines for a given news article from MOST relevant (best match) to LEAST relevant (worst match).

A good headline:
1. Accurately summarizes the main topic of the article
2. Captures the key actors, events, or decisions mentioned
3. Is factually consistent with the article content
4. Uses appropriate journalistic style

Some candidate titles may be near-duplicates with minor typos or variations — rank them similarly but note that exact matches to the article content should rank highest.
Some titles are about completely unrelated topics — these should rank lowest.

Respond with ONLY the ranking as space-separated title IDs from best to worst.
Example response: t5 t2 t9 t1 t6 t7 t10 t4 t3 t8

You MUST include all 10 title IDs (t1 through t10) exactly once."""


def rank_single(row: pd.Series, system_prompt: str, few_shot: str = "") -> str:
    """Rank titles for a single article using GPT-5.4."""
    titles_block = ""
    for i in range(1, 11):
        titles_block += f"  t{i}: {row[f'title_{i}']}\n"

    body = row["article_body"]
    # Truncate very long articles to ~2000 chars to save tokens
    if len(body) > 2000:
        body = body[:1000] + "\n[...]\n" + body[-1000:]

    user_msg = ""
    if few_shot:
        user_msg += f"Here are some examples of correct rankings:\n\n{few_shot}\n\n---\n\nNow rank the following:\n\n"

    user_msg += f"Article:\n{body}\n\nCandidate titles:\n{titles_block}\nRanking (best to worst):"

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=100,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            # Parse and validate
            tokens = raw.split()
            # Extract only t\d+ tokens
            valid = [t for t in tokens if re.match(r"^t\d+$", t)]
            if len(valid) == 10 and set(valid) == {f"t{i}" for i in range(1, 11)}:
                return " ".join(valid)
            else:
                print(f"  [WARN] Invalid response (attempt {attempt+1}): {raw[:100]}")
                # Try to salvage
                all_t = re.findall(r"t\d+", raw)
                seen = set()
                deduped = []
                for t in all_t:
                    if t not in seen and t in {f"t{i}" for i in range(1, 11)}:
                        seen.add(t)
                        deduped.append(t)
                if len(deduped) == 10:
                    return " ".join(deduped)
        except Exception as e:
            print(f"  [ERR] API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    # Fallback: default ordering
    print("  [FALLBACK] Using default t1 t2 ... t10")
    return " ".join(f"t{i}" for i in range(1, 11))


def evaluate(pred_rankings: list[str], true_rankings: list[str]) -> dict:
    """Compute ranking metrics: Kendall tau, Spearman, exact match."""
    taus = []
    spearmans = []
    exact = 0

    for pred, true in zip(pred_rankings, true_rankings):
        pred_pos = positions_to_array(ranking_to_positions(parse_ranking(pred)))
        true_pos = positions_to_array(ranking_to_positions(parse_ranking(true)))

        tau, _ = kendalltau(pred_pos, true_pos)
        rho, _ = spearmanr(pred_pos, true_pos)
        taus.append(tau)
        spearmans.append(rho)
        if pred.strip() == true.strip():
            exact += 1

    results = {
        "kendall_tau_mean": float(np.mean(taus)),
        "kendall_tau_std": float(np.std(taus)),
        "spearman_mean": float(np.mean(spearmans)),
        "spearman_std": float(np.std(spearmans)),
        "exact_match": exact / len(pred_rankings),
        "n_samples": len(pred_rankings),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "train"], default="dev")
    parser.add_argument("--few-shot", type=int, default=3)
    parser.add_argument("--tag", type=str, default="gpt54_text")
    args = parser.parse_args()

    print(f"=== PoliticHeadlinES GPT-5.4 Ranker ===")
    print(f"Split: {args.split} | Few-shot: {args.few_shot} | Tag: {args.tag}")
    print(f"Time: {datetime.now().isoformat()}")

    # Load data
    df = load_data(args.split)

    # Build few-shot examples from train
    few_shot = ""
    if args.few_shot > 0:
        train_df = load_data("train")
        few_shot = build_few_shot_examples(train_df, n=args.few_shot)
        print(f"Built {args.few_shot} few-shot examples")

    # Rank each article
    predictions = []
    for idx, (_, row) in enumerate(df.iterrows()):
        print(f"[{idx+1}/{len(df)}] Ranking article {row['id'][:16]}...")
        pred = rank_single(row, SYSTEM_PROMPT, few_shot=few_shot)
        predictions.append(pred)
        print(f"  Pred: {pred}")
        if "y_true" in df.columns:
            print(f"  True: {row['y_true']}")

    df["y_pred"] = predictions

    # Save predictions
    out_path = RESULTS_DIR / f"predictions_{args.tag}_{args.split}.csv"
    df[["id", "y_pred"]].to_csv(out_path, index=False)
    print(f"\nPredictions saved to {out_path}")

    # Evaluate if ground truth available
    if "y_true" in df.columns and args.split != "test":
        metrics = evaluate(predictions, df["y_true"].tolist())
        print(f"\n=== Results on {args.split} ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # Save metrics
        metrics_path = RESULTS_DIR / f"metrics_{args.tag}_{args.split}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    # Also save in submission format (for test)
    if args.split == "test":
        sub_path = RESULTS_DIR / f"submission_{args.tag}.csv"
        df[["id", "y_pred"]].rename(columns={"y_pred": "y_true"}).to_csv(sub_path, index=False)
        print(f"Submission file saved to {sub_path}")


if __name__ == "__main__":
    main()
