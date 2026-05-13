"""PoliticHeadlinES — GPT-5.5 listwise ranker with reinforced prompt.

Improvements over the previous gpt-5.4 baseline (3-shot):
  - 5 few-shot examples drawn diversely from train.
  - Explicit chain-of-thought structure inside the user prompt
    (asks the model to think about article topic first, identify
    on-topic vs off-topic candidates, then write the final ranking).
  - JSON-mode output to avoid parsing errors.
  - Parallel worker pool for speed.
"""
import os, json, time, argparse, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import kendalltau, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


SYSTEM = """You are an expert Spanish political-news editor. You receive an article and ten candidate headlines and must rank them from BEST to WORST relevance.

Reasoning checklist (do this silently, do NOT include in the JSON):
1. What is this article actually about? (main subject + key event)
2. For each headline, decide:
    a) is it on-topic for the article (same subject, same event)?
    b) is it factually consistent with what the article says?
    c) is it a near-duplicate of another headline (same content, minor typos)?
    d) is it about a completely different subject (clearly off-topic)?
3. Order: most-on-topic-and-accurate first; off-topic last.
   Near-duplicates of the best headline rank just after it.
   Off-topic noise ranks at the very bottom.

OUTPUT FORMAT — strict JSON:
  {"ranking": ["t?", "t?", "t?", "t?", "t?", "t?", "t?", "t?", "t?", "t?"]}

The ranking MUST contain exactly the ten ids t1..t10 (each appearing
once). No extra text, no comments, no Markdown. Just the JSON object."""


def parse_ranking(s: str) -> list[str]:
    return s.strip().split()


def load_data(split: str) -> pd.DataFrame:
    if split == "test":
        path = DATA_DIR / "test_public" / "test_public.csv"
    else:
        path = DATA_DIR / f"{split}_public.csv"
    return pd.read_csv(path, encoding="utf-8")


def build_few_shot(train_df: pd.DataFrame, n: int = 5) -> str:
    """Pick n diverse train examples (by varying article length / id hash)."""
    train_df = train_df.copy()
    train_df["_len"] = train_df["article_body"].astype(str).str.len()
    # Stratify by quartile of length to diversify
    train_df["_q"] = pd.qcut(train_df["_len"], 4, labels=False, duplicates="drop")
    samples = train_df.groupby("_q", group_keys=False).apply(
        lambda g: g.sample(min(2, len(g)), random_state=42)
    ).head(n)
    examples = []
    for _, row in samples.iterrows():
        body = row["article_body"]
        if len(body) > 700:
            body = body[:400] + " [...] " + body[-300:]
        titles_block = "\n".join(f'  t{i}: {row[f"title_{i}"]}' for i in range(1, 11))
        ranking = row["y_true"].strip().split()
        examples.append(
            f"ARTICLE:\n{body}\n\nCANDIDATE HEADLINES:\n{titles_block}\n\n"
            f'CORRECT RANKING (best to worst): {{"ranking": {json.dumps(ranking)}}}'
        )
    return "\n\n========\n\n".join(examples)


def parse_response(content: str) -> list[str] | None:
    if not content:
        return None
    content = content.strip()
    try:
        obj = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            # Last-resort: pull every t\d+ token in order
            tokens = re.findall(r"t\d+", content)
            seen = set()
            out = []
            for t in tokens:
                if t not in seen and t in {f"t{i}" for i in range(1, 11)}:
                    seen.add(t)
                    out.append(t)
            return out if len(out) == 10 else None
        try:
            obj = json.loads(m.group())
        except Exception:
            return None
    ranking = obj.get("ranking") if isinstance(obj, dict) else None
    if not isinstance(ranking, list):
        return None
    valid = [t for t in ranking if isinstance(t, str) and t in {f"t{i}" for i in range(1, 11)}]
    seen = set()
    deduped = []
    for t in valid:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    if len(deduped) == 10:
        return deduped
    return None


def rank_one(client: OpenAI, model: str, row: pd.Series, few_shot: str, retries: int = 3) -> str:
    body = str(row.get("article_body") or "")
    if len(body) > 2200:
        body = body[:1100] + "\n[...]\n" + body[-1100:]
    titles_block = "\n".join(f'  t{i}: {row[f"title_{i}"]}' for i in range(1, 11))
    user = (
        f"FEW-SHOT EXAMPLES:\n\n{few_shot}\n\n========\n\n"
        f"NOW RANK THIS ONE.\n\nARTICLE:\n{body}\n\n"
        f"CANDIDATE HEADLINES:\n{titles_block}\n\n"
        "Return ONLY the JSON ranking as specified."
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=400,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            ranking = parse_response(content)
            if ranking is not None:
                return " ".join(ranking)
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
                continue
    # Fallback: default order
    return " ".join(f"t{i}" for i in range(1, 11))


def evaluate(predictions: list[str], golds: list[str]) -> dict:
    taus, rhos = [], []
    exact = 0
    for p, g in zip(predictions, golds):
        # Build position arrays for the 10 titles t1..t10
        pp = parse_ranking(p)
        gg = parse_ranking(g)
        pos_p = {t: i for i, t in enumerate(pp)}
        pos_g = {t: i for i, t in enumerate(gg)}
        ids = [f"t{i}" for i in range(1, 11)]
        a = [pos_p.get(t, 9) for t in ids]
        b = [pos_g.get(t, 9) for t in ids]
        tau, _ = kendalltau(a, b)
        rho, _ = spearmanr(a, b)
        taus.append(tau)
        rhos.append(rho)
        if p.strip() == g.strip():
            exact += 1
    return {
        "kendall_tau_mean": float(np.mean(taus)),
        "kendall_tau_std": float(np.std(taus)),
        "spearman_mean": float(np.mean(rhos)),
        "exact_match": exact / max(1, len(predictions)),
        "n": len(predictions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--few_shot", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"{args.model.replace('.', '_')}_fs{args.few_shot}"
    print(f"=== Politic GPT ranker — model={args.model} split={args.split} workers={args.workers} ===")
    print(f"Started {datetime.now().isoformat(timespec='seconds')}")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    train = load_data("train")
    df = load_data(args.split)
    if args.limit:
        df = df.head(args.limit)
    print(f"Train rows: {len(train)}, target rows: {len(df)}")

    few_shot = build_few_shot(train, n=args.few_shot)
    print(f"Few-shot block: {len(few_shot)} chars")

    preds = [None] * len(df)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(rank_one, client, args.model, row, few_shot): i
                for i, (_, row) in enumerate(df.iterrows())}
        done = 0
        for f in as_completed(futs):
            i = futs[f]
            preds[i] = f.result()
            done += 1
            if done % 25 == 0 or done == len(df):
                rate = done / max(1, time.time() - t0)
                eta = (len(df) - done) / max(1e-3, rate)
                print(f"  {done}/{len(df)}  {rate:.1f}/s  eta={eta:.0f}s", flush=True)

    df = df.copy()
    df["y_pred"] = preds

    out_csv = RESULTS_DIR / f"predictions_{tag}_{args.split}.csv"
    if args.split == "test":
        df[["id", "y_pred"]].rename(columns={"y_pred": "task_1"}).assign(task_2=df["y_pred"]).to_csv(out_csv, index=False)
    else:
        df[["id", "y_pred"]].to_csv(out_csv, index=False)
    print(f"Predictions saved: {out_csv}")

    if "y_true" in df.columns and args.split != "test":
        metrics = evaluate(preds, df["y_true"].tolist())
        print(f"\n=== Metrics ({args.split}) ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}" if isinstance(v, int) else f"  {k}: {v:.4f}")
        with open(RESULTS_DIR / f"metrics_{tag}_{args.split}.json", "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
