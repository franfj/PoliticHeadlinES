"""PoliticHeadlinES — Groq listwise ranker (Llama-3.3-70B / Llama-4-scout).

Uses the Groq REST API (OpenAI-compatible). The same prompt structure as
gpt55_ranker.py / gpt_vision_ranker.py — text-only listwise with
JSON-mode output.
"""
import os, json, time, argparse, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from groq import Groq
from scipy.stats import kendalltau, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


SYSTEM = """You are an expert Spanish political-news editor. You receive an article and ten candidate headlines, and you must rank them from BEST to WORST relevance.

Reasoning checklist (silent):
1. Identify the article's main subject and key event.
2. For each headline, decide on-topic vs off-topic, factual consistency, near-duplicate vs unique, generic vs specific.
3. Rank: on-topic and accurate first; near-duplicates of the best one cluster near the top; off-topic noise at the bottom.

Output STRICT JSON:
  {"ranking": ["t?", "t?", "t?", "t?", "t?", "t?", "t?", "t?", "t?", "t?"]}

Exactly the ten ids t1..t10, each appearing once. No extra text."""


def parse_ranking(s: str) -> list[str]:
    return s.strip().split()


def load_data(split: str) -> pd.DataFrame:
    if split == "test":
        path = DATA_DIR / "test_public" / "test_public.csv"
    else:
        path = DATA_DIR / f"{split}_public.csv"
    return pd.read_csv(path, encoding="utf-8")


def build_few_shot(train_df: pd.DataFrame, n: int = 5) -> str:
    train_df = train_df.copy()
    train_df["_len"] = train_df["article_body"].astype(str).str.len()
    train_df["_q"] = pd.qcut(train_df["_len"], 4, labels=False, duplicates="drop")
    samples = train_df.groupby("_q", group_keys=False).apply(
        lambda g: g.sample(min(2, len(g)), random_state=42)
    ).head(n)
    examples = []
    for _, row in samples.iterrows():
        body = row["article_body"]
        if len(body) > 600:
            body = body[:300] + " [...] " + body[-300:]
        titles_block = "\n".join(f'  t{i}: {row[f"title_{i}"]}' for i in range(1, 11))
        ranking = row["y_true"].strip().split()
        examples.append(
            f"ARTICLE:\n{body}\n\nCANDIDATE HEADLINES:\n{titles_block}\n\n"
            f'CORRECT RANKING: {{"ranking": {json.dumps(ranking)}}}'
        )
    return "\n\n========\n\n".join(examples)


def parse_response(content: str):
    if not content:
        return None
    try:
        obj = json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except Exception:
            return None
    ranking = obj.get("ranking") if isinstance(obj, dict) else None
    if not isinstance(ranking, list):
        return None
    valid = [t for t in ranking if isinstance(t, str) and t in {f"t{i}" for i in range(1, 11)}]
    seen = set()
    out = []
    for t in valid:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out if len(out) == 10 else None


def rank_one(client: Groq, model: str, row: pd.Series, few_shot: str, retries: int = 3) -> str:
    body = str(row.get("article_body") or "")
    if len(body) > 2000:
        body = body[:1000] + "\n[...]\n" + body[-1000:]
    titles_block = "\n".join(f'  t{i}: {row[f"title_{i}"]}' for i in range(1, 11))
    user = (
        f"FEW-SHOT EXAMPLES:\n\n{few_shot}\n\n========\n\n"
        f"NOW RANK THIS ONE.\n\nARTICLE:\n{body}\n\n"
        f"CANDIDATE HEADLINES:\n{titles_block}\n\n"
        "Return ONLY the JSON ranking."
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=200,
                response_format={"type": "json_object"},
            )
            ranking = parse_response(resp.choices[0].message.content or "")
            if ranking is not None:
                return " ".join(ranking)
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
                continue
    return " ".join(f"t{i}" for i in range(1, 11))


def evaluate(predictions, golds):
    taus, rhos = [], []
    exact = 0
    for p, g in zip(predictions, golds):
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
        "spearman_mean": float(np.mean(rhos)),
        "exact_match": exact / max(1, len(predictions)),
        "n": len(predictions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--few_shot", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    safe_model = re.sub(r"[^A-Za-z0-9]+", "_", args.model)
    tag = args.tag or f"groq_{safe_model}_fs{args.few_shot}"
    print(f"=== Politic Groq ranker — {args.model} | {args.split} ===")
    print(datetime.now().isoformat(timespec="seconds"))
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    train = load_data("train")
    df = load_data(args.split)
    if args.limit:
        df = df.head(args.limit)
    print(f"Train: {len(train)}, target: {len(df)}")

    few_shot = build_few_shot(train, n=args.few_shot)
    print(f"Few-shot: {len(few_shot)} chars")

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
    print(f"Saved: {out_csv}")

    if "y_true" in df.columns and args.split != "test":
        m = evaluate(preds, df["y_true"].tolist())
        print(f"\n=== Metrics ({args.split}) ===")
        for k, v in m.items():
            print(f"  {k}: {v}" if isinstance(v, int) else f"  {k}: {v:.4f}")
        with open(RESULTS_DIR / f"metrics_{tag}_{args.split}.json", "w") as fp:
            json.dump(m, fp, indent=2)


if __name__ == "__main__":
    main()
