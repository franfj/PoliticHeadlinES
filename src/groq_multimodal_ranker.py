"""Groq Llama-4-scout multimodal listwise ranker (FREE).

Uses the image directly as multimodal input. Llama-4-scout supports
vision natively. Free tier on Groq.
"""
import os, json, time, argparse, re, base64
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


SYSTEM = """Eres un editor experto de noticias políticas en español. Recibes un artículo (texto + imagen) y diez titulares candidatos. Debes ordenarlos del MEJOR al PEOR según relevancia/fidelidad al artículo.

Pista: la imagen suele mostrar al protagonista o lugar clave; el titular real cita ese mismo sujeto explícitamente. Los demás candidatos pueden ser paráfrasis sutiles o titulares de otras noticias.

Devuelve ÚNICAMENTE un JSON con el formato:
  {"ranking": ["t?", "t?", "t?", "t?", "t?", "t?", "t?", "t?", "t?", "t?"]}

Exactamente los diez ids t1..t10 (cada uno una vez). No añadas comentarios."""


def parse_ranking(s: str): return s.strip().split()


def load_data(split: str) -> pd.DataFrame:
    if split == "test":
        return pd.read_csv(DATA_DIR / "test_public" / "test_public.csv", encoding="utf-8")
    return pd.read_csv(DATA_DIR / f"{split}_public.csv", encoding="utf-8")


def find_image_path(image_hash: str, split: str) -> Path | None:
    for d in (DATA_DIR / "images", DATA_DIR / "test_public" / "images"):
        p = d / f"{image_hash}.jpg"
        if p.exists():
            return p
    return None


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


def encode_image(p: Path) -> str:
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def rank_one(client: Groq, model: str, row: pd.Series, split: str, retries: int = 3) -> str:
    body = str(row.get("article_body") or "")
    if len(body) > 1500:
        body = body[:750] + "\n[...]\n" + body[-750:]
    titles_block = "\n".join(f'  t{i}: {row[f"title_{i}"]}' for i in range(1, 11))
    user_text = (
        f"ARTÍCULO:\n{body}\n\nCANDIDATOS:\n{titles_block}\n\n"
        "Devuelve solo el JSON con la ordenación."
    )
    image_path = find_image_path(str(row.get("image_hash") or ""), split)
    content = [{"type": "text", "text": user_text}]
    if image_path:
        b64 = encode_image(image_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": content},
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
    return " ".join(f"t{i}" for i in range(1, 11))


def evaluate(predictions, golds):
    taus, rhos = [], []
    exact = 0
    top1_correct = 0
    for p, g in zip(predictions, golds):
        pp = parse_ranking(p); gg = parse_ranking(g)
        pos_p = {t: i for i, t in enumerate(pp)}
        pos_g = {t: i for i, t in enumerate(gg)}
        ids = [f"t{i}" for i in range(1, 11)]
        a = [pos_p.get(t, 9) for t in ids]
        b = [pos_g.get(t, 9) for t in ids]
        tau, _ = kendalltau(a, b)
        rho, _ = spearmanr(a, b)
        taus.append(tau)
        rhos.append(rho)
        if pp[0] == gg[0]: top1_correct += 1
        if p.strip() == g.strip(): exact += 1
    return {
        "kendall_tau_mean": float(np.mean(taus)),
        "spearman_mean": float(np.mean(rhos)),
        "top1_acc": top1_correct / max(1, len(predictions)),
        "exact_match": exact / max(1, len(predictions)),
        "n": len(predictions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/llama-4-scout-17b-16e-instruct")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    safe_model = re.sub(r"[^A-Za-z0-9]+", "_", args.model)
    tag = args.tag or f"groq_mm_{safe_model}"
    print(f"=== Politic Groq multimodal — {args.model} | {args.split} ===")
    print(datetime.now().isoformat(timespec="seconds"))
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    df = load_data(args.split)
    if args.limit:
        df = df.head(args.limit)
    print(f"target: {len(df)}")

    preds = [None] * len(df)
    t0 = time.time()
    last_save = 0
    out_csv = RESULTS_DIR / f"predictions_{tag}_{args.split}.csv"
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(rank_one, client, args.model, row, args.split): i
                for i, (_, row) in enumerate(df.iterrows())}
        done = 0
        for f in as_completed(futs):
            i = futs[f]
            preds[i] = f.result()
            done += 1
            if done % 100 == 0 or done == len(df):
                rate = done / max(1, time.time() - t0)
                eta = (len(df) - done) / max(1e-3, rate)
                print(f"  {done}/{len(df)}  {rate:.1f}/s  eta={eta:.0f}s", flush=True)
                # Periodic checkpoint save
                if done - last_save >= 200 or done == len(df):
                    tmp = pd.DataFrame({"id": df["id"].values, "y_pred": [p or "" for p in preds]})
                    if args.split == "test":
                        tmp = tmp.rename(columns={"y_pred": "task_1"})
                        tmp["task_2"] = tmp["task_1"]
                    tmp.to_csv(out_csv, index=False)
                    last_save = done

    df = df.copy()
    df["y_pred"] = preds
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
