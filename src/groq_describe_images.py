"""Describe all 4912 test images via Groq Llama-4-scout (FREE).

Saves a JSON cache: image_hash -> description.
Resumes on re-run (skips already-described).
"""
import os, json, base64, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from groq import Groq


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = RESULTS_DIR / "image_descriptions.json"

PROMPT = (
    "Describe esta imagen de noticia política en español. "
    "Sé específico: menciona personas concretas (con nombre si las reconoces), "
    "lugares, edificios institucionales, símbolos/banderas y cualquier objeto o "
    "texto destacado. 2-3 frases. Si no hay personas, describe la escena."
)


def describe_one(client: Groq, image_path: Path, retries: int = 3) -> str | None:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                max_completion_tokens=250,
            )
            return r.choices[0].message.content
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    test = pd.read_csv(DATA_DIR / "test_public" / "test_public.csv")
    hashes = list(test["image_hash"].unique())
    if args.limit:
        hashes = hashes[: args.limit]

    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())
        print(f"Loaded existing cache: {len(cache)} entries")

    todo = [h for h in hashes if h not in cache]
    print(f"Total: {len(hashes)}  Already done: {len(hashes) - len(todo)}  To do: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    img_dir = DATA_DIR / "test_public" / "images"
    t0 = time.time()
    done_this_run = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(describe_one, client, img_dir / f"{h}.jpg"): h for h in todo}
        for fut in as_completed(futs):
            h = futs[fut]
            try:
                desc = fut.result()
            except Exception:
                desc = None
            if desc:
                cache[h] = desc
            else:
                failed += 1
            done_this_run += 1
            if done_this_run % 100 == 0 or done_this_run == len(todo):
                rate = done_this_run / max(1, time.time() - t0)
                eta = (len(todo) - done_this_run) / max(1e-3, rate)
                print(f"  {done_this_run}/{len(todo)}  {rate:.1f}/s  eta={eta:.0f}s  failed={failed}", flush=True)
                # Periodic save
                CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"\nFinished. Cache: {len(cache)} entries. Failed: {failed}.  Saved {CACHE_PATH}")


if __name__ == "__main__":
    main()
