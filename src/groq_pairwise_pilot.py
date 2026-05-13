"""Groq pairwise pilot — test 3 models on 10 dev instances, see divergence from GPT."""
import csv, re, json, os
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.stats import kendalltau

def parse(s): return [int(re.search(r'\d+', t).group()) for t in s.split()]

client = Groq()
dev = {r["id"]: r for r in csv.DictReader(open("tasks/politicheadlines/data/dev_public.csv"))}
gpt = {r["id"]: parse(r["y_pred"]) for r in csv.DictReader(open("tasks/politicheadlines/results/predictions_gpt54_text_3shot_dev.csv"))}

MODELS = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "deepseek-r1-distill-llama-70b",
]

def pairwise(model, article, t_a, t_b):
    prompt = f"""Lee el artículo y compara dos titulares. Responde SOLO con "A" o "B".

Artículo:
{article[:2000]}

Titular A: {t_a}
Titular B: {t_b}

Mejor titular (A/B):"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            max_completion_tokens=5,
            temperature=0,
        )
        txt = resp.choices[0].message.content.strip().upper()
        # Handle <think> output from reasoning models
        m = re.search(r'\b([AB])\b', txt)
        return m.group(1) if m else "A"
    except Exception as e:
        print(f"  ERROR {model}: {str(e)[:100]}")
        return "A"

def rerank_with_model(model, iid):
    d = dev[iid]
    article = d["article_body"] or ""
    order = gpt[iid]
    top3 = order[:3]
    tail = order[3:]
    titles = {t: d[f"title_{t}"] for t in top3}
    wins = {t: 0 for t in top3}
    for i in range(3):
        for j in range(i+1, 3):
            a, b = top3[i], top3[j]
            r = pairwise(model, article, titles[a], titles[b])
            if r == "A": wins[a] += 1
            else: wins[b] += 1
    reranked = sorted(top3, key=lambda t: (-wins[t], top3.index(t)))
    return reranked + tail

# Pilot on first 10 dev instances
pilot_ids = list(dev.keys())[:10]
print(f"Piloto: {len(pilot_ids)} instancias × 3 pares × 3 modelos = {len(pilot_ids)*3*3} llamadas Groq")

results = {m: {} for m in MODELS}
for model in MODELS:
    print(f"\n→ {model}")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(rerank_with_model, model, iid): iid for iid in pilot_ids}
        for f in as_completed(futures):
            iid = futures[f]
            try:
                results[model][iid] = f.result()
            except Exception as e:
                print(f"  err {iid}: {e}")

# Evaluate each
print(f"\n{'model':45s} {'Kendall':>8s} {'T1':>5s} {'T2':>5s} {'T3':>5s} {'diff_GPT':>10s}")

def metrics(pred_map, subset):
    taus, t1s, t2s, t3s = [], [], [], []
    diff = 0
    for iid in subset:
        if iid not in pred_map: continue
        d = dev[iid]
        true_order = parse(d["y_true"])
        pred_order = pred_map[iid]
        true_rank = {t:i+1 for i,t in enumerate(true_order)}
        pred_rank = {t:i+1 for i,t in enumerate(pred_order)}
        t,_ = kendalltau([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
        taus.append(t)
        t1s.append(1 if pred_order[0]==true_order[0] else 0)
        t2s.append(1 if set(pred_order[:2])==set(true_order[:2]) else 0)
        t3s.append(1 if set(pred_order[:3])==set(true_order[:3]) else 0)
        if pred_order != gpt[iid]: diff += 1
    return np.mean(taus), np.mean(t1s), np.mean(t2s), np.mean(t3s), diff

k,t1,t2,t3,_ = metrics(gpt, pilot_ids)
print(f"{'GPT baseline':45s} {k:>8.4f} {t1:>5.2f} {t2:>5.2f} {t3:>5.2f}")
for m in MODELS:
    k,t1,t2,t3,d = metrics(results[m], pilot_ids)
    print(f"{m:45s} {k:>8.4f} {t1:>5.2f} {t2:>5.2f} {t3:>5.2f} {d:>10d}/{len(pilot_ids)}")

with open("/tmp/groq_pilot_results.json","w") as f:
    json.dump(results, f)
