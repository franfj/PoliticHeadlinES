"""GPT-5.4 pairwise verification on top-3 for Politic.
For each dev instance:
  - Take GPT's top-3 candidates
  - 3 pairwise queries: which title is better for this article?
  - Aggregate by Copeland (wins count)
  - Keep GPT's 4-10 unchanged
"""
import csv, re, json, os, sys
import numpy as np
from scipy.stats import kendalltau, spearmanr
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

client = OpenAI()
MODEL = "gpt-5.4"

def parse(s): return [int(re.search(r'\d+', t).group()) for t in s.split()]

dev = {r["id"]: r for r in csv.DictReader(open("tasks/politicheadlines/data/dev_public.csv"))}
gpt = {r["id"]: parse(r["y_pred"]) for r in csv.DictReader(open("tasks/politicheadlines/results/predictions_gpt54_text_3shot_dev.csv"))}

def pairwise_call(article, title_a, title_b):
    """Ask GPT which title is the better headline. Returns 'A' or 'B'."""
    prompt = f"""Eres un editor político. Lee el artículo y compara dos titulares candidatos.
Responde SOLO con "A" o "B" según cuál es un mejor titular (más representativo, relevante, periodísticamente apropiado).

Artículo:
{article[:2500]}

Titular A: {title_a}
Titular B: {title_b}

Mejor titular (A/B):"""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":prompt}],
            max_completion_tokens=3,
        )
        txt = resp.choices[0].message.content.strip().upper()
        if txt.startswith("A"): return "A"
        if txt.startswith("B"): return "B"
        return "A"  # default
    except Exception as e:
        print(f"  ERROR: {e}")
        return "A"

def rerank_instance(iid):
    d = dev[iid]
    article = d["article_body"] or ""
    order = gpt[iid]
    top3 = order[:3]  # e.g. [10, 3, 5]
    tail = order[3:]
    # Get title texts
    titles = {t: d[f"title_{t}"] for t in top3}
    # 3 pairwise comparisons
    wins = {t: 0 for t in top3}
    for i in range(3):
        for j in range(i+1, 3):
            a, b = top3[i], top3[j]
            result = pairwise_call(article, titles[a], titles[b])
            if result == "A": wins[a] += 1
            else: wins[b] += 1
    # Sort top3 by wins desc, ties by original GPT order
    reranked = sorted(top3, key=lambda t: (-wins[t], top3.index(t)))
    return iid, reranked + tail

print(f"Pairwise on {len(dev)} dev instances × 3 pairs = {len(dev)*3} GPT calls")
results = {}
with ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(rerank_instance, iid) for iid in dev]
    for i, fut in enumerate(futures):
        iid, order = fut.result()
        results[iid] = order
        if (i+1) % 10 == 0: print(f"  {i+1}/{len(dev)}")

# Evaluate
def eval_preds(pred_map):
    taus, sps, top1, top2, top3 = [], [], [], [], []
    for iid, d in dev.items():
        if iid not in pred_map: continue
        true_order = parse(d["y_true"])
        pred_order = pred_map[iid]
        true_rank = {t:i+1 for i,t in enumerate(true_order)}
        pred_rank = {t:i+1 for i,t in enumerate(pred_order)}
        tau,_ = kendalltau([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
        sp,_ = spearmanr([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
        taus.append(tau); sps.append(sp)
        top1.append(1 if pred_order[0]==true_order[0] else 0)
        top2.append(1 if set(pred_order[:2])==set(true_order[:2]) else 0)
        top3.append(1 if set(pred_order[:3])==set(true_order[:3]) else 0)
    return np.mean(taus), np.mean(sps), np.mean(top1), np.mean(top2), np.mean(top3)

print(f"\n{'variant':40s} {'Kendall':>8s} {'Spearman':>8s} {'T1':>6s} {'T2':>6s} {'T3':>6s}")
k,s,t1,t2,t3 = eval_preds(gpt)
print(f"{'GPT baseline':40s} {k:>8.4f} {s:>8.4f} {t1:>6.2f} {t2:>6.2f} {t3:>6.2f}")
k,s,t1,t2,t3 = eval_preds(results)
print(f"{'GPT + pairwise rerank top-3':40s} {k:>8.4f} {s:>8.4f} {t1:>6.2f} {t2:>6.2f} {t3:>6.2f}")

# Save
with open("tasks/politicheadlines/results/dev_pairwise_rerank.json", "w") as f:
    json.dump({iid: order for iid, order in results.items()}, f)
print("\n✓ saved dev_pairwise_rerank.json")
