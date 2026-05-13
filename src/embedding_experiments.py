"""Local Politic experiments: embeddings + RRF + top-3 focused.
   Uses ONLY dev (50 instances) — CPU only."""
import csv, re, json, numpy as np
from scipy.stats import kendalltau, spearmanr
from sentence_transformers import SentenceTransformer
from collections import Counter, defaultdict

def parse(s): return [int(re.search(r'\d+', t).group()) for t in s.split()]

# Load dev + GPT preds
dev = {r["id"]: r for r in csv.DictReader(open("tasks/politicheadlines/data/dev_public.csv"))}
gpt = {r["id"]: parse(r["y_pred"]) for r in csv.DictReader(open("tasks/politicheadlines/results/predictions_gpt54_text_3shot_dev.csv"))}

print("Loading multilingual embedder (MiniLM)...")
# Use small multilingual model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("OK")

# Compute article-title cosine for each instance
embed_scores = {}  # id -> [score_t1, ..., score_t10]
for iid, d in dev.items():
    article = (d["article_body"] or "")[:512]  # cap for embedder
    titles = [d[f"title_{i}"] for i in range(1, 11)]
    all_texts = [article] + titles
    embs = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
    art_emb = embs[0]
    title_embs = embs[1:]
    sims = (title_embs @ art_emb).tolist()
    embed_scores[iid] = sims

def eval_preds(pred_map):
    taus, sps, tops = [], [], []
    top2_acc, top3_acc = [], []
    for iid, d in dev.items():
        if iid not in pred_map: continue
        true_order = parse(d["y_true"])
        pred_order = pred_map[iid]
        true_rank = {t:i+1 for i,t in enumerate(true_order)}
        pred_rank = {t:i+1 for i,t in enumerate(pred_order)}
        tau,_ = kendalltau([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
        sp,_ = spearmanr([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
        taus.append(tau); sps.append(sp)
        tops.append(1 if pred_order[0]==true_order[0] else 0)
        top2_acc.append(1 if set(pred_order[:2])==set(true_order[:2]) else 0)
        top3_acc.append(1 if set(pred_order[:3])==set(true_order[:3]) else 0)
    return np.mean(taus), np.mean(sps), np.mean(tops), np.mean(top2_acc), np.mean(top3_acc)

print(f"\n{'variant':50s} {'Kendall':>8s} {'Spearman':>8s} {'Top1':>6s} {'Top2':>6s} {'Top3':>6s}")
def show(name, pm):
    k,s,t1,t2,t3 = eval_preds(pm)
    print(f"{name:50s} {k:>8.4f} {s:>8.4f} {t1:>6.2f} {t2:>6.2f} {t3:>6.2f}")

# Baseline: GPT
show("GPT only", gpt)

# Embed only
embed_order = {iid: sorted(range(1,11), key=lambda i: -embed_scores[iid][i-1]) for iid in embed_scores}
show("Embed only (MiniLM)", embed_order)

# Embed preserve top1 from GPT
def embed_preserve(iid):
    rest = [i for i in sorted(range(1,11), key=lambda i: -embed_scores[iid][i-1]) if i != gpt[iid][0]]
    return [gpt[iid][0]] + rest
show("GPT top1 + embed rest", {iid: embed_preserve(iid) for iid in gpt})

# RRF: 1/(k + rank_GPT) + 1/(k + rank_embed)
def rrf(iid, k=60):
    gpt_rank = {t: i+1 for i, t in enumerate(gpt[iid])}
    emb_order = sorted(range(1,11), key=lambda i: -embed_scores[iid][i-1])
    emb_rank = {t: i+1 for i, t in enumerate(emb_order)}
    scores = {t: 1/(k+gpt_rank[t]) + 1/(k+emb_rank[t]) for t in range(1,11)}
    return sorted(range(1,11), key=lambda t: -scores[t])
for k in [5, 10, 30, 60, 100]:
    show(f"RRF k={k}", {iid: rrf(iid, k) for iid in gpt})

# Weighted (alpha on GPT)
for alpha in [0.3, 0.5, 0.7]:
    def w(iid, a=alpha):
        gpt_rank = {t: i+1 for i, t in enumerate(gpt[iid])}
        emb_order = sorted(range(1,11), key=lambda i: -embed_scores[iid][i-1])
        emb_rank = {t: i+1 for i, t in enumerate(emb_order)}
        scores = {t: -(a*gpt_rank[t] + (1-a)*emb_rank[t]) for t in range(1,11)}
        out = sorted(range(1,11), key=lambda t: -scores[t])
        # preserve GPT top1
        top = gpt[iid][0]
        if out[0] != top:
            out.remove(top); out.insert(0, top)
        return out
    show(f"Weighted α={alpha} + preserve top1", {iid: w(iid) for iid in gpt})

# Top-3 only re-rank: keep GPT's 4-10, re-order 1-3 by embedding
def top3_rerank(iid):
    gpt_order = gpt[iid]
    top3_set = set(gpt_order[:3])
    # Re-order top3 by embedding
    top3_sorted = sorted(top3_set, key=lambda i: -embed_scores[iid][i-1])
    return top3_sorted + gpt_order[3:]
show("Top-3 re-rank by embed", {iid: top3_rerank(iid) for iid in gpt})

# Top-3 keep GPT top1, re-order 2-3 by embed
def top23_rerank(iid):
    gpt_order = gpt[iid]
    top = gpt_order[0]
    rank2_3 = sorted([gpt_order[1], gpt_order[2]], key=lambda i: -embed_scores[iid][i-1])
    return [top] + rank2_3 + gpt_order[3:]
show("Top-2,3 re-rank by embed", {iid: top23_rerank(iid) for iid in gpt})
