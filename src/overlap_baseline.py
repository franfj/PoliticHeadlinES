"""Baseline: rank by article_overlap + combine with GPT rank 1."""
import csv, re, numpy as np
from scipy.stats import kendalltau, spearmanr
from collections import defaultdict

def parse(s):
    return [int(re.search(r'\d+', t).group()) for t in s.split()]

def overlap(t, article):
    t_words = set((t or "").lower().split())
    a_words = set((article or "")[:500].lower().split())
    return len(t_words & a_words) / max(len(t_words), 1)

# Load dev with y_true
dev = {}
with open("tasks/politicheadlines/data/dev_public.csv") as f:
    for row in csv.DictReader(f):
        dev[row["id"]] = row

# Load GPT preds
preds = {}
with open("tasks/politicheadlines/results/predictions_gpt54_text_3shot_dev.csv") as f:
    for row in csv.DictReader(f):
        preds[row["id"]] = row["y_pred"].split()

# Variants
variants = {
    "overlap_only": lambda d: sorted(range(1,11), key=lambda i: -overlap(d[f"title_{i}"], d["article_body"])),
    "gpt_only": lambda d: parse(" ".join(preds[d["id"]])),
    "gpt_top1_overlap_rest": None,  # computed below
    "gpt_top3_overlap_rest": None,
    "gpt_weighted_overlap": None,
}

def eval_ranking(pred_fn):
    taus, spears = [], []
    for did, d in dev.items():
        if did not in preds: continue
        true_order = parse(d["y_true"])
        pred_order = pred_fn(d)
        true_rank = {t: i+1 for i, t in enumerate(true_order)}
        pred_rank = {t: i+1 for i, t in enumerate(pred_order)}
        tau, _ = kendalltau([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
        sp, _ = spearmanr([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
        taus.append(tau); spears.append(sp)
    return np.mean(taus), np.mean(spears)

print(f"{'variant':40s} {'kendall':>10s} {'spearman':>10s}")
for name, fn in [("gpt_only", variants["gpt_only"]),
                 ("overlap_only", variants["overlap_only"])]:
    k, s = eval_ranking(fn)
    print(f"{name:40s} {k:>10.4f} {s:>10.4f}")

# Hybrid: keep GPT's top-1, re-rank rest by overlap
def gpt_top1_rest_overlap(d):
    gpt = parse(" ".join(preds[d["id"]]))
    top = gpt[0]
    rest = sorted([i for i in range(1,11) if i != top],
                  key=lambda i: -overlap(d[f"title_{i}"], d["article_body"]))
    return [top] + rest
k, s = eval_ranking(gpt_top1_rest_overlap)
print(f"{'gpt_top1 + overlap rest':40s} {k:>10.4f} {s:>10.4f}")

# Hybrid: keep GPT's top-3, re-rank rest by overlap
def gpt_top3_rest_overlap(d):
    gpt = parse(" ".join(preds[d["id"]]))
    top = gpt[:3]
    rest = sorted([i for i in range(1,11) if i not in top],
                  key=lambda i: -overlap(d[f"title_{i}"], d["article_body"]))
    return top + rest
k, s = eval_ranking(gpt_top3_rest_overlap)
print(f"{'gpt_top3 + overlap rest':40s} {k:>10.4f} {s:>10.4f}")

# Hybrid: weighted score = alpha*(-gpt_rank) + beta*overlap
for alpha, beta in [(1.0, 0.5), (1.0, 1.0), (0.7, 1.0), (0.5, 1.0), (0.3, 1.0), (1.0, 2.0), (1.0, 3.0)]:
    def weighted(d, a=alpha, b=beta):
        gpt = parse(" ".join(preds[d["id"]]))
        gpt_rank = {t: i+1 for i, t in enumerate(gpt)}
        scores = {i: -a*gpt_rank[i] + b*overlap(d[f"title_{i}"], d["article_body"]) for i in range(1,11)}
        return sorted(range(1,11), key=lambda i: -scores[i])
    k, s = eval_ranking(weighted)
    print(f"{'weighted α={:.1f} β={:.1f}'.format(alpha,beta):40s} {k:>10.4f} {s:>10.4f}")

# Multi-feature weighted
def multifeat(d, w_gpt=1.0, w_ov=2.0, w_len=-0.01, w_caps=-0.05):
    gpt = parse(" ".join(preds[d["id"]]))
    gpt_rank = {t: i+1 for i, t in enumerate(gpt)}
    scores = {}
    for i in range(1,11):
        t = d[f"title_{i}"]
        words = t.split()
        cap = sum(1 for w in words if len(w)>1 and w.isupper())
        scores[i] = (-w_gpt*gpt_rank[i]
                     + w_ov*overlap(t, d["article_body"])
                     + w_len*len(words)
                     + w_caps*cap)
    return sorted(range(1,11), key=lambda i: -scores[i])
k, s = eval_ranking(multifeat)
print(f"{'multifeat':40s} {k:>10.4f} {s:>10.4f}")
