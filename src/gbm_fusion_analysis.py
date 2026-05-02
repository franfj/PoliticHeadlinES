"""Fusion strategies GPT + GBM on dev."""
import csv, re, json, numpy as np
from scipy.stats import kendalltau

def parse(s): return [int(re.search(r'\d+', t).group()) for t in s.split()]

dev = {r["id"]: r for r in csv.DictReader(open("tasks/politicheadlines/data/dev_public.csv"))}
gpt = {r["id"]: parse(r["y_pred"]) for r in csv.DictReader(open("tasks/politicheadlines/results/predictions_gpt54_text_3shot_dev.csv"))}

# GBM dev predictions — need to compute. Let me load from test if present, else compute
# Actually we need dev predictions. Let me compute them.
# Use the already-trained GBM models... they're on remote. Easier: recompute on dev here.

# Actually simpler: just use the GBM dev output that the script already computed.
# The script saved test_predictions_gbm.csv but we need DEV predictions.
# Let me re-run quickly getting dev GBM scores.

import subprocess, sys
# Easier approach: recompute GBM dev scores using stored models on remote
# But models aren't saved. Re-run feature extraction + inference on dev only.

from difflib import SequenceMatcher
def tok(s): return re.findall(r'\w+', (s or '').lower())
def ratio(a,b): return SequenceMatcher(None, a or '', b or '').ratio()

# Load train for retraining (quick since we have it)
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

train_rows = []
for path in ["tasks/politicheadlines/data/train_public.csv",
             "tasks/politicheadlines/data/test_public/train_public.csv"]:
    for r in csv.DictReader(open(path)):
        if r.get("y_true"): train_rows.append(r)

def features(row, gpt_order):
    article = (row.get("article_body","") or "")[:1000]
    article_tok = set(tok(article))
    titles = [row.get(f"title_{i}","") for i in range(1,11)]
    sim = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            sim[i,j] = ratio(titles[i], titles[j]) if i != j else 0
    max_sim = sim.max(axis=1)
    mean_sim = sim.mean(axis=1)
    gpt_rank = {t-1: i+1 for i, t in enumerate(gpt_order)} if gpt_order else {i:0 for i in range(10)}
    feats = []
    for i in range(10):
        t = titles[i]
        t_tok_set = set(tok(t))
        words = t.split()
        caps = sum(1 for w in words if len(w)>1 and w.isupper())
        overlap = len(t_tok_set & article_tok) / max(len(t_tok_set), 1)
        literal_in_article = 1 if t and t in article else 0
        feats.append([
            len(t), len(tok(t)), caps, t.count("?")+t.count("¿"), t.count("!")+t.count("¡"),
            t.count(":"), t.count('"')+t.count("'")+t.count('“'),
            overlap, ratio(t, article),
            max_sim[i], mean_sim[i], literal_in_article,
            gpt_rank.get(i, 0),
            sum(1 for j in range(10) if j != i and sim[i,j] > 0.9),
            sum(1 for j in range(10) if j != i and sim[i,j] > 0.8),
            sum(1 for j in range(10) if j != i and sim[i,j] > 0.6),
        ])
    return np.array(feats, dtype=np.float32)

print("Computing train features (2 min)...")
X, y, g = [], [], []
for idx, r in enumerate(train_rows):
    order = parse(r["y_true"])
    feats = features(r, None)
    for i in range(10):
        rank = order.index(i+1) + 1
        X.append(feats[i]); y.append(1.0 - (rank-1)/9.0); g.append(idx)
    if (idx+1)%3000==0: print(f"  {idx+1}/{len(train_rows)}")
X=np.array(X); y=np.array(y); g=np.array(g)

print("Training 5-fold...")
kf = GroupKFold(n_splits=5); models=[]
for fold, (tr,va) in enumerate(kf.split(X,y,g)):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(20)])
    models.append(m)

# Dev predictions
dev_list = list(dev.values())
X_dev = []
for r in dev_list:
    feats = features(r, gpt[r["id"]])
    for i in range(10): X_dev.append(feats[i])
X_dev = np.array(X_dev)
dev_preds = np.mean([m.predict(X_dev) for m in models], axis=0)

# Build per-instance scores
gbm_scores = {}
for idx, r in enumerate(dev_list):
    gbm_scores[r["id"]] = dev_preds[idx*10:(idx+1)*10]

# Fusion strategies
def kendall_eval(pred_map):
    taus, t1s, t2s, t3s = [], [], [], []
    for iid, d in dev.items():
        if iid not in pred_map: continue
        true_order = parse(d["y_true"])
        pred_order = pred_map[iid]
        true_rank = {t:i+1 for i,t in enumerate(true_order)}
        pred_rank = {t:i+1 for i,t in enumerate(pred_order)}
        tau,_ = kendalltau([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
        taus.append(tau)
        t1s.append(1 if pred_order[0]==true_order[0] else 0)
        t2s.append(1 if set(pred_order[:2])==set(true_order[:2]) else 0)
        t3s.append(1 if set(pred_order[:3])==set(true_order[:3]) else 0)
    return np.mean(taus), np.mean(t1s), np.mean(t2s), np.mean(t3s)

def gbm_only(iid):
    return [i+1 for i in np.argsort(-gbm_scores[iid])]
def gbm_preserve1(iid):
    order = gbm_only(iid); top = gpt[iid][0]
    if order[0] != top: order.remove(top); order.insert(0, top)
    return order
def gbm_preserve2(iid):
    order = gbm_only(iid); top2 = gpt[iid][:2]
    rest = [t for t in order if t not in top2]
    return top2 + rest
def weighted(iid, alpha):
    gpt_rank = {t: i+1 for i, t in enumerate(gpt[iid])}
    gbm_order = gbm_only(iid); gbm_rank = {t: i+1 for i, t in enumerate(gbm_order)}
    scores = {t: alpha*gpt_rank[t] + (1-alpha)*gbm_rank[t] for t in range(1,11)}
    return sorted(range(1,11), key=lambda t: scores[t])
def weighted_preserve(iid, alpha):
    order = weighted(iid, alpha); top = gpt[iid][0]
    if order[0] != top: order.remove(top); order.insert(0, top)
    return order
def rrf(iid, k=60):
    gpt_rank = {t: i+1 for i, t in enumerate(gpt[iid])}
    gbm_order = gbm_only(iid); gbm_rank = {t: i+1 for i, t in enumerate(gbm_order)}
    s = {t: 1/(k+gpt_rank[t]) + 1/(k+gbm_rank[t]) for t in range(1,11)}
    return sorted(range(1,11), key=lambda t: -s[t])

print(f"\n{'strategy':45s} {'Kendall':>8s} {'T1':>5s} {'T2':>5s} {'T3':>5s}")
k,t1,t2,t3 = kendall_eval(gpt); print(f"{'GPT only':45s} {k:>8.4f} {t1:>5.2f} {t2:>5.2f} {t3:>5.2f}")
k,t1,t2,t3 = kendall_eval({iid: gbm_only(iid) for iid in dev}); print(f"{'GBM only':45s} {k:>8.4f} {t1:>5.2f} {t2:>5.2f} {t3:>5.2f}")
k,t1,t2,t3 = kendall_eval({iid: gbm_preserve1(iid) for iid in dev}); print(f"{'GBM + preserve GPT top1':45s} {k:>8.4f} {t1:>5.2f} {t2:>5.2f} {t3:>5.2f}")
k,t1,t2,t3 = kendall_eval({iid: gbm_preserve2(iid) for iid in dev}); print(f"{'GBM + preserve GPT top2':45s} {k:>8.4f} {t1:>5.2f} {t2:>5.2f} {t3:>5.2f}")

for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    k,t1,t2,t3 = kendall_eval({iid: weighted(iid, a) for iid in dev})
    print(f"{'weighted α='+str(a):45s} {k:>8.4f} {t1:>5.2f} {t2:>5.2f} {t3:>5.2f}")
    k,t1,t2,t3 = kendall_eval({iid: weighted_preserve(iid, a) for iid in dev})
    print(f"{'weighted α='+str(a)+' + preserve top1':45s} {k:>8.4f} {t1:>5.2f} {t2:>5.2f} {t3:>5.2f}")

for k_rrf in [10, 30, 60]:
    k,t1,t2,t3 = kendall_eval({iid: rrf(iid, k_rrf) for iid in dev})
    print(f"{'RRF k='+str(k_rrf):45s} {k:>8.4f} {t1:>5.2f} {t2:>5.2f} {t3:>5.2f}")

# Save best
import pickle
with open("/tmp/gbm_dev_scores.pkl","wb") as f:
    pickle.dump({"scores": gbm_scores, "models": models}, f)
print("\nModels saved /tmp/gbm_dev_scores.pkl")
