"""Generate final GBM+GPT fusion submission for Politic test."""
import csv, re, pickle, numpy as np
from difflib import SequenceMatcher
from collections import Counter

def parse(s): return [int(re.search(r'\d+', t).group()) for t in s.split()]
def tok(s): return re.findall(r'\w+', (s or '').lower())
def ratio(a,b): return SequenceMatcher(None, a or '', b or '').ratio()

# Load trained GBM models
with open("/tmp/gbm_dev_scores.pkl","rb") as f:
    data = pickle.load(f)
models = data["models"]

def features(row, gpt_order):
    article = (row.get("article_body","") or "")[:1000]
    article_tok = set(tok(article))
    titles = [row.get(f"title_{i}","") for i in range(1,11)]
    sim = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            sim[i,j] = ratio(titles[i], titles[j]) if i != j else 0
    max_sim = sim.max(axis=1); mean_sim = sim.mean(axis=1)
    gpt_rank = {t-1: i+1 for i, t in enumerate(gpt_order)} if gpt_order else {i:0 for i in range(10)}
    feats = []
    for i in range(10):
        t = titles[i]; t_tok_set = set(tok(t)); words = t.split()
        caps = sum(1 for w in words if len(w)>1 and w.isupper())
        overlap = len(t_tok_set & article_tok) / max(len(t_tok_set), 1)
        literal = 1 if t and t in article else 0
        feats.append([
            len(t), len(tok(t)), caps, t.count("?")+t.count("¿"), t.count("!")+t.count("¡"),
            t.count(":"), t.count('"')+t.count("'")+t.count('“'),
            overlap, ratio(t, article),
            max_sim[i], mean_sim[i], literal, gpt_rank.get(i,0),
            sum(1 for j in range(10) if j != i and sim[i,j] > 0.9),
            sum(1 for j in range(10) if j != i and sim[i,j] > 0.8),
            sum(1 for j in range(10) if j != i and sim[i,j] > 0.6),
        ])
    return np.array(feats, dtype=np.float32)

test_rows = list(csv.DictReader(open("tasks/politicheadlines/data/test_public/test_public.csv")))
gpt_test = {r["id"]: parse(r["y_pred"]) for r in csv.DictReader(open("tasks/politicheadlines/results/test_predictions_gpt54.csv"))}
print(f"Test: {len(test_rows)}")

# Compute test features
print("Computing test features...")
X = []
for idx, r in enumerate(test_rows):
    feats = features(r, gpt_test[r["id"]])
    for i in range(10): X.append(feats[i])
    if (idx+1)%1000==0: print(f"  {idx+1}/{len(test_rows)}")
X = np.array(X)

# Predict
print("Predicting with 5 GBM models...")
preds = np.mean([m.predict(X) for m in models], axis=0)

# Fusion: weighted α=0.3 + preserve GPT top1
ALPHA = 0.3
out_path = "tasks/politicheadlines/results/test_predictions_gbm_fusion_a03.csv"
with open(out_path, "w") as f:
    f.write("id,task_1,task_2\n")
    for idx, r in enumerate(test_rows):
        scores = preds[idx*10:(idx+1)*10]
        gbm_order = [i+1 for i in np.argsort(-scores)]
        gpt_order = gpt_test[r["id"]]
        gpt_rank = {t: i+1 for i, t in enumerate(gpt_order)}
        gbm_rank = {t: i+1 for i, t in enumerate(gbm_order)}
        combined = {t: ALPHA*gpt_rank[t] + (1-ALPHA)*gbm_rank[t] for t in range(1,11)}
        order = sorted(range(1,11), key=lambda t: combined[t])
        # Preserve GPT top1
        top = gpt_order[0]
        if order[0] != top:
            order.remove(top); order.insert(0, top)
        s = " ".join(f"t{t}" for t in order)
        f.write(f"{r['id']},{s},{s}\n")
print(f"Saved {out_path}")
