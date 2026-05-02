"""Rich features + LightGBM pointwise ranker for Politic.
Uses 16k train → validates with group k-fold → predicts on dev+test."""
import csv, re, json, os, numpy as np
from difflib import SequenceMatcher
from scipy.stats import kendalltau, spearmanr
from collections import Counter
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

def parse(s): return [int(re.search(r'\d+', t).group()) for t in s.split()]
def tok(s): return re.findall(r'\w+', (s or '').lower())
def ratio(a,b): return SequenceMatcher(None, a or '', b or '').ratio()

# Load everything
train_rows = []
for path in ["tasks/politicheadlines/data/train_public.csv",
             "tasks/politicheadlines/data/test_public/train_public.csv"]:
    for r in csv.DictReader(open(path)):
        if r.get("y_true"): train_rows.append(r)
dev_rows = list(csv.DictReader(open("tasks/politicheadlines/data/dev_public.csv")))
test_rows = list(csv.DictReader(open("tasks/politicheadlines/data/test_public/test_public.csv")))
gpt_dev = {r["id"]: parse(r["y_pred"]) for r in csv.DictReader(open("tasks/politicheadlines/results/predictions_gpt54_text_3shot_dev.csv"))}
gpt_test = {r["id"]: parse(r["y_pred"]) for r in csv.DictReader(open("tasks/politicheadlines/results/test_predictions_gpt54.csv"))}
print(f"Train: {len(train_rows)}, Dev: {len(dev_rows)}, Test: {len(test_rows)}")

def features(row, gpt_order):
    """Per-title features for one instance. Returns list of 10 feature vectors."""
    article = (row.get("article_body","") or "")[:1000]
    article_tok = set(tok(article))
    titles = [row.get(f"title_{i}","") for i in range(1,11)]

    # Pairwise similarity matrix
    sim = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            sim[i,j] = ratio(titles[i], titles[j]) if i != j else 0
    max_sim = sim.max(axis=1)  # max sim with any other title (perturbation cluster)
    argmax_sim = sim.argmax(axis=1)
    mean_sim = sim.mean(axis=1)

    gpt_rank = {t-1: i+1 for i, t in enumerate(gpt_order)} if gpt_order else {i:0 for i in range(10)}

    feats = []
    for i in range(10):
        t = titles[i]
        t_tok = tok(t)
        t_tok_set = set(t_tok)
        words = t.split()
        caps = sum(1 for w in words if len(w)>1 and w.isupper())
        overlap = len(t_tok_set & article_tok) / max(len(t_tok_set), 1)
        # Does title appear literally in article?
        literal_in_article = 1 if t and t in article else 0
        feats.append([
            len(t),
            len(t_tok),
            caps,
            t.count("?") + t.count("¿"),
            t.count("!") + t.count("¡"),
            t.count(":"),
            t.count('"') + t.count("'") + t.count('“'),
            overlap,
            ratio(t, article),
            max_sim[i],           # max sim with any other title
            mean_sim[i],
            literal_in_article,
            gpt_rank.get(i, 0),
            sum(1 for j in range(10) if j != i and sim[i,j] > 0.9),  # near-duplicate count
            sum(1 for j in range(10) if j != i and sim[i,j] > 0.8),
            sum(1 for j in range(10) if j != i and sim[i,j] > 0.6),
        ])
    return np.array(feats, dtype=np.float32)

print("Computing features (may take ~2 min)...")
X_train, y_train, group_train = [], [], []
for idx, r in enumerate(train_rows):
    order = parse(r["y_true"])
    feats = features(r, None)  # No GPT rank for train (don't have it)
    # target: higher=better rank, 1.0 → 0.0 scale
    for i in range(10):
        rank = order.index(i+1) + 1
        target = 1.0 - (rank-1)/9.0
        X_train.append(feats[i])
        y_train.append(target)
        group_train.append(idx)
    if (idx+1) % 2000 == 0: print(f"  {idx+1}/{len(train_rows)}")

X_train = np.array(X_train)
y_train = np.array(y_train)
group_train = np.array(group_train)
print(f"Features: X={X_train.shape}, y={y_train.shape}")

# Train LightGBM with 5-fold group CV
feature_names = ["len","words","caps","questions","excl","colon","quote",
                 "art_overlap","art_ratio","max_sim","mean_sim","literal_in_art",
                 "gpt_rank","near_dup_0.9","near_dup_0.8","near_dup_0.6"]

kf = GroupKFold(n_splits=5)
fold_k, fold_models = [], []
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train, group_train)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(20)])
    # Evaluate Kendall per group
    va_groups = group_train[va_idx]
    preds = model.predict(X_va)
    taus = []
    for g in np.unique(va_groups):
        mask = va_groups == g
        pred_order = np.argsort(-preds[mask])
        true_order = np.argsort(-y_va[mask])
        true_rank = {t:i for i,t in enumerate(true_order)}
        pred_rank = {t:i for i,t in enumerate(pred_order)}
        t,_ = kendalltau([true_rank[i] for i in range(len(mask[mask]))],
                         [pred_rank[i] for i in range(len(mask[mask]))])
        taus.append(t)
    fold_k.append(np.mean(taus))
    fold_models.append(model)
    print(f"Fold {fold+1}: Kendall={np.mean(taus):.4f}")
print(f"CV avg Kendall: {np.mean(fold_k):.4f}")

# Evaluate on dev (with GPT rank feature)
def build_dev_features(rows, gpt_map):
    X, groups = [], []
    for idx, r in enumerate(rows):
        feats = features(r, gpt_map.get(r["id"]))
        for i in range(10):
            X.append(feats[i]); groups.append(idx)
    return np.array(X), np.array(groups)

X_dev, g_dev = build_dev_features(dev_rows, gpt_dev)
# Average over folds
dev_preds = np.mean([m.predict(X_dev) for m in fold_models], axis=0)

# Kendall on dev (with and without GPT top1 preserve)
taus_gbm = []
taus_gbm_pres = []
for idx, r in enumerate(dev_rows):
    mask = g_dev == idx
    scores = dev_preds[mask]
    true_order = parse(r["y_true"])  # e.g., [10,3,5,...]
    pred_idx_order = np.argsort(-scores)  # 0-based indices sorted
    pred_order = [i+1 for i in pred_idx_order]  # 1-based title ids
    # Preserve top-1
    preserved = pred_order[:]
    gpt_top = gpt_dev[r["id"]][0]
    if preserved[0] != gpt_top:
        preserved.remove(gpt_top); preserved.insert(0, gpt_top)
    true_rank = {t:i+1 for i,t in enumerate(true_order)}
    pred_rank = {t:i+1 for i,t in enumerate(pred_order)}
    t,_ = kendalltau([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
    taus_gbm.append(t)
    pres_rank = {t:i+1 for i,t in enumerate(preserved)}
    t,_ = kendalltau([true_rank[t] for t in range(1,11)], [pres_rank[t] for t in range(1,11)])
    taus_gbm_pres.append(t)

print(f"\nDev Kendall:")
taus_gpt = []
for r in dev_rows:
    true_order = parse(r["y_true"])
    pred_order = gpt_dev[r["id"]]
    true_rank = {t:i+1 for i,t in enumerate(true_order)}
    pred_rank = {t:i+1 for i,t in enumerate(pred_order)}
    t,_ = kendalltau([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
    taus_gpt.append(t)
print(f"  GPT only          : {np.mean(taus_gpt):.4f}")
print(f"  GBM only          : {np.mean(taus_gbm):.4f}")
print(f"  GBM + preserve GPT top1: {np.mean(taus_gbm_pres):.4f}")

# Feature importances
print("\nTop features:")
imp = np.mean([m.feature_importances_ for m in fold_models], axis=0)
for fn, iv in sorted(zip(feature_names, imp), key=lambda x:-x[1])[:10]:
    print(f"  {fn}: {iv:.0f}")

# If GBM+preserve beats GPT, generate test submission
if np.mean(taus_gbm_pres) > np.mean(taus_gpt) + 0.005:
    print("\n→ GBM beats GPT, generating test predictions")
    X_test, g_test = build_dev_features(test_rows, gpt_test)
    test_preds = np.mean([m.predict(X_test) for m in fold_models], axis=0)
    out_path = "tasks/politicheadlines/results/test_predictions_gbm.csv"
    with open(out_path, "w") as f:
        f.write("id,task_1,task_2\n")
        for idx, r in enumerate(test_rows):
            mask = g_test == idx
            scores = test_preds[mask]
            pred_order = [i+1 for i in np.argsort(-scores)]
            gpt_top = gpt_test[r["id"]][0]
            if pred_order[0] != gpt_top:
                pred_order.remove(gpt_top); pred_order.insert(0, gpt_top)
            s = " ".join(f"t{t}" for t in pred_order)
            f.write(f"{r['id']},{s},{s}\n")
    print(f"Saved {out_path}")
else:
    print("\n→ GBM does NOT beat GPT. No test submission.")
