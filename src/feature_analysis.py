"""Feature analysis: do simple metadata features predict rank?"""
import csv, re, numpy as np
from scipy.stats import kendalltau, spearmanr
from collections import defaultdict

def parse(s):
    return [int(re.search(r'\d+', t).group()) for t in s.split()]

# Load big train (16k) + small train (100)
all_data = []
for path in ["tasks/politicheadlines/data/train_public.csv",
             "tasks/politicheadlines/data/test_public/train_public.csv"]:
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("y_true"):
                all_data.append(row)
print(f"Loaded {len(all_data)} instances with y_true")

# Feature extraction per title
def feats(title, article):
    t = title or ""
    words = t.split()
    return {
        "length": len(t),
        "words": len(words),
        "upper_ratio": sum(1 for c in t if c.isupper()) / max(len(t), 1),
        "question": 1 if "?" in t or "¿" in t else 0,
        "exclaim": 1 if "!" in t or "¡" in t else 0,
        "colon": 1 if ":" in t else 0,
        "quote": 1 if '"' in t or "'" in t or '“' in t else 0,
        "caps_word": sum(1 for w in words if len(w)>1 and w.isupper()),
        # Lexical overlap with article (rough)
        "article_overlap": len(set(t.lower().split()) & set((article or "")[:500].lower().split())) / max(len(words), 1),
    }

# Collect: for each instance, compute rank (1=best) and features for each title
rank_feats = defaultdict(list)  # feature_name -> list of (rank, value)

for row in all_data:
    if not row.get("y_true"): continue
    try:
        order = parse(row["y_true"])
    except:
        continue
    article = row.get("article_body", "")
    for rank, title_idx in enumerate(order, start=1):
        t = row.get(f"title_{title_idx}", "")
        if not t: continue
        f = feats(t, article)
        for k, v in f.items():
            rank_feats[k].append((rank, v))

print("\n=== Feature → Rank correlations (Spearman) ===")
print("  (negative = higher feature = better rank; positive = higher = worse rank)")
rows = []
for fname, pairs in rank_feats.items():
    if not pairs: continue
    ranks, values = zip(*pairs)
    sp, p = spearmanr(values, ranks)
    rows.append((fname, sp, p, len(pairs)))
rows.sort(key=lambda x: abs(x[1]), reverse=True)
for fname, sp, p, n in rows:
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    print(f"  {fname:20s}: ρ={sp:+.4f} p={p:.2e} n={n} {sig}")

# Mean feature value per rank position
print("\n=== Mean feature values per rank ===")
print(f"{'feature':20s} " + " ".join(f"r{i:<7d}" for i in range(1,11)))
for fname, _, _, _ in rows[:6]:
    pairs = rank_feats[fname]
    by_rank = defaultdict(list)
    for r, v in pairs:
        by_rank[r].append(v)
    means = [np.mean(by_rank[r]) for r in range(1,11)]
    print(f"{fname:20s} " + " ".join(f"{m:7.3f} " for m in means))
