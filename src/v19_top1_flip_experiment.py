"""v19: Flip GPT top-1 only when GBM has STRONG disagreement signal.
Rationale: metric is ~0.9 * top1 + 0.1 * nDCG. Each correctly flipped top-1 = +0.018.
But wrong flip = -0.018. Only flip when confident.

Confidence signal: GBM's top-1 title has STRONGLY higher max_sim than GPT's top-1,
OR GBM top-1 score >> GBM second-ranked score.
"""
import csv, re, pickle, numpy as np, json
from difflib import SequenceMatcher
from scipy.stats import kendalltau

def parse(s): return [int(re.search(r'\d+', t).group()) for t in s.split()]
def tok(s): return re.findall(r'\w+', (s or '').lower())
def ratio(a,b): return SequenceMatcher(None, a or '', b or '').ratio()

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
    return np.array(feats, dtype=np.float32), max_sim

# Evaluate flip strategy on DEV first
dev_rows = list(csv.DictReader(open("tasks/politicheadlines/data/dev_public.csv")))
gpt_dev = {r["id"]: parse(r["y_pred"]) for r in csv.DictReader(open("tasks/politicheadlines/results/predictions_gpt54_text_3shot_dev.csv"))}

print("Analyzing top-1 flip signals on DEV...")
for FLIP_THRESHOLD in [0.01, 0.02, 0.05, 0.1, 0.2]:
    flipped = 0; correct_flip = 0; wrong_flip = 0; unchanged_correct = 0
    for r in dev_rows:
        feats, max_sim = features(r, gpt_dev[r["id"]])
        preds = np.mean([m.predict(feats) for m in models], axis=0)
        gpt_top = gpt_dev[r["id"]][0]
        gbm_top = int(np.argmax(preds)) + 1
        true_order = parse(r["y_true"])
        true_top = true_order[0]

        gap = preds[gbm_top-1] - preds[gpt_top-1]  # how much GBM prefers its top over GPT's
        flip = gbm_top != gpt_top and gap > FLIP_THRESHOLD
        if flip:
            flipped += 1
            if gbm_top == true_top: correct_flip += 1
            elif gpt_top == true_top: wrong_flip += 1
        else:
            if gpt_top == true_top: unchanged_correct += 1

    total_correct = correct_flip + unchanged_correct
    print(f"  gap>{FLIP_THRESHOLD:.2f}: flipped {flipped}/50, correct_flips {correct_flip}, wrong_flips {wrong_flip}, "
          f"top-1 acc after: {100*total_correct/50:.0f}% (GPT baseline: {100*sum(1 for r in dev_rows if gpt_dev[r['id']][0]==parse(r['y_true'])[0])/50:.0f}%)")

# Now generate test submission with best threshold
BEST_GAP = 0.05  # adjust after seeing dev results
print(f"\nGenerating test with flip threshold {BEST_GAP}")

test_rows = list(csv.DictReader(open("tasks/politicheadlines/data/test_public/test_public.csv")))
gpt_test = {r["id"]: parse(r["y_pred"]) for r in csv.DictReader(open("tasks/politicheadlines/results/test_predictions_gpt54.csv"))}

# Also load GBM fusion predictions (our current v18)
gbm_fusion = {r["id"]: parse(r["task_1"]) for r in csv.DictReader(open("tasks/politicheadlines/results/test_predictions_gbm_fusion_a03.csv"))}

flipped_test = 0
out_path = "tasks/politicheadlines/results/test_predictions_v19_flip.csv"
with open(out_path, "w") as f:
    f.write("id,task_1,task_2\n")
    for idx, r in enumerate(test_rows):
        feats, max_sim = features(r, gpt_test[r["id"]])
        preds = np.mean([m.predict(feats) for m in models], axis=0)
        gpt_top = gpt_test[r["id"]][0]
        gbm_top = int(np.argmax(preds)) + 1
        gap = preds[gbm_top-1] - preds[gpt_top-1]

        # Start from GBM fusion ordering (α=0.3 + preserve top1)
        order = gbm_fusion[r["id"]]
        if gbm_top != gpt_top and gap > BEST_GAP:
            # FLIP: move gbm_top to position 1
            order = [t for t in order if t != gbm_top]
            order.insert(0, gbm_top)
            flipped_test += 1
        s = " ".join(f"t{t}" for t in order)
        f.write(f"{r['id']},{s},{s}\n")
        if (idx+1) % 1000 == 0: print(f"  {idx+1}/{len(test_rows)}")

print(f"Flipped {flipped_test}/{len(test_rows)} top-1 positions")
print(f"Saved {out_path}")
