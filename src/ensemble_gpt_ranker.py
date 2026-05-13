"""Ensemble GPT ranking + learned ranker for PoliticHeadlinES.

Strategy:
- GPT clavó rank 1 al 100% on dev — preserve it.
- For positions 2-10, weighted combination of GPT rank and learned ranker score.
- Evaluate multiple fusion strategies on dev, pick best.
- Apply best to test.
"""
import csv, re, json, os, torch, numpy as np
from scipy.stats import kendalltau, spearmanr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse(s):
    return [int(re.search(r'\d+', t).group()) for t in s.split()]

def load_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))

# Load dev with GPT predictions
dev = {r["id"]: r for r in load_csv_rows("tasks/politicheadlines/data/dev_public.csv")}
gpt_dev = {}
for r in load_csv_rows("tasks/politicheadlines/results/predictions_gpt54_text_3shot_dev.csv"):
    gpt_dev[r["id"]] = r["y_pred"].split()

# Load test (no y_true) and GPT test predictions
test = {r["id"]: r for r in load_csv_rows("tasks/politicheadlines/data/test_public/test_public.csv")}
gpt_test = {}
for r in load_csv_rows("tasks/politicheadlines/results/test_predictions_gpt54.csv"):
    gpt_test[r["id"]] = r["y_pred"].split()

print(f"Dev: {len(dev)}, Test: {len(test)}")

# Score titles with trained ranker
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def score_with_ranker(model_dir, instances_map, gpt_map, bs=32, ml=256):
    """Return {id: [score_for_title_1, ..., score_for_title_10]} (higher = better)."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()
    scores = {}
    all_inputs = []
    keys = []  # (id, title_idx)
    for iid, row in instances_map.items():
        art = (row.get("article_body","") or "")[:400]
        for ti in range(1, 11):
            t = row.get(f"title_{ti}", "")
            all_inputs.append((t, art))
            keys.append((iid, ti))

    outs = []
    with torch.no_grad():
        for i in range(0, len(all_inputs), bs):
            batch = all_inputs[i:i+bs]
            e = tok([x[0] for x in batch], [x[1] for x in batch],
                    max_length=ml, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            s = model(**e).logits.squeeze(-1).cpu().numpy().tolist()
            outs.extend(s)
    for (iid, ti), s in zip(keys, outs):
        scores.setdefault(iid, [0]*10)[ti-1] = s
    del model; torch.cuda.empty_cache()
    return scores

def eval_on_dev(pred_map):
    taus, sps = [], []
    for iid, d in dev.items():
        if iid not in pred_map: continue
        true_order = parse(d["y_true"])
        pred_order = pred_map[iid]
        true_rank = {t:i+1 for i,t in enumerate(true_order)}
        pred_rank = {t:i+1 for i,t in enumerate(pred_order)}
        tau,_ = kendalltau([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
        sp,_ = spearmanr([true_rank[t] for t in range(1,11)], [pred_rank[t] for t in range(1,11)])
        taus.append(tau); sps.append(sp)
    return np.mean(taus), np.mean(sps)

def fuse(gpt_order, ranker_scores, alpha, preserve_top1=True):
    """alpha in [0,1]: 0 = pure ranker, 1 = pure GPT.
    Combines via weighted sum of rank positions (inverted)."""
    gpt_rank = {t: i+1 for i, t in enumerate(gpt_order)}  # 1=best
    ranker_rank_order = sorted(range(1,11), key=lambda i: -ranker_scores[i-1])
    ranker_rank = {t: i+1 for i, t in enumerate(ranker_rank_order)}
    combined = {}
    for t in range(1, 11):
        combined[t] = alpha * gpt_rank[t] + (1-alpha) * ranker_rank[t]
    out = sorted(range(1,11), key=lambda t: combined[t])
    if preserve_top1:
        gpt_top = gpt_order[0]
        if out[0] != gpt_top:
            out.remove(gpt_top)
            out.insert(0, gpt_top)
    return out

# Run if ranker exists
for short in ["xlm-roberta-base", "roberta-base-bne"]:
    model_dir = f"tasks/politicheadlines/experiments/ranker_{short}"
    if not os.path.isdir(model_dir):
        print(f"SKIP {short} — not trained yet")
        continue
    print(f"\n=== Using {short} ===")
    dev_scores = score_with_ranker(model_dir, dev, gpt_dev)

    # Baseline GPT
    gpt_pred = {iid: [int(re.search(r'\d+',t).group()) for t in g] for iid, g in gpt_dev.items()}
    k, s = eval_on_dev(gpt_pred)
    print(f"  GPT only      : Kendall={k:.4f} Spearman={s:.4f}")
    # Baseline ranker
    ranker_pred = {iid: sorted(range(1,11), key=lambda t: -dev_scores[iid][t-1]) for iid in dev_scores}
    k, s = eval_on_dev(ranker_pred)
    print(f"  Ranker only   : Kendall={k:.4f} Spearman={s:.4f}")

    # Grid search alpha
    best_alpha, best_k = None, -1
    for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        for preserve in [True, False]:
            fused = {iid: fuse([int(re.search(r'\d+',t).group()) for t in gpt_dev[iid]],
                               dev_scores[iid], alpha, preserve) for iid in dev_scores}
            k, s = eval_on_dev(fused)
            print(f"  α={alpha:.1f} preserve={preserve}: Kendall={k:.4f} Spearman={s:.4f}")
            if k > best_k:
                best_k, best_alpha, best_preserve = k, alpha, preserve
    print(f"  BEST fusion: α={best_alpha} preserve={best_preserve}: Kendall={best_k:.4f}")

    # Apply to test
    test_scores = score_with_ranker(model_dir, test, gpt_test)
    test_out = {}
    for iid in gpt_test:
        if iid not in test_scores: continue
        gpt_order = [int(re.search(r'\d+',t).group()) for t in gpt_test[iid]]
        test_out[iid] = fuse(gpt_order, test_scores[iid], best_alpha, best_preserve)

    # Save as submission CSV
    out_path = f"tasks/politicheadlines/results/test_predictions_ensemble_{short}.csv"
    with open(out_path, "w") as f:
        f.write("id,task_1,task_2\n")
        for iid, order in test_out.items():
            s = " ".join(f"t{t}" for t in order)
            f.write(f"{iid},{s},{s}\n")
    print(f"  Saved {out_path}")

print("\nDONE")
