"""Train pointwise XLM-R ranker for PoliticHeadlinES.
Input: [article[:400]] [SEP] [title]
Output: scalar score (lower = better rank)
Regression on rank 1..10.
"""
import csv, re, json, os, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau, spearmanr

DEVICE = torch.device("cuda")

def parse(s):
    return [int(re.search(r'\d+', t).group()) for t in s.split()]

# Load both train files
rows = []
for path in ["tasks/politicheadlines/data/train_public.csv",
             "tasks/politicheadlines/data/test_public/train_public.csv"]:
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("y_true"):
                rows.append(r)
print(f"Total train rows: {len(rows)}")

# Build pointwise dataset: (article+title, rank_as_float 0-1 reversed so higher=better)
samples = []
for r in rows:
    try:
        order = parse(r["y_true"])
    except:
        continue
    art = (r.get("article_body","") or "")[:400]
    for rank, tid in enumerate(order, start=1):
        t = r.get(f"title_{tid}", "")
        if not t: continue
        # Target: higher = better. Use 1.0 - (rank-1)/9 in [0,1]
        target = 1.0 - (rank - 1) / 9.0
        samples.append((art, t, target, r["id"], tid))

print(f"Pointwise samples: {len(samples)}")

# Split by instance id (so no leakage)
ids_all = list({s[3] for s in samples})
tr_ids, va_ids = train_test_split(ids_all, test_size=0.1, random_state=42)
tr_set, va_set = set(tr_ids), set(va_ids)

tr_samples = [s for s in samples if s[3] in tr_set]
va_samples = [s for s in samples if s[3] in va_set]
print(f"Train: {len(tr_samples)}, Val: {len(va_samples)} ({len(va_ids)} instances)")

class DS(Dataset):
    def __init__(s, data, tok, ml=256):
        s.d, s.tok, s.ml = data, tok, ml
    def __len__(s): return len(s.d)
    def __getitem__(s, i):
        art, title, target, _, _ = s.d[i]
        e = s.tok(title, art, max_length=s.ml, padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": e["input_ids"].squeeze(),
                "attention_mask": e["attention_mask"].squeeze(),
                "labels": torch.tensor(target, dtype=torch.float)}

def eval_ranking(model, tok, va_data, ml=256, bs=32):
    model.eval()
    # Group by instance id
    by_id = {}
    for art, title, target, iid, tid in va_data:
        by_id.setdefault(iid, []).append((art, title, target, tid))
    taus, spears = [], []
    for iid, titles in by_id.items():
        # Batch predict
        inputs = [(t[0], t[1]) for t in titles]
        scores = []
        for i in range(0, len(inputs), bs):
            batch = inputs[i:i+bs]
            e = tok([x[1] for x in batch], [x[0] for x in batch],
                    max_length=ml, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**e).logits.squeeze(-1).cpu().numpy()
            scores.extend(out.tolist())
        # True rank: from target (higher=better → rank 1)
        true_order = sorted(range(len(titles)), key=lambda i: -titles[i][2])
        pred_order = sorted(range(len(titles)), key=lambda i: -scores[i])
        true_rank_arr = [0]*len(titles); pred_rank_arr = [0]*len(titles)
        for r, idx in enumerate(true_order): true_rank_arr[idx] = r+1
        for r, idx in enumerate(pred_order): pred_rank_arr[idx] = r+1
        tau, _ = kendalltau(true_rank_arr, pred_rank_arr)
        sp, _ = spearmanr(true_rank_arr, pred_rank_arr)
        if tau is not None: taus.append(tau)
        if sp is not None: spears.append(sp)
    return np.mean(taus), np.mean(spears)

for model_name in ["xlm-roberta-base", "PlanTL-GOB-ES/roberta-base-bne"]:
    short = model_name.split("/")[-1]
    print(f"\n=== Training {model_name} ===")
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, problem_type="regression"
        ).to(DEVICE)

        tr_dl = DataLoader(DS(tr_samples, tok), batch_size=32, shuffle=True, num_workers=2)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

        best_tau = -1
        for ep in range(3):
            model.train()
            total_loss = 0; n = 0
            for b in tr_dl:
                opt.zero_grad()
                out = model(input_ids=b["input_ids"].to(DEVICE),
                            attention_mask=b["attention_mask"].to(DEVICE),
                            labels=b["labels"].to(DEVICE))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += out.loss.item(); n += 1
            tau, sp = eval_ranking(model, tok, va_samples)
            print(f"  Epoch {ep+1}: loss={total_loss/n:.4f} val_kendall={tau:.4f} val_spearman={sp:.4f}")
            if tau > best_tau:
                best_tau = tau
                save_dir = f"tasks/politicheadlines/experiments/ranker_{short}"
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tok.save_pretrained(save_dir)
        print(f"BEST {short}: tau={best_tau:.4f}")
        del model; torch.cuda.empty_cache()
    except Exception as e:
        print(f"ERROR {model_name}: {e}")
        torch.cuda.empty_cache()

print("\nDONE")
