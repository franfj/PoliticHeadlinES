# Reproduction recipe

Step-by-step guide to reproduce the OpenCódice submission to
PoliticHeadlinES 2026 from scratch.

## 0. Prerequisites

* Python 3.10+
* A CUDA-capable GPU with at least 24 GB of VRAM (we used a single
  RTX A6000 on a rented vast.ai instance). CPU-only inference is
  feasible for the GPT and LightGBM components but training the
  XLM-RoBERTa-large pointwise ranker requires a GPU.
* The official task data placed in `data/` (see below).
* `OPENAI_API_KEY` exported in the environment for the GPT-based
  components.
* Optional: `GROQ_API_KEY` for the Llama-on-Groq pilot
  experiments.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

Major dependencies: `torch`, `transformers`, `lightgbm`, `openai`,
`pandas`, `numpy`, `scikit-learn`, `tqdm`.

## 2. Place the official data

The dataset must be obtained from the task organisers via CodaBench.
Place the files as follows:

```
data/
  train_public.csv
  dev_public.csv
  test_public/
    test_public.csv
    images/
      <article-id>.jpg
      ...
```

The CSVs have one row per article, with columns including the article
body, ten candidate headlines, an article image identifier and (for
train/dev) the gold candidate index.

## 3. Reproduce the submitted fusion

The submitted system fuses three score vectors. Reproduce each, then
combine.

### 3.1 Pointwise XLM-RoBERTa-large ranker

```bash
python src/train_ranker.py \
    --model xlm-roberta-large \
    --epochs 3 \
    --batch-size 8 \
    --max-len 256
```

The script trains a pointwise regressor whose label is the negative
rank position of the candidate among the ten candidates of the
article. Outputs per-candidate scores for the dev and test splits to
`results/test_predictions_ensemble_xlm-roberta-base.csv` (despite the
filename, this is the XLM-R-large checkpoint). Wall-clock time on a
single A6000: roughly 45 minutes.

### 3.2 LightGBM ranker on hand-crafted features

```bash
python src/feature_analysis.py   # writes features
python src/train_gbm.py            # trains GBM
python src/gbm_inference.py        # predicts on test
```

The features include token-overlap statistics, proper-noun overlap,
the XLM-R score from step 3.1, the position of the candidate in the
shuffled list, and a few more. Wall-clock time: under five minutes
on CPU. Output: `results/test_predictions_gbm.csv`.

### 3.3 GPT-5.4 listwise prompt

```bash
python src/gpt_ranker.py --split test
```

Sends each article to GPT-5.4 with the article body, the ten
candidates (labelled `t1` to `t10`), and a system prompt asking for a
permutation in strict JSON format. The exact prompt text is in
Section B.1 of the paper. Output: `results/test_predictions_gpt54.csv`.

### 3.4 Score-level fusion

```bash
python src/gbm_inference.py --alpha 0.3 \
    --out release/test_predictions.csv
```

The fusion takes the three score vectors, normalises each to zero
mean and unit variance per article, and combines them as
`s_fused = 0.7 * (s_xlmr + s_gbm) / 2 + 0.3 * s_gpt`, then sorts the
candidates by `s_fused`. The output CSV has three columns
(`id, task_1, task_2`) where each task column holds the
space-separated permutation `t<i>` strings.

## 4. Variants explored (not in the final submission)

* **Borda-count ensemble** (`src/build_v6_borda_ensemble.py`,
  `release/submission_v6_borda.zip`): combines rankings rather than
  scores; produced an identical top-1 to the score-level fusion on
  every test article.
* **Post-hoc top-1 picker** (`src/post_hoc_top1_picker.py`,
  `release/submission_v7_post_hoc.zip` and
  `release/submission_v8_post_hoc.zip`): re-ranks only the top
  position using a GPT-5.4 pairwise call on the two highest-scoring
  candidates; the leaderboard return was indistinguishable from the
  score-level fusion to within $\pm 0.0001$ nDCG@10.
* **Body-grounded post-hoc** (`src/post_hoc_v9_body.py`,
  `release/submission_v9_final.zip`): same idea, with the article
  body fed to the GPT call as additional context.
* **GPT-5.4 vision listwise prompt** (`src/gpt_vision_ranker.py`):
  the multimodal variant, attaching the article image as a base64
  `image_url` payload at `detail=low`. The variant only completed on
  roughly a third of the test set before hitting the vision-API rate
  limit and was not used in any final submission.
* **Qwen2-VL-2B image captions** (`src/vast_describe_images.py`):
  per-article Spanish-language image captions used as a side feature
  in early multimodal experiments. Stored in
  `results/descriptions_qwen2vl-2b.json` (1.7 MB).
* **Groq baselines** (`src/groq_ranker.py`,
  `src/groq_multimodal_ranker.py`): Llama-3.3-70B and Llama-4-Scout
  listwise prompts run on Groq for cost comparison; their scores were
  within $\pm 0.01$ nDCG@10 of GPT-5.4 on the dev set but were not
  used in the final fusion.

Every one of these variants has its predictions in `results/` and its
zipped submission archive in `release/` so that the convergence
pattern described in the paper can be verified directly from this
repository.

## 5. Evaluation

The official scoring program is run by the task organisers on the
private test set. For local evaluation against the public split:

```bash
python src/error_analysis.py \
    --predictions release/test_predictions.csv \
    --gold data/dev_public.csv
```

This emits a JSON summary (mean rank of the gold candidate, top-1
accuracy, position-wise agreement against alternative rankers) and a
per-article CSV of candidate cases for inspection. The
`results/error_analysis_summary.json` shipped with this repository
was produced by exactly this command.

## 6. Notes on reproducibility

* The GPT calls use `seed=42` and `temperature=0` in the
  `chat.completions` API. Repeated runs on the same data return
  identical permutations modulo OpenAI's internal non-determinism,
  which we observed at roughly $0.1$ per cent of articles.
* The XLM-R training is seeded but reproducibility across hardware is
  not guaranteed: the CUDA non-determinism in `scatter_add_` can
  produce per-instance score differences in the fourth decimal.
* The LightGBM training is deterministic on a fixed CPU thread count.
* The submitted `test_predictions.csv` in `release/` is the file
  uploaded to CodaBench. Comparing a freshly reproduced output
  against it is the cleanest end-to-end check.
