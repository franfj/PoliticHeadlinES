# PoliticHeadlinES @ IberLEF 2026

Source code, predictions and the reproduction recipe for the
**OpenCódice** submission to the
[PoliticHeadlinES 2026](https://www.codabench.org/competitions/13546/)
shared task at IberLEF 2026.

The task asks systems to rank ten Spanish-language candidate headlines for
a given political news article (body + image). Exactly one candidate is
the original published headline; the other nine are paraphrase or
off-topic distractors. The official metric is a position-aware variant of
nDCG@10.

## Submitted system

A score-level fusion of three independent rankers:

1. **Pointwise XLM-RoBERTa-large ranker** trained on the full 16k-article
   training set with MSE loss against the gold rank position
   (`src/train_ranker.py`).
2. **LightGBM ranker** on hand-crafted features that combine token-overlap
   statistics, proper-noun overlap, the XLM-R score, and positional
   priors (`src/train_gbm.py`, `src/feature_analysis.py`).
3. **GPT-5.4 listwise prompt** that returns a full permutation of the ten
   candidates (`src/gpt_ranker.py`).

The three score vectors are normalised per article and combined linearly
with mixing weight α = 0.3 in front of the GPT score.

The final submission scored **0.8841** nDCG@10 on both subtasks (text-only
and multimodal) and ranked **7th of 19 participating teams** (plus the
official baseline) on the public test leaderboard.

## Repository layout

```
src/                          Training, inference and analysis scripts
release/
  submission_v9_final.zip     Submission archive uploaded to CodaBench
  submission_v6_borda.zip     Earlier Borda-count fusion variant
  submission_v7_post_hoc.zip  Post-hoc Top-1 picker variant
  submission_v8_post_hoc.zip  Post-hoc Top-1 picker variant
  test_predictions.csv        Final test predictions (Task 1 + Task 2)
results/
  predictions_*.csv           Per-strategy dev/test predictions
  metrics_*.json              Per-strategy evaluation metrics
  descriptions_*.json         Image captions used in multimodal variants
  error_analysis_*.json       Top-1 agreement and paraphrase-density stats
RECIPE.md                     Step-by-step reproduction recipe
requirements.txt              Python dependencies
LICENSE                       MIT
```

## Quick reproduction

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...   # required only for GPT-based steps

# 1. Train the XLM-R-large pointwise ranker (single GPU, ~3 epochs)
python src/train_ranker.py

# 2. Train the LightGBM ranker on hand-crafted features
python src/train_gbm.py

# 3. Generate listwise rankings with GPT-5.4
python src/gpt_ranker.py --split test

# 4. Fuse the three score vectors and produce the submission file
python src/gbm_inference.py --alpha 0.3 --out release/test_predictions.csv
```

See [`RECIPE.md`](RECIPE.md) for the complete cookbook including data
preparation, environment variables, GPU notes and expected wall-clock
times.

The training set, development set and test set are not included in this
repository (they belong to the task organisers). Place the official files
at `data/train_public.csv`, `data/dev_public.csv`,
`data/test_public/test_public.csv`, and the test images under
`data/test_public/images/`.

## System description paper

The methodology and full set of strategies explored during the campaign
are documented in the system description paper, available at the
IberLEF 2026 working notes on CEUR-WS.

```bibtex
@inproceedings{rodrigogines2026politicheadlines,
  author    = {Rodrigo-Gin{\'e}s, Francisco-Javier and Chamorro-Padial, Jorge},
  title     = {{OpenC{\'o}dice at PoliticHeadlinES-IberLEF 2026: An Agentic
               Approach to Multimodal Political-Headline Ranking}},
  booktitle = {Proceedings of the Iberian Languages Evaluation Forum
               (IberLEF 2026), co-located with SEPLN 2026, CEUR-WS.org},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
