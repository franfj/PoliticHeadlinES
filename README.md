# PoliticHeadlinES @ IberLEF 2026

Source code, predictions and reproduction recipe for our submission to the
[PoliticHeadlinES 2026](https://www.codabench.org/competitions/13546/) shared task at IberLEF 2026.

The task asks systems to rank ten Spanish-language candidate headlines for a
given political news article (body + image). Exactly one candidate is the
original published headline; the other nine are paraphrase or off-topic
distractors.

## Submitted system

A score-level fusion of three independent rankers:

1. **Pointwise XLM-RoBERTa-large ranker** trained on the full 16k-article
   training set with MSE loss against the gold rank position
   (`src/train_ranker.py`).
2. **LightGBM ranker** on hand-crafted features that combine token-overlap
   statistics, proper-noun overlap, the XLM-R score, and positional priors
   (`src/train_gbm.py`, `src/feature_analysis.py`).
3. **GPT-5.4 listwise prompt** that returns a full permutation of the ten
   candidates (`src/gpt_ranker.py`).

The three score vectors are normalised per article and combined linearly
with mixing weight α = 0.3 in front of the GPT score.

The final submission scored **0.8841** on the official PA-nDCG@K metric and
ranked **6th of 19** on the public test leaderboard.

## Repository layout

```
src/                    Training and inference scripts
release/
  predictions.csv       Test predictions of the submitted system
  submission.zip        Submission archive uploaded to CodaBench
README.md               This file
requirements.txt        Python dependencies
LICENSE                 MIT
```

## Reproducing the submission

The training set, development set and test set are not included in this
repository (they belong to the task organisers). Place the official files at
`data/test_public/train_public.csv`, `data/dev_public.csv`,
`data/test_public/test_public.csv`, and the test images under
`data/test_public/images/`.

```bash
pip install -r requirements.txt

# 1. Train the XLM-R-large pointwise ranker (single GPU, ~3 epochs)
python src/train_ranker.py

# 2. Train the LightGBM ranker on hand-crafted features
python src/train_gbm.py

# 3. Generate listwise rankings with GPT-5.4 (requires OPENAI_API_KEY)
python src/gpt_ranker.py --split test

# 4. Fuse the three score vectors and produce the submission file
python src/gbm_inference.py --alpha 0.3 --out release/predictions.csv
```

## Citation

If you use this code, please cite the system description paper.

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
