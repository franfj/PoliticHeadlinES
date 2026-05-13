#!/usr/bin/env python3
"""
PoliticHeadlinES — Multimodal Political Headline Ranking
IberLEF 2026 | CodaBench: codabench.org/competitions/13546/

Multimodal ranking of political headlines using text + images.
Language: Spanish

Primary approach: XLM-RoBERTa (text) + CLIP/ViT (image) fusion for ranking
Backup: GPT-5.4 text-only ranking via OpenAI API
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

TEXT_MODEL_NAME = "xlm-roberta-large"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
MAX_TEXT_LENGTH = 256
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10


# =====================================================================
# 1. DATA LOADING
# =====================================================================
def load_dataset(split: str = "train") -> pd.DataFrame:
    """Load PoliticHeadlinES dataset.

    Expected format: TSV/CSV with columns like
    [id, headline, image_path, rank/score, ...] or
    [id, headline_a, headline_b, image_path, preferred, ...]
    Adjust once real data schema is known.
    """
    for ext in [".tsv", ".csv", ".json", ".jsonl"]:
        path = DATA_DIR / f"{split}{ext}"
        if path.exists():
            if ext == ".jsonl":
                df = pd.read_json(path, lines=True)
            elif ext == ".json":
                df = pd.read_json(path)
            else:
                sep = "\t" if ext == ".tsv" else ","
                df = pd.read_csv(path, sep=sep)
            logger.info(f"Loaded {len(df)} rows from {path}")
            return df

    raise FileNotFoundError(
        f"No data for split={split} in {DATA_DIR}. "
        f"Download from CodaBench: codabench.org/competitions/13546/"
    )


def load_image(image_path: str) -> "PIL.Image.Image":
    """Load an image from path."""
    from PIL import Image
    full_path = Path(image_path)
    if not full_path.is_absolute():
        full_path = DATA_DIR / "images" / image_path
    return Image.open(full_path).convert("RGB")


# =====================================================================
# 2. TEXT-ONLY RANKING BASELINE (XLM-RoBERTa)
# =====================================================================
def train_text_ranker():
    """Fine-tune XLM-RoBERTa for headline ranking (pointwise regression or pairwise)."""
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel

    train_df = load_dataset("train")
    dev_df = load_dataset("dev")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    class HeadlineRanker(nn.Module):
        """Pointwise ranker: predicts a relevance/importance score per headline."""

        def __init__(self, model_name: str = TEXT_MODEL_NAME, dropout: float = 0.3):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden = self.encoder.config.hidden_size
            self.regressor = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

        def forward(self, input_ids, attention_mask, labels=None):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = out.last_hidden_state[:, 0, :]
            score = self.regressor(cls_emb).squeeze(-1)
            loss = None
            if labels is not None:
                loss = nn.MSELoss()(score, labels.float())
            return {"loss": loss, "score": score}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeadlineRanker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Determine score column (adapt to actual data)
    score_col = None
    for col_name in ["rank", "score", "relevance", "rating", "label"]:
        if col_name in train_df.columns:
            score_col = col_name
            break
    if score_col is None:
        logger.error("Cannot find score/rank column in data. Available: " + str(train_df.columns.tolist()))
        return None

    text_col = None
    for col_name in ["headline", "text", "title"]:
        if col_name in train_df.columns:
            text_col = col_name
            break

    output_dir = EXPERIMENTS_DIR / "text_ranker"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_corr = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        shuffled = train_df.sample(frac=1, random_state=42 + epoch)

        for i in range(0, len(shuffled), BATCH_SIZE):
            batch_df = shuffled.iloc[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch_df[text_col].tolist(),
                truncation=True,
                padding="max_length",
                max_length=MAX_TEXT_LENGTH,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor(batch_df[score_col].values, dtype=torch.float32).to(device)
            out = model(**enc, labels=labels)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out["loss"].item()

        # Eval: Spearman correlation
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for i in range(0, len(dev_df), BATCH_SIZE * 2):
                batch_df = dev_df.iloc[i : i + BATCH_SIZE * 2]
                enc = tokenizer(
                    batch_df[text_col].tolist(),
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_TEXT_LENGTH,
                    return_tensors="pt",
                ).to(device)
                out = model(**enc)
                all_scores.extend(out["score"].cpu().numpy())
                all_labels.extend(batch_df[score_col].values)

        from scipy.stats import spearmanr
        corr, _ = spearmanr(all_scores, all_labels)
        logger.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {total_loss / max(1, len(shuffled) // BATCH_SIZE):.4f} | "
            f"Spearman: {corr:.4f}"
        )
        if corr > best_corr:
            best_corr = corr
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    logger.info(f"Best dev Spearman: {best_corr:.4f}")
    return model


# =====================================================================
# 3. MULTIMODAL RANKING (Text + Image)
# =====================================================================
def train_multimodal_ranker():
    """Train multimodal ranker using XLM-RoBERTa (text) + CLIP ViT (image)."""
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor

    class MultimodalRanker(nn.Module):
        """Fuses XLM-RoBERTa text embeddings with CLIP image embeddings for ranking."""

        def __init__(self, fusion_dim: int = 512, dropout: float = 0.3):
            super().__init__()
            self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME)
            self.clip = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

            text_hidden = self.text_encoder.config.hidden_size
            clip_dim = self.clip.config.projection_dim  # 768

            self.text_proj = nn.Sequential(
                nn.Linear(text_hidden, fusion_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.image_proj = nn.Sequential(
                nn.Linear(clip_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.regressor = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, 1),
            )

        def forward(self, input_ids, attention_mask, pixel_values, labels=None):
            # Text features
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = self.text_proj(text_out.last_hidden_state[:, 0, :])

            # Image features via CLIP vision encoder
            image_feat = self.clip.get_image_features(pixel_values=pixel_values)
            image_feat = self.image_proj(image_feat)

            fused = torch.cat([text_feat, image_feat], dim=-1)
            score = self.regressor(fused).squeeze(-1)

            loss = None
            if labels is not None:
                loss = nn.MSELoss()(score, labels.float())
            return {"loss": loss, "score": score}

    train_df = load_dataset("train")
    dev_df = load_dataset("dev")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalRanker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Auto-detect columns
    text_col = next((c for c in ["headline", "text", "title"] if c in train_df.columns), train_df.columns[1])
    score_col = next((c for c in ["rank", "score", "relevance", "rating", "label"] if c in train_df.columns), None)
    image_col = next((c for c in ["image_path", "image", "image_file"] if c in train_df.columns), None)

    if score_col is None or image_col is None:
        logger.error(f"Missing required columns. Available: {train_df.columns.tolist()}")
        return None

    output_dir = EXPERIMENTS_DIR / "multimodal_ranker"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_corr = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        shuffled = train_df.sample(frac=1, random_state=42 + epoch)

        for i in range(0, len(shuffled), BATCH_SIZE):
            batch_df = shuffled.iloc[i : i + BATCH_SIZE]
            text_enc = tokenizer(
                batch_df[text_col].tolist(),
                truncation=True, padding="max_length", max_length=MAX_TEXT_LENGTH,
                return_tensors="pt",
            ).to(device)

            images = [load_image(p) for p in batch_df[image_col]]
            image_enc = clip_processor(images=images, return_tensors="pt")["pixel_values"].to(device)

            labels = torch.tensor(batch_df[score_col].values, dtype=torch.float32).to(device)

            out = model(
                input_ids=text_enc["input_ids"],
                attention_mask=text_enc["attention_mask"],
                pixel_values=image_enc,
                labels=labels,
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out["loss"].item()

        # Eval
        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for i in range(0, len(dev_df), BATCH_SIZE * 2):
                batch_df = dev_df.iloc[i : i + BATCH_SIZE * 2]
                text_enc = tokenizer(
                    batch_df[text_col].tolist(),
                    truncation=True, padding="max_length", max_length=MAX_TEXT_LENGTH,
                    return_tensors="pt",
                ).to(device)
                images = [load_image(p) for p in batch_df[image_col]]
                image_enc = clip_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
                out = model(
                    input_ids=text_enc["input_ids"],
                    attention_mask=text_enc["attention_mask"],
                    pixel_values=image_enc,
                )
                all_scores.extend(out["score"].cpu().numpy())
                all_labels.extend(batch_df[score_col].values)

        from scipy.stats import spearmanr
        corr, _ = spearmanr(all_scores, all_labels)
        logger.info(f"Epoch {epoch + 1} | Loss: {total_loss:.4f} | Spearman: {corr:.4f}")

        if corr > best_corr:
            best_corr = corr
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    logger.info(f"Best dev Spearman: {best_corr:.4f}")
    return model


# =====================================================================
# 4. PREDICTION FUNCTIONS
# =====================================================================
def predict_text_ranker(model_path: Optional[str] = None):
    """Generate ranking predictions using trained text-only model."""
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel

    if model_path is None:
        model_path = str(EXPERIMENTS_DIR / "text_ranker" / "best_model.pt")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    class HeadlineRanker(nn.Module):
        def __init__(self, model_name: str = TEXT_MODEL_NAME, dropout: float = 0.3):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden = self.encoder.config.hidden_size
            self.regressor = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

        def forward(self, input_ids, attention_mask, labels=None):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = out.last_hidden_state[:, 0, :]
            score = self.regressor(cls_emb).squeeze(-1)
            loss = None
            if labels is not None:
                loss = nn.MSELoss()(score, labels.float())
            return {"loss": loss, "score": score}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeadlineRanker()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    test_df = load_dataset("test")
    text_col = next((c for c in ["headline", "text", "title"] if c in test_df.columns), test_df.columns[1])

    all_scores = []
    with torch.no_grad():
        for i in range(0, len(test_df), BATCH_SIZE * 2):
            batch_df = test_df.iloc[i : i + BATCH_SIZE * 2]
            enc = tokenizer(
                batch_df[text_col].tolist(),
                truncation=True, padding="max_length", max_length=MAX_TEXT_LENGTH,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            all_scores.extend(out["score"].cpu().numpy())

    test_df["predicted_score"] = all_scores
    out_path = RESULTS_DIR / "predictions_text_xlmr.tsv"
    test_df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Text-only ranking predictions saved to {out_path}")
    return test_df


def predict_multimodal_ranker(model_path: Optional[str] = None):
    """Generate ranking predictions using trained multimodal model."""
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor

    if model_path is None:
        model_path = str(EXPERIMENTS_DIR / "multimodal_ranker" / "best_model.pt")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    class MultimodalRanker(nn.Module):
        def __init__(self, fusion_dim: int = 512, dropout: float = 0.3):
            super().__init__()
            self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME)
            self.clip = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            text_hidden = self.text_encoder.config.hidden_size
            clip_dim = self.clip.config.projection_dim
            self.text_proj = nn.Sequential(
                nn.Linear(text_hidden, fusion_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.image_proj = nn.Sequential(
                nn.Linear(clip_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.regressor = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, 1),
            )

        def forward(self, input_ids, attention_mask, pixel_values, labels=None):
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = self.text_proj(text_out.last_hidden_state[:, 0, :])
            image_feat = self.clip.get_image_features(pixel_values=pixel_values)
            image_feat = self.image_proj(image_feat)
            fused = torch.cat([text_feat, image_feat], dim=-1)
            score = self.regressor(fused).squeeze(-1)
            loss = None
            if labels is not None:
                loss = nn.MSELoss()(score, labels.float())
            return {"loss": loss, "score": score}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalRanker()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    test_df = load_dataset("test")
    text_col = next((c for c in ["headline", "text", "title"] if c in test_df.columns), test_df.columns[1])
    image_col = next((c for c in ["image_path", "image", "image_file"] if c in test_df.columns), None)

    all_scores = []
    with torch.no_grad():
        for i in range(0, len(test_df), BATCH_SIZE * 2):
            batch_df = test_df.iloc[i : i + BATCH_SIZE * 2]
            text_enc = tokenizer(
                batch_df[text_col].tolist(),
                truncation=True, padding="max_length", max_length=MAX_TEXT_LENGTH,
                return_tensors="pt",
            ).to(device)
            images = [load_image(p) for p in batch_df[image_col]]
            image_enc = clip_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            out = model(
                input_ids=text_enc["input_ids"],
                attention_mask=text_enc["attention_mask"],
                pixel_values=image_enc,
            )
            all_scores.extend(out["score"].cpu().numpy())

    test_df["predicted_score"] = all_scores
    out_path = RESULTS_DIR / "predictions_multimodal.tsv"
    test_df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Multimodal ranking predictions saved to {out_path}")
    return test_df


# =====================================================================
# 5. GPT-5.4 BACKUP (Text-only ranking)
# =====================================================================
def rank_with_gpt(few_shot_k: int = 3):
    """Rank headlines using GPT-5.4 as backup (text-only)."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    test_df = load_dataset("test")
    text_col = next((c for c in ["headline", "text", "title"] if c in test_df.columns), test_df.columns[1])

    # Few-shot examples
    examples_text = ""
    if few_shot_k > 0:
        try:
            train_df = load_dataset("train")
            score_col = next((c for c in ["rank", "score", "relevance", "rating", "label"] if c in train_df.columns), None)
            if score_col:
                samples = train_df.sample(min(few_shot_k, len(train_df)), random_state=42)
                for _, row in samples.iterrows():
                    examples_text += f'Headline: "{row[text_col]}"\nScore: {row[score_col]}\n\n'
        except FileNotFoundError:
            pass

    system_prompt = (
        "You are an expert at evaluating political news headlines in Spanish. "
        "Rate the political relevance/impact of each headline on a scale from 0 to 1. "
        "Consider newsworthiness, political significance, and public impact. "
        "Respond with ONLY a decimal number between 0 and 1, nothing else."
    )

    predictions = []
    for idx, row in test_df.iterrows():
        user_msg = ""
        if examples_text:
            user_msg += f"Examples:\n{examples_text}"
        user_msg += f'Rate this headline:\nHeadline: "{row[text_col]}"\nScore:'

        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=20,
            temperature=0.0,
        )
        try:
            score = float(response.choices[0].message.content.strip())
        except ValueError:
            score = 0.5
        predictions.append(score)

        if (idx + 1) % 50 == 0:
            logger.info(f"GPT-5.4 progress: {idx + 1}/{len(test_df)}")

    test_df["predicted_score"] = predictions
    out_path = RESULTS_DIR / "predictions_gpt54.tsv"
    test_df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"GPT-5.4 rankings saved to {out_path}")
    return test_df


# =====================================================================
# 6. EVALUATION
# =====================================================================
def evaluate_rankings(pred_df: pd.DataFrame):
    """Evaluate ranking predictions (Spearman, Kendall tau, NDCG)."""
    from scipy.stats import spearmanr, kendalltau

    score_col = next((c for c in ["rank", "score", "relevance", "rating", "label"] if c in pred_df.columns), None)
    pred_col = "predicted_score" if "predicted_score" in pred_df.columns else "prediction"

    if score_col is None:
        logger.error("No ground truth score column found.")
        return None

    y_true = pred_df[score_col].values
    y_pred = pred_df[pred_col].values

    spearman, sp_p = spearmanr(y_true, y_pred)
    kendall, kt_p = kendalltau(y_true, y_pred)

    logger.info(f"Spearman: {spearman:.4f} (p={sp_p:.4e})")
    logger.info(f"Kendall tau: {kendall:.4f} (p={kt_p:.4e})")

    return {"spearman": spearman, "kendall_tau": kendall}


# =====================================================================
# 7. MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="PoliticHeadlinES — Headline Ranking")
    parser.add_argument(
        "--mode",
        choices=[
            "train-text", "predict-text",
            "train-multimodal", "predict-multimodal",
            "gpt", "evaluate",
        ],
        default="train-text",
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--few-shot-k", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "train-text":
        train_text_ranker()
    elif args.mode == "predict-text":
        predict_text_ranker(model_path=args.model_path)
    elif args.mode == "train-multimodal":
        train_multimodal_ranker()
    elif args.mode == "predict-multimodal":
        predict_multimodal_ranker(model_path=args.model_path)
    elif args.mode == "gpt":
        rank_with_gpt(few_shot_k=args.few_shot_k)
    elif args.mode == "evaluate":
        pred_path = RESULTS_DIR / "predictions_multimodal.tsv"
        if not pred_path.exists():
            pred_path = RESULTS_DIR / "predictions_text_xlmr.tsv"
        if not pred_path.exists():
            pred_path = RESULTS_DIR / "predictions_gpt54.tsv"
        pred_df = pd.read_csv(pred_path, sep="\t")
        evaluate_rankings(pred_df)


if __name__ == "__main__":
    main()
