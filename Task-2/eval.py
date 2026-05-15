"""
Inference + evaluation for the Qwen2.5-1.5B tweet generation model.

Fixes over v1:
  - Loads the correct model (Qwen2.5-1.5B-Instruct + qwen15b_tweet_final adapter)
  - Uses tokenizer.apply_chat_template with the SAME prompt as training
  - Beam search instead of sampling (metrics reward mode-seeking)
  - Real metrics: BLEU 1-4 (nltk), ROUGE-1/2/L (rouge-score), CIDEr (pycocoevalcap)
  - Submission preserves all original test columns + carries 'id'
  - Test-set VLM enrichment: auto-enriches test rows that have media URLs
"""

import os
import re
import gc
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from prompt_utils import build_messages

# ---------------------------------------------------------------------------
# Config — update MODEL_NAME if you change the base during training
# ---------------------------------------------------------------------------
MODEL_NAME        = "Qwen/Qwen2.5-1.5B-Instruct"
TRAINED_MODEL_DIR = "./qwen15b_tweet_final"
TEST_BRANDS_PATH  = "test_unseen_brands.csv"
TEST_TIME_PATH    = "test_unseen_time.csv"
MAX_NEW_TOKENS    = 100
NUM_BEAMS         = 4

print("=" * 70)
print("TWEET GENERATION EVALUATION")
print("=" * 70)

# ---------------------------------------------------------------------------
# Shared 4-bit config (reused for both VLM and LLM)
# ---------------------------------------------------------------------------
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ---------------------------------------------------------------------------
# VLM enrichment for test rows
# Runs FIRST, then frees VRAM, then LLM is loaded — avoids 3 GB+ co-residence.
# Uses Qwen2.5-VL-3B (same model as training-side enrichment in enrich_vlm.py)
# so the LLM sees the same caption distribution at train and test time.
# ---------------------------------------------------------------------------
def maybe_enrich_test(df: pd.DataFrame) -> pd.DataFrame:
    """Run Qwen2.5-VL-3B enrichment on test rows that have media but no description."""
    if "vlm_description" not in df.columns:
        df["vlm_description"] = ""

    needs = df["vlm_description"].isna() | (df["vlm_description"].astype(str).str.strip() == "")
    has_media = df["media"].notna() & (df["media"].astype(str).str.strip() != "")
    todo = df.index[needs & has_media].tolist()

    if not todo:
        return df

    print(f"  Enriching {len(todo)} test rows with Qwen2.5-VL-3B...")
    try:
        import tempfile
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
        )
        import enrich_vlm as ev

        vlm_processor = AutoProcessor.from_pretrained(ev.MODEL_ID)
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ev.MODEL_ID,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=QUANT_CONFIG,
            low_cpu_mem_usage=True,
        )
        vlm_model.eval()
        tmp = os.path.join(tempfile.gettempdir(), "qwen_test_img.jpg")

        for idx in tqdm(todo, desc="Test VLM"):
            row = df.loc[idx]
            url = ev.extract_image_url(str(row.get("media", "")))
            if url and ev.download_to_disk(url, tmp):
                try:
                    desc = ev.run_vlm(vlm_model, vlm_processor, tmp)
                    df.at[idx, "vlm_description"] = desc or "media could not be processed"
                except Exception as e:
                    print(f"    [vlm error] {e}")
                    df.at[idx, "vlm_description"] = "media could not be processed"
                finally:
                    ev.delete_temp(tmp)
            else:
                df.at[idx, "vlm_description"] = "no media"

        # Free VLM before LLM loads
        del vlm_model, vlm_processor
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  VLM freed — VRAM headroom restored.")
    except Exception as e:
        print(f"  VLM enrichment skipped: {e}")

    return df


# ---------------------------------------------------------------------------
# LLM loader — called AFTER VLM enrichment so we never have both in VRAM
# ---------------------------------------------------------------------------
def load_llm():
    print("\nLoading fine-tuned LLM...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=QUANT_CONFIG,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_DIR)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_DIR, trust_remote_code=True)
    print("LLM loaded.")
    return model, tokenizer

# ---------------------------------------------------------------------------
# 3. Load test data
# ---------------------------------------------------------------------------
print("\nLoading test datasets...")

def load_test(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  {path} not found — using dummy data for demo")
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "date": ["2024-01-15 10:00:00"] * 3,
            "likes": [100, 500, 50],
            "username": ["Nike", "adidas", "PUMA"],
            "media": [""] * 3,
            "inferred company": ["nike", "adidas", "puma"],
            "content": ["Sample tweet 1", "Sample tweet 2", "Sample tweet 3"],
        })
    # normalise column names
    df.rename(columns={"inferred company": "company", "date": "date"}, inplace=True)
    if "company" not in df.columns and "inferred company" in df.columns:
        df.rename(columns={"inferred company": "company"}, inplace=True)
    return df

test_brands = load_test(TEST_BRANDS_PATH)
test_time   = load_test(TEST_TIME_PATH)

print(f"  Unseen brands : {len(test_brands)} rows")
print(f"  Unseen time   : {len(test_time)} rows")

# Enrich test media (loads Qwen2.5-VL, runs, then frees it before LLM loads)
test_brands = maybe_enrich_test(test_brands)
test_time   = maybe_enrich_test(test_time)

# Now load the LLM — VLM is already freed, no co-residence in VRAM
model, tokenizer = load_llm()

# ---------------------------------------------------------------------------
# 4. Tweet generation
# ---------------------------------------------------------------------------
def clean_tweet(text: str) -> str:
    """Strip common LLM preamble artifacts."""
    text = text.strip()
    text = re.sub(r'^["\'`]+|["\'`]+$', '', text).strip()
    # Remove "Sure, here's a tweet:" style preambles
    text = re.sub(r'^(sure[,!]?\s*here[\'s]*\s*(is|a)[^:]*:?\s*)', '', text,
                  flags=re.IGNORECASE).strip()
    # Truncate at first double newline
    text = text.split("\n\n")[0].strip()
    return text


def generate_tweet(row: dict) -> str:
    messages = build_messages(row, include_response=False)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt_text, return_tensors="pt", truncation=True, max_length=512
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][prompt_len:]
    tweet = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return clean_tweet(tweet)


def generate_batch(test_df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    print(f"\nGenerating: {task_name}")
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        generated = generate_tweet(row.to_dict())
        results.append({
            "id":               row.get("id", ""),
            "date":             row.get("date", ""),
            "likes":            row.get("likes", ""),
            "username":         row.get("username", ""),
            "media":            row.get("media", ""),
            "inferred company": row.get("company", row.get("inferred company", "")),
            "generated":        generated,
            "actual":           str(row.get("content", "")),
        })
    return pd.DataFrame(results)

print("\n" + "=" * 70)
preds_brands = generate_batch(test_brands, "Unseen Brands")
preds_time   = generate_batch(test_time,   "Unseen Time Period")

preds_brands.to_csv("predictions_unseen_brands.csv", index=False)
preds_time.to_csv("predictions_unseen_time.csv",   index=False)
print("Predictions saved.")

# ---------------------------------------------------------------------------
# 5. Metrics: BLEU 1-4, ROUGE-1/2/L, CIDEr
# ---------------------------------------------------------------------------
def compute_bleu(preds: list, refs: list) -> dict:
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smoother = SmoothingFunction().method3
        refs_tok  = [[r.lower().split()] for r in refs]
        hyps_tok  = [p.lower().split() for p in preds]
        return {
            "BLEU-1": corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=smoother),
            "BLEU-2": corpus_bleu(refs_tok, hyps_tok, weights=(.5,.5,0,0), smoothing_function=smoother),
            "BLEU-3": corpus_bleu(refs_tok, hyps_tok, weights=(1/3,1/3,1/3,0), smoothing_function=smoother),
            "BLEU-4": corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=smoother),
        }
    except ImportError:
        print("  [warn] nltk not installed — pip install nltk")
        return {}


def compute_rouge(preds: list, refs: list) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                          use_stemmer=True)
        scores = [scorer.score(r, p) for r, p in zip(refs, preds)]
        return {
            "ROUGE-1": np.mean([s["rouge1"].fmeasure for s in scores]),
            "ROUGE-2": np.mean([s["rouge2"].fmeasure for s in scores]),
            "ROUGE-L": np.mean([s["rougeL"].fmeasure for s in scores]),
        }
    except ImportError:
        print("  [warn] rouge-score not installed — pip install rouge-score")
        return {}


def compute_cider(preds: list, refs: list) -> dict:
    try:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()
        gts = {i: [r] for i, r in enumerate(refs)}
        res = {i: [p] for i, p in enumerate(preds)}
        score, _ = scorer.compute_score(gts, res)
        return {"CIDEr": score}
    except ImportError:
        print("  [warn] pycocoevalcap not installed — pip install pycocoevalcap")
        return {}
    except Exception as e:
        print(f"  [warn] CIDEr failed: {e}")
        return {}


def print_metrics(df: pd.DataFrame, label: str):
    preds = df["generated"].tolist()
    refs  = df["actual"].tolist()

    print(f"\n{'='*70}")
    print(f"RESULTS — {label}")
    print(f"{'='*70}")

    for name, val in compute_bleu(preds, refs).items():
        print(f"  {name:<12}: {val:.4f}")
    for name, val in compute_rouge(preds, refs).items():
        print(f"  {name:<12}: {val:.4f}")
    for name, val in compute_cider(preds, refs).items():
        print(f"  {name:<12}: {val:.4f}")

    avg_len_gen = np.mean([len(p.split()) for p in preds])
    avg_len_ref = np.mean([len(r.split()) for r in refs])
    print(f"  Gen length  : {avg_len_gen:.1f} words (ref: {avg_len_ref:.1f})")


print_metrics(preds_brands, "Unseen Brands")
print_metrics(preds_time,   "Unseen Time Period")

# ---------------------------------------------------------------------------
# 6. Submission files — preserve id + all original input columns
# ---------------------------------------------------------------------------
def make_submission(preds_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "id":               preds_df["id"],
        "date":             preds_df["date"],
        "likes":            preds_df["likes"],
        "username":         preds_df["username"],
        "media":            preds_df["media"],
        "inferred company": preds_df["inferred company"],
        "content":          preds_df["generated"],
    })

make_submission(preds_brands).to_csv("submission_unseen_brands.csv", index=False)
make_submission(preds_time).to_csv("submission_unseen_time.csv",   index=False)
print("\nSubmission files saved: submission_unseen_brands.csv, submission_unseen_time.csv")

# Cleanup
del model, base_model
gc.collect()
torch.cuda.empty_cache()
print("\nDone.")
