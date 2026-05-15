"""
Qwen2.5-1.5B-Instruct QLoRA fine-tune for tweet generation.

Changes from v1 (0.5B):
  - Base model : Qwen2.5-0.5B → Qwen2.5-1.5B-Instruct
  - LoRA       : r=8, 2 modules → r=16, 4 attention modules
  - Filtering  : character-length heuristic → SFTConfig max_seq_length (token-based)
  - Eval split : added 5% hold-out to monitor overfitting
  - Scheduler  : config uses SFTConfig (not TrainingArguments) consistently
  - Output dir : ./qwen15b_tweet_final
"""

import json
import os
import gc
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

VRAM_GUARD_PCT = 98   # trigger threshold


class VRAMGuardCallback(TrainerCallback):
    """
    After each optimizer step (weights just modified), check VRAM via PyTorch API
    (zero subprocess overhead). If usage >= VRAM_GUARD_PCT, free the CUDA allocator
    cache before the next step starts.
    Only dead-tensor cache is freed — model weights and live gradients are untouched.
    """
    def _vram_pct(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        reserved = torch.cuda.memory_reserved(0)
        total    = torch.cuda.get_device_properties(0).total_memory
        return reserved / total * 100

    def on_step_end(self, args, state, control, **kwargs):
        pct = self._vram_pct()
        if pct >= VRAM_GUARD_PCT:
            gc.collect()
            torch.cuda.empty_cache()
            after = self._vram_pct()
            print(
                f"\n[VRAMGuard] step {state.global_step}  |  "
                f"{pct:.1f}% -> {after:.1f}%  ({pct - after:.1f}% freed)",
                flush=True,
            )

MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH   = "llm_train_data.jsonl"
CSV_PATH    = "train_enriched.csv"   # for metadata-aware split (falls back to train.csv)
OUTPUT_DIR  = "./qwen15b_tweet_final"
MAX_SEQ_LEN = 256                    # was 512; tweets are short, 2x speed/memory win
SEED        = 42

# Regime-mirroring eval split (mirrors competition's unseen-brands + unseen-time regimes)
EVAL_BRAND_FRAC = 0.05   # hold out this fraction of distinct brands fully (unseen-brand mirror)
EVAL_TIME_FRAC  = 0.05   # from remaining rows, hold out latest fraction by date (unseen-time mirror)


def make_regime_split(df: pd.DataFrame, eval_brand_frac: float, eval_time_frac: float, seed: int):
    """
    Split a tweet metadata dataframe into train and eval indices that mirror
    the competition's two evaluation regimes simultaneously:

      eval = (all rows from a random held-out subset of brands)
             UNION
             (latest `eval_time_frac` of rows from the remaining brands by date)

    Returns parallel lists of integer positions into df after reset_index.
    """
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)

    brand_col = "inferred company" if "inferred company" in df.columns else "company"
    if brand_col not in df.columns:
        # Fallback to username if no company column exists
        brand_col = "username"

    brands = pd.Series(df[brand_col]).dropna().unique()
    n_held_brands = max(1, int(round(len(brands) * eval_brand_frac)))
    held_brands = set(rng.choice(brands, size=n_held_brands, replace=False).tolist())
    brand_mask = df[brand_col].isin(held_brands)

    remaining = df.loc[~brand_mask].copy()
    remaining["_dt"] = pd.to_datetime(remaining["date"], errors="coerce")
    remaining = remaining.sort_values("_dt", kind="stable")
    n_held_time = int(round(len(remaining) * eval_time_frac))
    time_idx = remaining.tail(n_held_time).index if n_held_time else pd.Index([])

    eval_pos = sorted(set(df.index[brand_mask].tolist()) | set(time_idx.tolist()))
    train_pos = sorted(set(df.index.tolist()) - set(eval_pos))
    return train_pos, eval_pos, {
        "held_brands": sorted(held_brands)[:5],
        "n_held_brands": n_held_brands,
        "n_held_time_rows": n_held_time,
        "brand_col": brand_col,
    }


def main():
    # ------------------------------------------------------------------
    # 1. Load & format dataset
    # ------------------------------------------------------------------
    print("Loading dataset...")
    data = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"  Raw JSONL records: {len(data)}")

    # Load matching CSV for metadata (brand + date) used by the regime split.
    # Apply the SAME filter as prep_llm_data.py so rows align with JSONL.
    csv_to_use = CSV_PATH if os.path.exists(CSV_PATH) else "train.csv"
    df_meta = pd.read_csv(csv_to_use)
    df_meta = df_meta[df_meta["content"].notna() & (df_meta["content"].str.strip() != "")]
    df_meta = df_meta.reset_index(drop=True)
    print(f"  Metadata rows ({csv_to_use}): {len(df_meta)}")
    if len(df_meta) != len(data):
        raise RuntimeError(
            f"JSONL ({len(data)}) and CSV ({len(df_meta)}) row counts disagree — "
            f"rerun prep_llm_data.py to regenerate {DATA_PATH}."
        )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Apply Qwen ChatML template — SFTTrainer will tokenise the 'text' field
    texts = [
        tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False
        )
        for item in data
    ]
    print(f"  Formatted records: {len(texts)}")

    # Regime-mirroring split: held-out brands + held-out latest time window
    train_pos, eval_pos, split_info = make_regime_split(
        df_meta, EVAL_BRAND_FRAC, EVAL_TIME_FRAC, SEED
    )
    train_dataset = Dataset.from_dict({"text": [texts[i] for i in train_pos]})
    eval_dataset  = Dataset.from_dict({"text": [texts[i] for i in eval_pos]})
    print(f"  Train: {len(train_dataset)}  |  Eval: {len(eval_dataset)}")
    print(f"  Held-out brands ({split_info['n_held_brands']}, "
          f"using col '{split_info['brand_col']}'): "
          f"{split_info['held_brands']}{'...' if split_info['n_held_brands'] > 5 else ''}")
    print(f"  Held-out latest-time rows: {split_info['n_held_time_rows']}")

    # ------------------------------------------------------------------
    # 2. Load model in 4-bit QLoRA
    # ------------------------------------------------------------------
    print(f"\nLoading {MODEL_NAME} in 4-bit...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    print("  Model loaded.")

    # ------------------------------------------------------------------
    # 3. LoRA config — target all 4 attention projections at r=16
    # ------------------------------------------------------------------
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # ------------------------------------------------------------------
    # 4. Training config (SFTConfig handles max_seq_length natively)
    # ------------------------------------------------------------------
    # Compute warmup_steps from effective dataset size (replaces deprecated warmup_ratio)
    total_optim_steps = max(1, (len(train_dataset) // 16) * 3)   # 16 = grad_accum, 3 = epochs
    warmup_steps = max(10, int(total_optim_steps * 0.03))

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=0.3,
        fp16=False,
        gradient_checkpointing=True,
        eval_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        average_tokens_across_devices=False,
    )
    print(f"  Warmup: {warmup_steps} steps  |  Total optim steps: ~{total_optim_steps}")

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()

    print("\nInitialising trainer...")
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,
        callbacks=[VRAMGuardCallback()],
    )
    print("Starting fine-tune...")
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Save adapter + tokenizer
    # ------------------------------------------------------------------
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nAdapter saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
