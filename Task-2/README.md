# Task 2 — Content Simulation
## Adobe Behaviour Simulation Challenge (Inter IIT Tech Meet)

This repository contains the implementation for **Task 2: Content Simulation** of the Adobe Behaviour Simulation Challenge. The objective is to generate the text content of a marketing tweet given only its metadata (company, posting time, attached media URL, expected like-count target). The evaluation regime simulates a real marketing scenario: the model is tested on **brands it has never seen during training** and on **time periods after the training cut-off**.

---

## 1. Problem Statement in One Paragraph

> Given a row of structured metadata about a tweet — `{company, username, date, media_url, target_likes}` — generate the tweet text that company would plausibly have posted. The model is scored against the actual tweet using **BLEU 1–4**, **ROUGE-1/2/L**, and **CIDEr** on two held-out test sets: `test_unseen_brands.csv` (companies absent from training) and `test_unseen_time.csv` (the latest time window). Grading weight: Metrics 50%, Approach (efficiency + novelty) 35%, Presentation 15%.

The challenge is genuinely hard for three reasons:
1. **Cold-start generalization** — the model must extrapolate brand voice for companies whose tweets it has never read.
2. **Multimodal context** — most marketing tweets reference an image. Without "seeing" that image, the model is guessing.
3. **Hardware constraint** — this entire pipeline runs on a laptop with an **RTX 3050 Laptop GPU (4 GB VRAM)**. Most large-model defaults assume 24 GB+ data-centre cards.

---

## 2. Pipeline Overview

```
                        train.csv                     test_unseen_brands.csv
                            |                         test_unseen_time.csv
                            v                                 |
                  +-----------------+                         v
                  | enrich_vlm.py   |              +------------------+
                  | Qwen2.5-VL-3B   |              | eval.py          |
                  | 4-bit, single-  |              | 1) maybe_enrich  |
                  | sentence prompt |              |    _test (VLM)   |
                  +--------+--------+              | 2) load_llm      |
                           |                       | 3) beam-search   |
                           v                       |    generate      |
                  train_enriched.csv               | 4) BLEU/ROUGE/   |
                           |                       |    CIDEr         |
                           v                       +------+-----------+
                  +-----------------+                     |
                  | prep_llm_data.py|                     v
                  | chat-template + |              predictions_*.csv
                  | likes bucketing |              submission_*.csv
                  | (prompt_utils)  |
                  +--------+--------+
                           |
                           v
                  llm_train_data.jsonl
                           |
                           v
                  +-----------------+
                  | finetune_qwen.py|
                  | Qwen2.5-1.5B    |
                  | QLoRA (r=16)    |
                  | regime-mirror   |
                  | train/val split |
                  +--------+--------+
                           |
                           v
                  qwen15b_tweet_final/
                  (LoRA adapter)
```

The pipeline has **four sequential stages**. Each is a standalone Python script that reads from disk and writes to disk, so any stage can be re-run independently after a code change.

---

## 3. Component-by-Component Walkthrough

### Stage 1 — `enrich_vlm.py`
**Purpose:** Convert media URLs into one-sentence textual descriptions.

A Vision-Language Model (Qwen2.5-VL-3B-Instruct) downloads each image, then produces a single sentence of the form: *"Brand text 'XYZ Coffee'. A barista pours latte art into a white cup on a wooden counter."* This caption becomes a feature the LLM can condition on.

Key implementation details:
- Loaded in **4-bit NF4 quantization** (bitsandbytes) so a 3 B parameter model fits in ~2 GB of VRAM.
- The prompt explicitly demands a single sentence with OCR-first, no markdown, no newlines. This is the fix for BUG-12 described in §5.
- Resume is **keyed on tweet `id`**, not row count, so the script survives interruption and divergence between input and output row counts.
- Never overwrites the input CSV (separate input/output paths) — protects raw data from being clobbered by a crash mid-write.

### Stage 2 — `prep_llm_data.py` + `prompt_utils.py`
**Purpose:** Convert enriched rows into the JSONL format expected by `SFTTrainer`.

`prompt_utils.py` is the **single source of truth for the prompt template**. Both training (`prep_llm_data.py`) and inference (`eval.py`) import the same `build_messages()` function. This guarantees the prompt at train time and inference time are byte-identical — a frequent source of silent quality regressions in instruction-tuned LLMs.

The user message contains:
- Company + username (for brand identification)
- Day-of-week + hour (extracted from `date`)
- Cleaned VLM description (if available)
- Likes bucket (`high` / `moderate` / blank) — Task 2 input that the original baseline ignored
- Stylistic instruction to include hashtags

The full prompt is wrapped in Qwen's **ChatML** template via `tokenizer.apply_chat_template`, with `add_generation_prompt=False` at training and `True` at inference.

### Stage 3 — `finetune_qwen.py`
**Purpose:** Fine-tune Qwen2.5-1.5B-Instruct on the tweet generation task using QLoRA.

- **Base model:** Qwen2.5-1.5B-Instruct (chosen for the best quality-per-VRAM trade-off on a 4 GB card — see §4).
- **Adapter:** LoRA at rank 16, alpha 32, targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`. Only ~7 M parameters are trainable; the 1.5 B base model is frozen and held in 4-bit.
- **Optimizer:** `paged_adamw_8bit` — keeps the optimizer's momentum and variance state in CPU pinned memory and pages it onto the GPU only when needed. Without this, the optimizer state alone would exceed our VRAM budget.
- **Gradient checkpointing:** ON. Trades recomputation during backward for ~30% less activation memory.
- **Sequence length:** 256 tokens (tweets are short; this is 2x faster than the original 512-token setup).
- **Effective batch size:** 1 x 16 (per-device x gradient accumulation) = 16.
- **Schedule:** 3 epochs, cosine decay, 3% warmup, learning rate 2e-4.
- **Train/val split:** **regime-mirroring** — eval set is the union of (all rows from 5% of held-out brands) and (latest 5% of rows from remaining brands by date). This makes eval loss correlate with the leaderboard objective rather than random in-distribution loss.
- **VRAM Guard callback:** custom `TrainerCallback` that monitors GPU usage via `torch.cuda.memory_reserved()` after each optimizer step. If usage crosses 98%, it issues `torch.cuda.empty_cache()` between steps to free the CUDA allocator cache before the next forward pass.

### Stage 4 — `eval.py`
**Purpose:** Generate predictions for the two test sets, compute metrics, and write submission CSVs.

Execution order matters here for VRAM reasons:
1. Load both test CSVs.
2. **VLM enrichment first** — load Qwen2.5-VL-3B, run `maybe_enrich_test()` on rows missing descriptions, then `del` and `empty_cache()` to free 2+ GB.
3. **LLM loading second** — load Qwen2.5-1.5B + the trained LoRA adapter. This way both models never reside in VRAM simultaneously.
4. Generate via **beam search** (`num_beams=4`, `no_repeat_ngram_size=3`, `do_sample=False`). Sampling is the wrong choice here — BLEU/ROUGE/CIDEr reward overlap with a single reference, so the mode of the distribution is the better target.
5. Post-process: strip preambles like "Sure, here's a tweet:", remove surrounding quotes, truncate at first paragraph break.
6. Compute metrics with `nltk` (BLEU 1–4 with method-3 smoothing), `rouge-score` (ROUGE-1/2/L F-measure), and `pycocoevalcap` (CIDEr).
7. Build submission CSVs that preserve every original input column and only overwrite `content` with the generated tweet.

---

## 4. Model Choices and Their Justification

| Model | Role | Size | Why |
|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | VLM | 3 B (4-bit) | Best scene understanding + OCR on lifestyle marketing images; runs in ~2 GB. |
| **Qwen2.5-1.5B-Instruct** | LLM base | 1.5 B (4-bit) | Sweet spot of quality/VRAM. 0.5 B underfits; 3 B+ would OOM during QLoRA training on 4 GB. |
| **LoRA r=16, alpha=32** | Adapter | ~7 M params | Standard QLoRA recipe; r=16 hits diminishing returns above this for ~1 B models. |
| **paged_adamw_8bit** | Optimizer | — | Without paging, AdamW alone would exceed 4 GB. |

Models considered and rejected:
- **Florence-2-large:** great OCR, fast, but weaker at scene description for the lifestyle photography that dominates marketing tweets.
- **Mistral-7B / Llama-3-8B:** cannot QLoRA-train on 4 GB (would OOM even at short sequence lengths).
- **Flan-T5-base (250 M):** considered as an efficiency-narrative baseline; viable but would likely lag a 1.5 B decoder on a free-text generation task.

---

## 5. Issues Faced and How They Were Resolved

A representative selection — full root-cause analysis lives in `progress.md`.

### 5.1 Train/inference prompt mismatch (BUG-02)
The original `prep_llm_data.py` and `eval.py` each had their own hand-rolled prompt builder, and the wording drifted. A fine-tuned LLM is highly sensitive to prompt format. **Fix:** extracted `build_messages()` into `prompt_utils.py` and imported it from both scripts.

### 5.2 Wrong base model in `eval.py` (BUG-01)
The first version of `eval.py` tried to load Mistral-7B and apply a Qwen-trained LoRA adapter to it. This would silently fail because the layer names differ. **Fix:** rewrote `eval.py` to target Qwen2.5-1.5B and switched the prompt builder to ChatML.

### 5.3 Train CSV being overwritten (BUG-03)
`enrich_vlm.py` was writing checkpoint state back to `INPUT_CSV`, which slowly corrupted the raw data. After a crash mid-write, the resume guard (which compared row counts) silently no-op'd because the file lengths were now inconsistent. **Fix:** treat `INPUT_CSV` as read-only and key resume off the tweet `id` column via a merge.

### 5.4 99.8% VLM enrichment failure
Of 17,331 rows, only ~35 had a usable VLM description. Two reasons:
- Most Twitter media URLs from tweets several years old have expired (`pbs.twimg.com` returns 404).
- The original VLM prompt was verbose markdown, producing multi-line outputs that broke CSV parsing.

**Fixes:**
- Replaced the prompt with a strict single-sentence formulation (see `enrich_vlm.py:47`).
- Added defensive `re.sub(r"\s+", " ", desc).strip()` normalization in `prompt_utils.build_instruction()` so the LLM is never fed multi-line garbage even if older rows still contain it.
- Accepted that for dead URLs, the prompt degrades gracefully (the `Image:` line is simply omitted).

### 5.5 VRAM crisis during training
On a 4 GB card, training a 1.5 B model — even with 4-bit quantization, 8-bit optimizer, and gradient checkpointing — runs at **94% VRAM steady-state with only ~30 MB free**. Several workarounds were tried and discarded:
- `torch.cuda.empty_cache()` after every step **made things worse** — it forces PyTorch to cold-allocate fresh CUDA memory each step rather than reusing its cached pool. Throughput dropped from 10 s/it to 14 s/it.
- `gc.collect()` after every step added another 4–5 s/step of pure Python overhead.

**Final design:** the `VRAMGuardCallback` only fires `empty_cache()` when usage actually crosses **98%**, leaving PyTorch's allocator pool intact during normal operation.

### 5.6 TRL 0.29.0 API breakage
TRL 0.29.0 renamed `SFTConfig.max_seq_length` -> `max_length`. The first training attempt died with `TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`. **Fix:** inspected the signature with `inspect.signature(SFTConfig.__init__)` and updated to the new name. Also replaced the now-deprecated `warmup_ratio` with `warmup_steps`.

### 5.7 UnicodeEncodeError on Windows
A Unicode arrow character in a `print()` statement inside the VRAM callback crashed training mid-step because Windows `cp1252` console encoding cannot represent it. **Fix:** replaced with ASCII `->`. (The same issue had previously bitten `prep_llm_data.py`; resolved with `ensure_ascii=True` there.)

### 5.8 Florence-2 contamination in eval pipeline
A stale code path in `eval.py` was loading Florence-2-large for test-set enrichment while training used Qwen2.5-VL. This would create a train/test distribution shift on the visual feature. **Fix:** rewrote `maybe_enrich_test()` to use the same Qwen2.5-VL pipeline as `enrich_vlm.py`, and reordered the script so VLM loads-and-frees *before* the LLM is loaded (otherwise both 4-bit models would attempt to co-reside in 4 GB).

---

## 6. Reproduction

```bash
# Dependencies — transformers >= 5.0, trl >= 0.29, peft, bitsandbytes, qwen-vl-utils,
#                nltk, rouge-score, pycocoevalcap
pip install -r requirements.txt

# 1. (Optional, slow) Enrich training images with VLM descriptions
python enrich_vlm.py

# 2. Build the JSONL the trainer expects
python prep_llm_data.py

# 3. Train (about 8 hours on RTX 3050 Laptop, 4 GB)
python finetune_qwen.py

# 4. Generate predictions, compute metrics, write submissions
python eval.py
```

Outputs:
- `predictions_unseen_brands.csv` / `predictions_unseen_time.csv` — generations + actual tweet for inspection.
- `submission_unseen_brands.csv` / `submission_unseen_time.csv` — competition-format submissions.

---
