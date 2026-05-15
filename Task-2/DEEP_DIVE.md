# DEEP_DIVE — Task 2 Implementation Reasoning

A personal study companion to the `README.md`. Where the README explains **what we built**, this document explains **why every decision was made**, so the author can defend each choice in front of an ML expert.

Each section is structured as:
- **Decision** — what was actually done.
- **Why this** — the reasoning chain.
- **What else we considered** — the alternatives and why they were rejected.
- **Expert-level pushback** — the most likely question an interviewer will ask, with a defensible answer.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Two-Stage VLM → LLM Architecture](#2-two-stage-vlm--llm-architecture)
3. [Why Qwen2.5-VL-3B for Stage 1](#3-why-qwen25-vl-3b-for-stage-1)
4. [Why Qwen2.5-1.5B-Instruct for Stage 2](#4-why-qwen25-15b-instruct-for-stage-2)
5. [QLoRA — All Three Letters Explained](#5-qlora--all-three-letters-explained)
6. [LoRA Hyperparameters — r=16, alpha=32, attention-only](#6-lora-hyperparameters--r16-alpha32-attention-only)
7. [Optimizer — `paged_adamw_8bit`](#7-optimizer--paged_adamw_8bit)
8. [Sequence Length — Why 256](#8-sequence-length--why-256)
9. [Gradient Checkpointing — Trade-off Analysis](#9-gradient-checkpointing--trade-off-analysis)
10. [Effective Batch Size — Why 1 x 16 = 16](#10-effective-batch-size--why-1-x-16--16)
11. [Learning Rate & Schedule](#11-learning-rate--schedule)
12. [Regime-Mirroring Train/Val Split](#12-regime-mirroring-trainval-split)
13. [Prompt Format and Likes Bucketing](#13-prompt-format-and-likes-bucketing)
14. [VLM Prompt Engineering](#14-vlm-prompt-engineering)
15. [Decoding — Beam Search, Not Sampling](#15-decoding--beam-search-not-sampling)
16. [Metrics — Why BLEU + ROUGE + CIDEr](#16-metrics--why-bleu--rouge--cider)
17. [The VRAMGuardCallback — Engineering for the 4 GB Limit](#17-the-vramguardcallback--engineering-for-the-4-gb-limit)
18. [Things We Did NOT Do — and Why That's Defensible](#18-things-we-did-not-do--and-why-thats-defensible)
19. [Likely Interview Questions](#19-likely-interview-questions)

---

## 1. Problem Framing

**Decision:** Treat this as a conditional sequence generation problem: `p(tweet_text | metadata, image_caption)`.

**Why this:**
- The grading metrics (BLEU, ROUGE, CIDEr) reward token overlap with a single reference. That's the same scoring family used for image captioning and machine translation, where conditional language modelling is the standard approach.
- The input is a fixed schema of structured fields, not a free-form natural language query. Wrapping the schema into a textual prompt and asking an instruction-tuned LLM to complete it is straightforward.

**What else we considered:**
- **Pure retrieval (k-NN over training tweets):** would give a strong baseline on BLEU but cannot generalize to *unseen brands*. The test set explicitly tests that.
- **Encoder-decoder (e.g. Flan-T5):** good fit, but on 4 GB VRAM a 3 B encoder-decoder won't QLoRA-train. We pick a decoder-only model in the same parameter ballpark and get more capability per parameter.
- **Reinforcement learning from a metric reward (PPO with BLEU as reward):** academically interesting, but BLEU/ROUGE are non-differentiable, sparse rewards prone to reward hacking, and we don't have the budget for the extra training pass.

**Expert pushback — "Why not multi-task with Task 1's like-count prediction?"**
Because the prompt already conditions on a *target* like-count (we provide it as input). Predicting likes would require a regression head, doubling training complexity for a problem that's already over-constrained on VRAM. We instead use the provided target to bucket the prompt.

---

## 2. Two-Stage VLM → LLM Architecture

**Decision:** Run a VLM offline to convert images to single-sentence captions, then condition a text-only LLM on `metadata + caption`. The two models never run in the same forward pass.

**Why this:**
- **VRAM realism.** A native multimodal model (e.g. fine-tuning Qwen2.5-VL itself on tweet generation) would need both the vision tower and the LLM in memory simultaneously. On 4 GB that's a non-starter.
- **Caching.** Image captions are generated once and saved to `train_enriched.csv`. The LLM iterates over the dataset 3 epochs; the VLM only runs once. Total compute drops dramatically.
- **Failure isolation.** The dataset has dead media URLs (most of Twitter's `pbs.twimg.com` from 2021 returns 404). A two-stage pipeline lets us treat caption absence as just another input value (the prompt degrades gracefully) rather than corrupting an end-to-end model.

**What else we considered:**
- **End-to-end multimodal fine-tuning of Qwen2.5-VL-3B with LoRA.** Two problems: (a) requires loading vision tower + language model + adapters + activations in the same VRAM budget, which doesn't fit; (b) the trainable parameter count is larger and convergence on 17 K rows is iffy.
- **Visual features from a separate encoder (e.g. CLIP image embeddings) prepended to the prompt.** Would need to teach the LLM to ground continuous embeddings — that's prefix-tuning territory and adds engineering complexity for unclear gain over natural-language captions.

**Expert pushback — "Doesn't the caption bottleneck throw away information?"**
Yes — pixels carry more information than 60 words. But on this dataset:
1. The reference tweet was written by a human looking at the same image and produced ~30 words. So the caption bandwidth is roughly matched to the output bandwidth.
2. Most marketing tweets reference *what's depicted* (a product, a slogan) rather than fine visual texture. OCR + scene description captures the relevant signal.
3. Empirically, the only end-to-end VLM baseline we could fit OOMs at batch size 1; this isn't an option, it's a hardware constraint.

---

## 3. Why Qwen2.5-VL-3B for Stage 1

**Decision:** `Qwen/Qwen2.5-VL-3B-Instruct` in 4-bit NF4.

**Why this:**
- **OCR quality.** Marketing images frequently contain text (brand name, product title, slogan). That text often reappears verbatim in the tweet, which means correct OCR translates directly into BLEU.
- **Scene description quality.** For lifestyle/aspiration photography (a barista, an athlete, a sunset over a logo), Qwen2.5-VL gives more useful scene captions than OCR-first models like Florence-2.
- **Fits in 4 GB at 4-bit.** ~2 GB for the model leaves room for inputs and intermediate tensors.

**What else we considered:**
- **Florence-2-large (0.77 B).** Excellent OCR, much faster, fits in fp16. But initial testing showed it under-describes scenes — a barista photo would return *"a person in a kitchen"* whereas Qwen2.5-VL returns *"a barista pouring milk into a cup of coffee with latte art on a wooden counter, branded 'Tim Hortons'"*. The richer description gives the LLM more to condition on.
- **Moondream2 (1.8 B).** Edge-optimized, ~2 GB in fp16. Reasonable middle ground but no clear win over Qwen2.5-VL for our use case, and we already had the Qwen ecosystem.
- **BLIP-2 OPT-2.7B.** Heavier, weaker OCR — strictly dominated.

**Expert pushback — "If OCR is so valuable, why not run a dedicated OCR model alongside the VLM?"**
We could, and that would probably help on BLEU. We chose not to because (a) it doubles VRAM budget for marginal gain since Qwen2.5-VL already does competent OCR via instructed prompting; (b) handling two outputs per image adds pipeline complexity for diminishing returns. If we had a 6 GB+ card, OCR + VLM would be the cleaner approach.

---

## 4. Why Qwen2.5-1.5B-Instruct for Stage 2

**Decision:** `Qwen/Qwen2.5-1.5B-Instruct` in 4-bit NF4 with LoRA.

**Why this:**
- The classic QLoRA paper showed that with proper rank-decomposition adapters, a 4-bit base model fine-tunes to within a few % of a full-precision baseline.
- On a 4 GB card, the model weights themselves use ~1.2 GB at 4-bit, leaving ~2.8 GB for activations + gradients + optimizer state pages + CUDA overhead.
- Qwen2.5-1.5B-Instruct is *already* tuned for instruction following and ChatML. We're not starting from a base model that needs to learn the chat format — we only need to teach it the *style* of marketing tweets.

**What else we considered (all rejected):**
- **Qwen2.5-0.5B-Instruct.** Tried first. The model underfit — generations were generic and metric scores were poor. The capacity-to-task-difficulty ratio was wrong.
- **Qwen2.5-3B-Instruct.** Tried; OOMed during the backward pass at batch size 1 with sequence length 256. 3 B in 4-bit is ~1.6 GB, but the activation memory at backward time blows the budget. Doable on 6 GB+.
- **Llama-3.2-1B-Instruct.** A reasonable alternative. We picked Qwen because (a) its tokenizer is denser for tweet-style text, (b) we already had its VLM sibling, keeping the ecosystem consistent.
- **Mistral-7B-Instruct.** Impossible on 4 GB for training — even inference is borderline.

**Expert pushback — "Are you sure 1.5 B is enough for a generative marketing task?"**
For *generation quality* (fluency, grammar), 1.5 B is overkill. For *brand-specific style*, the bottleneck is training data and prompt design, not model capacity. We have 17 K tweets across ~hundreds of brands — that's ~100 examples per brand on average. A larger model would overfit faster on so little data. If we had the full 300 K corpus, scaling up the base would help; with 17 K it would not.

---

## 5. QLoRA — All Three Letters Explained

**Decision:** Use QLoRA = Q (4-bit Quantization) + LoRA (Low-Rank Adapters) + paged optimizer.

**Why this — the math:**
- Full-precision fp16 Qwen2.5-1.5B has ~1.5 B parameters × 2 bytes = **3 GB just for weights**, before activations or gradients. Doesn't fit on a 4 GB card with any usable batch.
- 4-bit NF4 quantization compresses each weight to ~0.5 bytes, giving **~0.75 GB for weights**. We now have ~3.25 GB for everything else.
- LoRA freezes the quantized base and only trains low-rank adapters: `W' = W + (A * B)` where A is `(d, r)` and B is `(r, d)`. With r=16 and d=1536 (Qwen2.5-1.5B hidden size), each adapted matrix has `16 * 1536 * 2 = ~50 K` trainable params. Across 4 attention matrices × 28 layers ≈ **5.6 M trainable params** versus 1.5 B frozen. Gradient memory is correspondingly tiny.

**Why each component is non-optional on 4 GB:**
- Drop quantization → fp16 weights alone don't fit. Dead.
- Drop LoRA → full fine-tuning means gradients for all 1.5 B params at fp16 = 3 GB of gradient buffer. Dead.
- Drop paged optimizer → AdamW state (m + v) per trainable param at fp32 = 8 bytes per param, so even for 5.6 M LoRA params that's only 45 MB — but during a temporary spike Adam might need to materialize state for more params, which gets dicey.

**What else we considered:**
- **DoRA (weight-decomposed LoRA).** Better quality per parameter but slower training, and the implementation in PEFT is newer/less battle-tested.
- **AdaLoRA.** Dynamically allocates rank across layers. Promising but adds another moving part to tune.
- **Full fine-tuning of a smaller model** (e.g. full-precision GPT-2 medium 355 M). Possible, but 355 M is too small for instruction-following quality.

**Expert pushback — "Why NF4 and not standard int4 or fp4?"**
NF4 (Normal Float 4) is a quantization scheme calibrated to the empirical distribution of neural network weights, which are roughly Gaussian after training. It places its 16 representable values along quantiles of N(0,1), so it preserves more of the weight distribution's variance than uniform int4 at the same bit-width. The QLoRA paper showed NF4 outperforms int4 on downstream task metrics.

---

## 6. LoRA Hyperparameters — r=16, alpha=32, attention-only

**Decision:** LoRA rank 16, scaling alpha 32 (so `lora_alpha / r = 2`), targeting only the four attention projections `q_proj, k_proj, v_proj, o_proj`.

**Why this:**
- **r=16** is the most common QLoRA-paper recipe for 1 B-scale models. r=8 gives ~half the parameters and slightly worse quality; r=32+ gives diminishing returns and risks overfitting on 17 K rows.
- **alpha = 2 × r** is a heuristic from the LoRA paper: the effective learning rate of the adapter scales as `alpha / r`. Setting them in a 2:1 ratio keeps the adapter's update magnitude roughly constant as you change rank.
- **Attention-only targets** is the classic LoRA recipe. The intuition: attention is where the model decides *what to attend to* (i.e. brand context, image caption, time), while MLPs handle the heavy nonlinear computation. Adapting attention is enough to steer behaviour for most fine-tuning tasks.

**What else we considered:**
- **Adapting MLPs too** (`gate_proj, up_proj, down_proj`). Best practice in 2024+ literature. Listed as a future optimization in `progress.md` §5 — would add ~3× more trainable params for ~30% additional VRAM and a quality bump.
- **Larger rank (r=32 or 64).** Risk of overfitting on 17 K examples without clear quality return. Worth trying if we get the full 300 K corpus.
- **`lora_dropout=0.05`.** Mild regularization; standard value. Could be raised to 0.1 if eval loss diverges from train loss.

**Expert pushback — "Have you confirmed that attention-only is enough?"**
For instruction-style adaptation tasks (style, format, voice), attention-only LoRA matches MLP-inclusive LoRA in most published ablations within ~1 BLEU point. Where MLP-LoRA dominates is in *capability acquisition* (e.g. teaching a model a new domain like medicine). We're teaching style on top of an already-capable instruction model, which is the regime where attention-only works.

---

## 7. Optimizer — `paged_adamw_8bit`

**Decision:** `optim="paged_adamw_8bit"` (bitsandbytes).

**Why this:**
- AdamW maintains two state tensors per trainable parameter: first moment (m) and second moment (v). At fp32 that's 8 bytes per param.
- 8-bit AdamW from bitsandbytes quantizes these state tensors with block-wise quantization, cutting memory ~4×.
- **Paged** adds a layer that stores the state tensors in *CPU pinned memory* and pages them into GPU only when needed for the actual parameter update. This means the optimizer state effectively lives in RAM, with GPU acting as a working set.
- For our 5.6 M LoRA params, the optimizer state is small in absolute terms, but during initialization and AdamW step, transient allocations spike. Paging makes the spikes safe.

**What else we considered:**
- **Plain `adamw_8bit`** (non-paged). Faster (no CPU↔GPU transfer overhead), and would probably work given our small adapter footprint. Switching to non-paged is a potential speed optimization on a future run.
- **SGD with momentum.** Fewer state tensors but historically worse for LLM fine-tuning. Not worth the risk.
- **Adafactor.** Adam-family memory-efficient. Older, sometimes unstable for LM finetuning at low precision.

**Expert pushback — "Are you sure paging doesn't add latency?"**
On a 4 GB card with a small adapter, the paging cost is negligible — bitsandbytes uses CUDA streams and overlaps transfers with compute. The only time we'd see overhead is if the *entire* state needed to be paged in for every step, but with a 5.6 M adapter the working set fits comfortably on GPU. We chose paged_adamw_8bit primarily as a safety margin against transient OOMs.

---

## 8. Sequence Length — Why 256

**Decision:** `MAX_SEQ_LEN = 256` tokens, set on `SFTConfig.max_length`.

**Why this:**
- Tweets are capped at 280 *characters*, which is ~50-70 BPE tokens. Adding the system message + user message (company, time, caption, instruction) brings the typical sample to ~150-220 tokens.
- 256 leaves headroom for the longest 99th-percentile sample without burning budget on padding.
- Sequence length determines activation memory and FLOPs. Going from 512 to 256 roughly **halves both** at no quality cost (we're not truncating actual content).

**What else we considered:**
- **512 (original).** Worked but wasted 2× compute and memory per step. Replaced when we noticed >95% of samples were under 256 tokens.
- **128.** Would truncate the longest VLM descriptions and the multi-line samples. Risky.
- **Dynamic padding to max in batch.** Helps with mixed lengths but at batch size 1 it's effectively no different from a fixed cap.

**Expert pushback — "Did you actually measure the token-length distribution?"**
Not formally with a histogram; we inspected the prompt builder output and the median chat-templated sample. A token-count histogram would let us defend 256 numerically — that's a hygiene step we should add to the next iteration. The current choice is conservative: 256 is comfortably above the qualitative max we saw.

---

## 9. Gradient Checkpointing — Trade-off Analysis

**Decision:** `gradient_checkpointing=True`.

**Why this:**
- During backward pass, PyTorch needs intermediate activations to compute gradients. Without checkpointing, every layer's output is stored from the forward pass — memory grows linearly in sequence length and layer count.
- Gradient checkpointing stores only a subset of activations (e.g. one per transformer block) and *recomputes* the rest during the backward pass. Memory drops ~30-50%, compute increases ~30%.
- On a 4 GB card with 1.5 B params at sequence length 256, the activation memory at backward is the biggest spike. Without checkpointing we'd OOM.

**What else we considered:**
- **Off.** Tried in initial experiments — OOM at the backward pass.
- **Selective layers.** PyTorch supports finer-grained control (`gradient_checkpointing_kwargs`), but the default is fine for our scale.

**Expert pushback — "The 30% compute overhead is significant; have you tried turning it off with the shorter sequence length?"**
Worth trying. At seq 256 the activation memory roughly halved, so checkpointing-off might now fit. The reason we kept it on is conservatism: VRAM was already at 94% steady-state. Turning off checkpointing might recover some speed but risks OOM at the eval pass (where activation memory spikes if eval batch >1 or if some samples are unusually long). A future ablation worth running.

---

## 10. Effective Batch Size — Why 1 × 16 = 16

**Decision:** `per_device_train_batch_size=1, gradient_accumulation_steps=16` → effective batch = 16.

**Why this:**
- We are VRAM-bound. At batch size 1, sequence 256, with all the other memory-saving tricks, we're already at 94% VRAM. We cannot fit batch 2.
- Gradient accumulation simulates a larger effective batch *for the optimizer update direction* without increasing memory: gradients are averaged across 16 forward/backward passes before `optimizer.step()` is called.
- Effective batch 16 is on the small side for LLM finetuning. The literature suggests 32-128 for stable training, but smaller works fine with cosine LR + warmup.

**What else we considered:**
- **Higher gradient_accumulation (e.g. 32).** Would give effective batch 32 at the cost of fewer optimizer updates per epoch. With 17 K rows / 16 = ~1029 steps per epoch, going to 32 would halve to ~515. We chose 16 for more frequent updates.
- **Lower gradient_accumulation (e.g. 4 or 8).** More updates per epoch but noisier gradient estimates. Risk of unstable loss especially at the start of training.

**Expert pushback — "An effective batch of 16 is small; doesn't this hurt convergence?"**
For LoRA fine-tuning, no — LoRA adapters have such a small parameter count (~5.6 M) that gradient noise is already low. Full-finetuning on 1.5 B params would benefit from larger batches because the gradient estimate is over many more dimensions, but a 5.6 M adapter converges fine on batch 16.

---

## 11. Learning Rate & Schedule

**Decision:** `learning_rate=2e-4`, cosine decay, 3% warmup.

**Why this:**
- **2e-4 is the canonical QLoRA learning rate** for 1-3 B models. It's an order of magnitude higher than full-finetuning rates (typically 1e-5 to 5e-5) because LoRA's effective parameter count is much smaller, requiring larger steps.
- **Cosine decay** gives a smooth deceleration to near-zero by the end of training. Compared to linear decay, cosine spends more time at high learning rates early and decays faster near the end, which tends to improve final loss empirically.
- **3% warmup** (= ~93 steps out of 3087) ramps up from 0 to the target LR to avoid early gradient explosions when the optimizer state is uninitialized.

**What else we considered:**
- **1e-4.** Safer, slower convergence. Reasonable fallback if we saw loss spikes.
- **5e-4.** Tried in informal experiments; loss diverged for a 0.5 B model. Probably too aggressive.
- **Linear decay.** Cosine usually wins in this regime — chose by convention.

**Expert pushback — "Have you tuned the LR? 2e-4 is just a default."**
Honest answer: no, we did not run a sweep — we used the QLoRA-paper recipe. Given the VRAM constraint, a proper sweep would mean 5+ training runs of 8 hours each. We can defend the choice on prior literature but cannot claim it's optimal for this exact dataset. With more compute budget, a one-cycle scheduler with LR-finder would be the next refinement.

---

## 12. Regime-Mirroring Train/Val Split

**Decision:** Eval set is the union of (rows from 5% of held-out brands) and (latest 5% of remaining rows by date).

**Why this:**
- The competition has two test regimes: **unseen brands** and **unseen time**. A random in-distribution split would give an eval loss that overestimates how well the model generalizes to *both* held-out conditions.
- By mirroring both regimes in our eval set, the `eval_loss` metric correlates better with leaderboard score. `load_best_model_at_end=True` then actually picks the checkpoint that's best on the relevant axis.

**Why we didn't make it 50/50 between regimes:**
- We don't know how the competition weights the two test sets. 5% + 5% is a balanced compromise that doesn't over-concentrate eval signal on either regime.

**What else we considered:**
- **Pure random split (the baseline).** Easy, but eval loss decouples from leaderboard performance, especially for the unseen-brand regime where the model is essentially doing zero-shot generation.
- **Brand-only hold-out.** Misses the time-shift signal.
- **Time-only hold-out.** Misses the brand-shift signal.

**Expert pushback — "Won't this make your training set smaller in a non-iid way?"**
Yes — we lose ~10% of the data. For 17 K rows that's ~1.7 K. The trade-off is that the remaining 90% is iid with respect to the *training distribution* (it sees all the same brands across all the same time periods minus the latest), while the eval set is OOD with respect to that. This is exactly what we want for honest validation. On a small dataset the data loss stings; on a 300 K-row dataset it would be free.

---

## 13. Prompt Format and Likes Bucketing

**Decision:** ChatML format via `tokenizer.apply_chat_template`, with the user message including a likes bucket.

**Why this:**
- **ChatML** is what Qwen2.5-Instruct was post-trained on. Using a different format would put the input out-of-distribution and waste the instruction-following capability.
- **Single source of truth** (`prompt_utils.py`) means training and inference can't diverge. This is the BUG-02 fix; LLMs are notoriously prompt-format-sensitive.
- **Likes bucketing** turns a numerical feature (`likes=3500`) into a discrete signal the LLM can condition on (`"Expected engagement: high (3,500 likes target)"`). Discrete tokens are easier for an LLM to learn to react to than raw numbers.

**Bucket thresholds (>=1000 high, >=100 moderate):** chosen by inspection of the distribution. Not calibrated to actual percentiles — a refinement we noted in the supervision report.

**Expert pushback — "Why not just use the raw like-count as a token?"**
LLMs are bad at numerical reasoning, especially with large vocabulary digits. Bucketing into ~3 levels (`high`, `moderate`, blank) reduces a continuous variable to a categorical one the model can pattern-match. It's the same principle as ordinal encoding for tree-based models.

---

## 14. VLM Prompt Engineering

**Decision (BUG-12 fix):**

```
In one sentence, describe this image for a marketing tweet:
first state any visible text or brand names, then describe the scene.
No bullet points, no markdown, no line breaks.
```

**Why this:**
- **"In one sentence"** prevents the multi-paragraph markdown output that broke CSV parsing in earlier runs.
- **"first state visible text or brand names"** forces OCR to come first. This matters because the brand name often appears in the tweet text verbatim — direct BLEU win.
- **"then describe the scene"** keeps the model from being purely OCR (some images have no text but rich scenes).
- **"No bullet points, no markdown, no line breaks"** belt-and-braces against the model's default markdown style.

**Defensive post-processing:** `re.sub(r"\s+", " ", desc).strip()` in `prompt_utils.build_instruction()` normalizes any old-format descriptions so the LLM never sees newlines/tabs in the caption.

**Expert pushback — "Why not ask the VLM directly to produce the tweet?"**
Tested informally; quality was poor. Qwen2.5-VL-3B is trained for visual question-answering and captioning, not for emulating brand voice. The fine-tuned LLM is the brand-voice specialist; the VLM is the visual translator. Keeping the roles separated lets us tune each stage independently.

---

## 15. Decoding — Beam Search, Not Sampling

**Decision:** `num_beams=4, no_repeat_ngram_size=3, do_sample=False`.

**Why this — fundamentally about the loss surface:**
- BLEU/ROUGE/CIDEr reward token overlap with a *single* reference. The optimal output is therefore the **mode** of `p(tweet | input)`, not a random draw.
- Sampling (with `do_sample=True`) draws from the distribution. Each call returns a different output, and the metrics penalize the variance away from the reference.
- Beam search approximates the mode by exploring `num_beams` parallel hypotheses and selecting the one with highest joint probability.

**Why these specific parameters:**
- **num_beams=4** is the standard for text generation. Going higher (8, 16) gives diminishing returns and explodes inference time.
- **no_repeat_ngram_size=3** prevents the most common pathology of beam search: repeating phrases ("...your story your story your story"). This is a brute force trick but works.
- **No length penalty** (default 1.0). Could be tuned to prefer shorter or longer outputs.

**What else we considered:**
- **Nucleus sampling (top-p=0.9).** Standard for chatbot generation but wrong for our metric-bound task.
- **Greedy (num_beams=1).** Slightly worse than beam, but 4× faster. Acceptable fallback.
- **Contrastive search.** Better diversity, but our task wants the mode, not diversity.

**Expert pushback — "Beam search is known to produce bland, repetitive text. Is that what you want?"**
For BLEU/ROUGE/CIDEr, yes — "bland" means "high-probability under the conditional model", which means "close to what a human would write". The papers that complain about beam blandness are evaluating with human ratings or distinct-n diversity metrics, where this is a real problem. For surface-form-overlap metrics, beam wins.

---

## 16. Metrics — Why BLEU + ROUGE + CIDEr

**Decision:** Compute BLEU-1/2/3/4, ROUGE-1/2/L, and CIDEr; the competition uses these.

**What each measures:**
- **BLEU-N:** precision-weighted n-gram overlap, with brevity penalty. Sensitive to phrasing.
- **ROUGE-N:** recall-weighted n-gram overlap. Complements BLEU.
- **ROUGE-L:** longest common subsequence — captures word-order similarity without requiring contiguous overlap.
- **CIDEr:** TF-IDF-weighted n-gram cosine similarity. Down-weights common n-grams (like "the", "a"), up-weights brand-specific terms. Originally for image captioning; well-suited here.

**Implementation choices:**
- **NLTK with SmoothingFunction.method3** for BLEU — handles low-count n-grams gracefully (corpus-level smoothing).
- **rouge-score** library (not the older `rouge` package) — official Google reimplementation, more reliable.
- **pycocoevalcap.cider** — the COCO captioning eval-server CIDEr implementation. Standard reference.

**Expert pushback — "These metrics are known to correlate poorly with human judgment of generation quality. Why not use BERTScore or BLEURT?"**
Because we don't choose the metric — the competition does. BLEU/ROUGE/CIDEr are what's scored. If the goal were aesthetic quality we'd add BERTScore as an internal sanity check, but that wouldn't change the optimization target.

---

## 17. The VRAMGuardCallback — Engineering for the 4 GB Limit

**Decision:** A `TrainerCallback` that monitors `torch.cuda.memory_reserved()` after every optimizer step. Triggers `torch.cuda.empty_cache()` only when reserved memory ≥ 98% of total.

**Why "only at 98%":**
- Naive cache-clearing every step **hurts performance**: PyTorch's allocator caches dead tensors for fast reuse. Clearing the cache forces the next allocation to go to the CUDA driver, which is slower. We measured this: every-step cache clearing dropped throughput from 10 s/it to 14 s/it.
- The cache is only a problem when it competes with live tensors. At 98% reserved, we're about to OOM regardless, so paying the cold-allocation cost is the lesser evil.

**Why `memory_reserved()` and not `nvidia-smi`:**
- `nvidia-smi` is a subprocess. On Windows, spawning subprocesses adds 200-500 ms of overhead. Calling it per step destroyed throughput.
- `torch.cuda.memory_reserved()` is a Python API call to a CUDA query, sub-millisecond.

**Why "after step_end" and not "before forward":**
- `on_step_end` fires once per **optimizer step** (after gradient accumulation completes and weights actually update). This is exactly when the user said: "free memory once weights are modified."
- Firing on `on_substep_end` (per micro-batch) would call the callback 16× more often with no real benefit.

**Expert pushback — "If you wanted to truly minimize VRAM, why not move to 8-bit weights instead of 4-bit, which is more lossy?"**
4-bit NF4 has been empirically shown to give within ~1% of 8-bit performance on a wide range of downstream tasks (QLoRA paper, table 2). The memory savings are 2× (8-bit → 4-bit halves weight memory), which on a 4 GB card is the difference between fitting and not fitting. The quality cost is below the noise floor of our task.

---

## 18. Things We Did NOT Do — and Why That's Defensible

### Retrieval-augmented generation (RAG)
**Why not yet:** the time budget for the competition forced us to ship the base pipeline first. RAG is the single highest-leverage upgrade — flagged in `progress.md` §6.1 as the next priority.

### Full 300 K training set
**Why not yet:** only ~17 K rows came packaged with the starter kit. Sourcing the rest is a data-engineering task we deferred.

### MLP-targeted LoRA
**Why not yet:** flagged in the supervision report (§ "what can be optimized"). Adds ~30% memory; worth a retrain on the next pass.

### Sweep over LR / rank / target modules
**Why not yet:** at 8 hours per training run on a 4 GB card, even a tiny grid sweep would consume days. We picked literature defaults.

### DPO / RLHF
**Why not at all (for this competition):** these require a reward model or preference data. We have neither.

**Expert pushback — "None of these are real reasons; they're just 'I didn't have time'."**
True, and that's a legitimate engineering answer. Real ML projects are budget-constrained. The defensible part is that we *identified* these gaps explicitly (see the supervision report) and prioritized base pipeline correctness over speculative gains, knowing that a working baseline is the prerequisite for any further iteration.

---

## 19. Likely Interview Questions

Quick answers to canonical questions:

> **Why is your loss starting at 3.2 and what's a reasonable endpoint?**
> Cross-entropy loss on a 150K-vocab token distribution where the model has to predict ~50-100 specific tokens given the prompt. Starting at 3.2 means the model assigns ~exp(-3.2) ≈ 4% probability to the right token on average, which is fine for a base instruction model that hasn't seen marketing tweets yet. A well-fit model on this task should reach ~1.5-2.0.

> **How would you know if you're overfitting?**
> Train loss continues to drop while eval loss (computed every 500 steps on the regime-mirroring split) plateaus or rises. `load_best_model_at_end=True` saves the best-eval-loss checkpoint, so even if late epochs overfit, we still ship the best.

> **What's the expected variance of your reported BLEU?**
> Beam search is deterministic, so variance from the model is zero given a fixed checkpoint. Variance across runs comes from (a) random LoRA initialization with `seed=42` controlling it, (b) data shuffling — also seeded. The biggest source of unexplained variance would be CUDA non-determinism in kernels, which is small for our scale.

> **Why not use Reinforcement Learning to directly optimize BLEU?**
> BLEU is non-differentiable. RL methods like REINFORCE can backprop through a sample, but they suffer from high variance and reward hacking (the model finds n-gram patterns that game BLEU without producing sensible tweets). For this task, supervised cross-entropy is more reliable and within the time budget.

> **What's the risk of your model just memorizing brand patterns?**
> For *seen brands* (the time-shift test), some memorization is *desired* — that's how brand voice transfers. For *unseen brands*, the model relies on the instruction-tuned prior + image caption + likes signal. The regime-mirroring eval split lets us watch that generalization directly.

> **Why ChatML over a custom prompt template?**
> Qwen2.5-Instruct was post-trained on ChatML. Any other format is out-of-distribution, requiring the LLM to relearn the format from scratch using its very limited LoRA capacity. We want LoRA to learn marketing style, not format.

> **What's the bottleneck for further quality improvement?**
> In order: (1) data scale (17 K → 300 K), (2) retrieval-augmented prompting, (3) VLM coverage (currently 0.2% of rows have descriptions due to dead URLs). All three are non-modelling improvements.

---

## Closing Self-Critique

The pipeline is correct, defensible, and runs on consumer hardware. The biggest weaknesses are:

1. **No hyperparameter tuning.** Defended by compute budget, but a 5-run mini-sweep would be honest.
2. **Tiny VLM coverage.** Most of the "multimodal" claim is currently aspirational because Twitter URLs are dead.
3. **No retrieval baseline.** The highest-leverage improvement that wasn't implemented.

If asked "what would you do differently with twice the compute?", the answer is: bigger LR sweep, retrieval module, MLP-LoRA, full 300 K training set. None of those required different models — the choice of Qwen2.5-1.5B + QLoRA was right for the constraint.
