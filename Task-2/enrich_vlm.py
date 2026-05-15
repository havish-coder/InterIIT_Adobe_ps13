"""
Qwen2.5-VL-3B-Instruct Tweet Media Enrichment
===============================================
Reverted from Florence-2: Qwen2.5-VL-3B gives better scene understanding
for lifestyle/photography images (the majority of marketing tweets).
The original failure was BUG-12 — a verbose markdown prompt. Fixed here.

Output format: single line, max 60 words, no markdown, no newlines.
  "Brand name / visible text. One-sentence scene description."

Resume: keyed on 'id' column — safe when files differ in row count.
Safety: INPUT_CSV is never written to.
"""

import os
import re
import gc
import sys
import tempfile
import torch
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_CSV      = "train_enriched.csv"
OUTPUT_CSV     = "train_enriched.csv"
MODEL_ID       = "Qwen/Qwen2.5-VL-3B-Instruct"
SAVE_INTERVAL  = 10
MAX_NEW_TOKENS = 80
TIMEOUT        = 10
MAX_DESC_WORDS = 60

TMP_IMAGE_PATH = os.path.join(tempfile.gettempdir(), "qwen_current_img.jpg")

# Fixed prompt — concise, single-line, OCR-first (BUG-12 fix)
VLM_PROMPT = (
    "In one sentence, describe this image for a marketing tweet: "
    "first state any visible text or brand names, then describe the scene. "
    "No bullet points, no markdown, no line breaks."
)


def log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------
def extract_image_url(media_string: str):
    if not isinstance(media_string, str):
        return None
    m = re.search(r"fullUrl='(https?://[^']+)'", media_string)
    if m:
        return m.group(1)
    m = re.search(r"thumbnailUrl='(https?://[^']+)'", media_string)
    if m:
        return m.group(1)
    if media_string.startswith("http"):
        return media_string
    return None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_to_disk(url: str, save_path: str) -> bool:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=TIMEOUT, headers=headers, stream=True)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(save_path, format="JPEG")
        log(f"    [dl ok]  {url[:70]}  ({os.path.getsize(save_path)//1024} KB)")
        return True
    except Exception as e:
        log(f"    [dl fail]  {e}")
        return False


# ---------------------------------------------------------------------------
# VLM inference — fixed pixel shape, aggressive cleanup
# ---------------------------------------------------------------------------
def run_vlm(model, processor, image_path: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text": VLM_PROMPT},
            ],
        }
    ]

    inputs        = None
    generated_ids = None
    image_inputs  = None
    video_inputs  = None

    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        del image_inputs, video_inputs
        image_inputs = video_inputs = None

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        trimmed = [out[len(inp):]
                   for inp, out in zip(inputs["input_ids"], generated_ids)]

        result = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # Collapse any newlines the model snuck in despite the prompt
        result = re.sub(r"\s+", " ", result).strip()

        # Truncate to word budget
        words = result.split()
        if len(words) > MAX_DESC_WORDS:
            result = " ".join(words[:MAX_DESC_WORDS])

        log(f"    [vlm]  {result[:120]}")
        return result

    finally:
        if inputs        is not None: del inputs
        if generated_ids is not None: del generated_ids
        if image_inputs  is not None: del image_inputs
        if video_inputs  is not None: del video_inputs
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Delete temp file
# ---------------------------------------------------------------------------
def delete_temp(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Resume by id — safe when row counts differ
# ---------------------------------------------------------------------------
def load_with_resume(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if "vlm_description" not in df.columns:
        df["vlm_description"] = ""

    if os.path.exists(output_path) and output_path != input_path:
        saved = pd.read_csv(output_path)
        if "id" in saved.columns and "vlm_description" in saved.columns:
            saved = saved[["id", "vlm_description"]].rename(
                columns={"vlm_description": "_saved"})
            df = df.merge(saved, on="id", how="left")
            mask = df["_saved"].notna() & (df["_saved"].astype(str).str.strip() != "")
            df.loc[mask, "vlm_description"] = df.loc[mask, "_saved"]
            df.drop(columns=["_saved"], inplace=True)
            log(f"Resumed: merged saved descriptions by id")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 60)
    log(f"  Qwen2.5-VL-3B Enrichment  (fixed prompt)")
    log(f"  Input  : {INPUT_CSV}")
    log(f"  Output : {OUTPUT_CSV}")
    log("=" * 60)

    if not os.path.exists(INPUT_CSV):
        log(f"ERROR: {INPUT_CSV} not found.")
        sys.exit(1)

    log("Loading model...")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
    )
    model.eval()
    log("Model ready.\n")

    df = load_with_resume(INPUT_CSV, OUTPUT_CSV)

    needs   = df["vlm_description"].isna() | (df["vlm_description"].astype(str).str.strip() == "")
    indices = df.index[needs].tolist()
    log(f"Already done : {len(df) - len(indices)} / {len(df)}")
    log(f"Remaining    : {len(indices)}\n")

    if not indices:
        log("Nothing to do.")
        return

    loop_n = 0
    try:
        for idx in tqdm(indices, desc="VLM Progress", file=sys.stdout):
            row = df.loc[idx]
            log(f"\n{'─'*60}")
            log(f"  Row {idx}  |  @{row.get('username','?')}  |  #{loop_n+1}")

            url = extract_image_url(str(row.get("media", "")))
            if not url:
                df.at[idx, "vlm_description"] = "no media"
                loop_n += 1
                continue

            log(f"    [url]  {url[:80]}")
            if not download_to_disk(url, TMP_IMAGE_PATH):
                df.at[idx, "vlm_description"] = "media could not be processed"
                loop_n += 1
                continue

            try:
                desc = run_vlm(model, processor, TMP_IMAGE_PATH)
                df.at[idx, "vlm_description"] = desc if desc else "media could not be processed"
            except Exception as e:
                log(f"    [error]  {e}")
                df.at[idx, "vlm_description"] = "media could not be processed"
            finally:
                delete_temp(TMP_IMAGE_PATH)

            loop_n += 1

            if loop_n % SAVE_INTERVAL == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                log(f"\n  [checkpoint]  {loop_n} done → {OUTPUT_CSV}")

    except KeyboardInterrupt:
        log("\nInterrupted — saving...")
    finally:
        delete_temp(TMP_IMAGE_PATH)
        df.to_csv(OUTPUT_CSV, index=False)
        log(f"\nFinal save → {OUTPUT_CSV}")

    valid  = df["vlm_description"].notna() & ~df["vlm_description"].isin(
                 ["", "no media", "media could not be processed"])
    log(f"\n{'='*60}")
    log(f"  Valid desc  : {valid.sum()}")
    log(f"  No media    : {(df['vlm_description']=='no media').sum()}")
    log(f"  Failed      : {(df['vlm_description']=='media could not be processed').sum()}")
    log(f"  Pending     : {df['vlm_description'].isna().sum()}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
