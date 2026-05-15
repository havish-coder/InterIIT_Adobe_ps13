"""
Shared prompt builder — imported by both prep_llm_data.py and eval.py.
Changing the prompt here changes it consistently in train AND inference.
"""

import re
import pandas as pd

SYSTEM_MSG = "You are a helpful assistant."


def build_instruction(row: dict) -> str:
    """Return the user instruction string from a tweet metadata row dict."""
    try:
        dt = pd.to_datetime(str(row.get("date", "")))
        day_name = dt.strftime("%A")
        hour = dt.hour
    except Exception:
        day_name = "a weekday"
        hour = 12

    vlm = re.sub(r"\s+", " ", str(row.get("vlm_description", ""))).strip()
    if vlm.lower() in {"", "nan", "no media", "media could not be processed",
                       "media could not be downloaded"}:
        visual_context = ""
    else:
        visual_context = f"\nImage: {vlm}"

    company = row.get("inferred company", row.get("company", "Unknown"))
    username = row.get("username", "Unknown")

    try:
        likes = int(float(row.get("likes", 0)))
        if likes >= 1000:
            likes_ctx = f"\nExpected engagement: high ({likes:,} likes target)"
        elif likes >= 100:
            likes_ctx = f"\nExpected engagement: moderate ({likes} likes target)"
        else:
            likes_ctx = ""
    except Exception:
        likes_ctx = ""

    return (
        f"You are an expert social media manager. Write an engaging marketing tweet for {company} "
        f"(username: @{username}).\n"
        f"Context: It's {day_name} at {hour}:00.{visual_context}{likes_ctx}\n"
        f"Ensure the tweet fits the brand's style and incorporates appropriate hashtags."
    )


def build_messages(row: dict, include_response: bool = False) -> list:
    """Return the full chat messages list for Qwen ChatML format."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_instruction(row)},
    ]
    if include_response:
        content = re.sub(r"\s+", " ", str(row.get("content", ""))).strip()
        messages.append({"role": "assistant", "content": content})
    return messages
