"""Ask a local Gemma (via Ollama) to find sensitive info in a document image
and return bounding boxes.

CLI:  .venv/bin/python detect.py Docs/SAMPLE_Arztbrief.png [--model gemma4:e2b-it-qat] [--think]
      -> writes <image>_boxes.png next to the scratch output + prints JSON
"""
import argparse
import base64
import io
import json
import re
import sys
import time

import requests
from PIL import Image, ImageDraw, ImageFont

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma4:e2b-it-qat"

LABELS = ["name", "email", "phone", "address", "iban", "id_number", "date", "company"]

PROMPT = """You are a PII detector for scanned documents.
Find every piece of personal / sensitive information in this image:
person names, company names, email addresses, phone numbers, postal addresses
(street, postcode+city, country), IBAN / bank details, ID-like numbers
(order numbers, invoice numbers, customer numbers, tracking numbers, tax IDs, register numbers).

Scan the WHOLE page top to bottom: the header block (order/invoice numbers and dates),
sender and recipient address blocks, tables, and the small print footer.
The first line of an address block is usually a person's name -> label it "name".
Do not include field labels like "Rechnungsnummer" - only the value.

Return ONLY a JSON object of this form:
{"items": [{"label": "<one of: %s>", "text": "<exact text as printed>", "box_2d": [ymin, xmin, ymax, xmax]}]}

box_2d must tightly enclose just that text, with coordinates normalized to 0-1000
(0,0 = top-left of the image, 1000,1000 = bottom-right).
Make one item per line of text (e.g. street and city are separate items).""" % ", ".join(LABELS)


def _to_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _parse_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # grab the outermost {...} or [...]
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        return json.loads(m.group(1)) if m else {"items": []}


def pad_to_square(img: Image.Image) -> Image.Image:
    """White-pad to a square (top-left anchored) so the vision encoder doesn't squash the aspect ratio."""
    w, h = img.size
    s = max(w, h)
    sq = Image.new("RGB", (s, s), "white")
    sq.paste(img.convert("RGB"), (0, 0))
    return sq


def detect(img: Image.Image, model: str = DEFAULT_MODEL, think: bool = True,
           coord_mode: str = "yxyx1000", square: bool = False):
    """Returns (items, raw_text, seconds). Each item: label, text, box=[x1,y1,x2,y2] in pixels, raw_box."""
    if square:
        # boxes come back relative to the square; since it's top-left anchored,
        # pixel coords are identical to the original image's.
        img = pad_to_square(img)
    w, h = img.size
    t0 = time.time()
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT, "images": [_to_png_b64(img)]}],
            "format": "json",
            "think": think,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=600,
    )
    r.raise_for_status()
    raw = r.json()["message"]["content"]
    dt = time.time() - t0

    data = _parse_json(raw)
    items = data.get("items", data) if isinstance(data, dict) else data
    out = []
    for it in items if isinstance(items, list) else []:
        b = it.get("box_2d") or it.get("box") or it.get("bbox")
        if not (isinstance(b, list) and len(b) == 4):
            continue
        try:
            b = [float(v) for v in b]
        except (TypeError, ValueError):
            continue
        out.append({
            "label": str(it.get("label", "?")),
            "text": str(it.get("text", "")),
            "raw_box": b,
            "box": to_pixels(b, w, h, coord_mode),
        })
    return out, raw, dt


def to_pixels(b, w, h, mode="yxyx1000"):
    """Convert a model box to [x1,y1,x2,y2] pixels."""
    if mode == "yxyx1000":
        y1, x1, y2, x2 = b
        x1, x2, y1, y2 = x1 / 1000 * w, x2 / 1000 * w, y1 / 1000 * h, y2 / 1000 * h
    elif mode == "xyxy1000":
        x1, y1, x2, y2 = b
        x1, x2, y1, y2 = x1 / 1000 * w, x2 / 1000 * w, y1 / 1000 * h, y2 / 1000 * h
    elif mode == "xyxy_px":
        x1, y1, x2, y2 = b
    else:
        raise ValueError(mode)
    return [round(min(x1, x2)), round(min(y1, y2)), round(max(x1, x2)), round(max(y1, y2))]


COLORS = {
    "name": "#e6194b", "email": "#3cb44b", "phone": "#4363d8", "address": "#f58231",
    "iban": "#911eb4", "id_number": "#008080", "date": "#9a6324", "company": "#f032e6",
}


def draw(img: Image.Image, items) -> Image.Image:
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()
    for it in items:
        c = COLORS.get(it["label"], "#000000")
        x1, y1, x2, y2 = it["box"]
        d.rectangle([x1, y1, x2, y2], outline=c, width=3)
        d.text((x1, max(0, y1 - 18)), it["label"], fill=c, font=font)
    return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--coords", default="yxyx1000", choices=["yxyx1000", "xyxy1000", "xyxy_px"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    img = Image.open(a.image)
    items, raw, dt = detect(img, a.model, a.think, a.coords)
    print(f"# {a.model} think={a.think} took {dt:.1f}s, {len(items)} items", file=sys.stderr)
    print(raw)
    out = a.out or a.image.rsplit(".", 1)[0] + "_boxes.png"
    draw(img, items).save(out)
    print(f"# saved {out}", file=sys.stderr)
