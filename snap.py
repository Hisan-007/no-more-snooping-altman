"""Snap Gemma's (loose) detections onto pixel-exact OCR word boxes.

For every Gemma item we search all contiguous word spans inside every OCR line,
score them by fuzzy text similarity, and use Gemma's rough box only as a
tie-breaker (e.g. "Deutschland" appears 4x on the invoice).
"""
import math
import re
from difflib import SequenceMatcher


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zäöüß@.+]", "", s.lower())


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def snap(items, lines, img_size, min_ratio=0.7, dist_weight=2.0, full_line_bonus=0.15):
    """Returns new list of items with `box` replaced by the OCR span box,
    `gemma_box` keeping the original, `ocr_text` the matched text and `match` the score.
    Items that can't be matched are kept with snapped=False."""
    w, h = img_size
    diag = math.hypot(w, h)
    out = []
    used = {}  # line index -> list of (i, j) word spans already claimed

    for it in items:
        target = _norm(it["text"])
        if not target:
            continue
        n_target_words = max(1, len(it["text"].split()))
        gc = _center(it["box"])
        best = None
        for li, ln in enumerate(lines):
            ws = ln["words"]
            for i in range(len(ws)):
                for j in range(i + 1, min(len(ws), i + n_target_words + 3) + 1):
                    if any(i < uj and ui < j for ui, uj in used.get(li, [])):
                        continue  # overlaps something already matched
                    # exact substring of the line, so inner punctuation ("Straße, 11") is kept
                    span_text = ln["text"][ws[i]["start"]:ws[j - 1]["end"]] if "start" in ws[i] \
                        else " ".join(x["text"] for x in ws[i:j])
                    ratio = SequenceMatcher(None, target, _norm(span_text)).ratio()
                    if ratio < min_ratio:
                        continue
                    box = _union([x["box"] for x in ws[i:j]])
                    c = _center(box)
                    dist = math.hypot(c[0] - gc[0], c[1] - gc[1]) / diag
                    score = ratio - dist_weight * dist
                    if i == 0 and j == len(ws):
                        score += full_line_bonus
                    if best is None or score > best[0]:
                        best = (score, ratio, box, span_text, (li, i, j))
        if best is None:
            out.append({**it, "gemma_box": it["box"], "snapped": False, "match": 0.0})
            continue
        _, ratio, box, span_text, (li, i, j) = best
        used.setdefault(li, []).append((i, j))
        ln, last = lines[li], lines[li]["words"][j - 1]
        # punctuation glued to the end of the span (e.g. "Sauret,") sits inside the OCR box
        trail = re.match(r"[,;:.|]*", ln["text"][last["end"]:]).group(0) if "end" in last else ""
        out.append({**it, "gemma_box": it["box"], "box": box, "ocr_text": span_text, "trail": trail,
                    "full_line": i == 0 and j == len(lines[li]["words"]),
                    "snapped": True, "match": round(ratio, 2)})
    return out
