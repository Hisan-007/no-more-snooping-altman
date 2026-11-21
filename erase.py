"""Erase text inside boxes so the background looks untouched.

Per box we look at a thin ring of background just outside the box and pick a level:
  fill - flat/gradient background: robust plane fit + matched noise    (microseconds)
  cv   - mild texture / lines crossing: OpenCV Telea on the ink mask     (milliseconds)
  lama - busy background: LaMa on a local crop (Apple GPU via MPS)       (~0.5s)

Only *ink* pixels (letters + a small halo) are replaced, so table borders,
underlines, highlights etc. outside the glyphs are never touched.
After erasing, an optional ghost check re-runs OCR and escalates any box where
text is still readable.
"""
import time

import cv2
import numpy as np
from PIL import Image

LEVELS = ["fill", "cv", "lama"]

INK_PAD = 3      # search ink this far outside the OCR box (descenders, anti-aliasing)
RING_IN = 5      # background ring: from box+RING_IN ...
RING_OUT = 9     # ... to box+RING_OUT


def _clip(b, w, h):
    x1, y1, x2, y2 = b
    return [max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))]


def _grow(b, d):
    return [b[0] - d, b[1] - d, b[2] + d, b[3] + d]


def _ring_pixels(arr, box):
    h, w = arr.shape[:2]
    ox1, oy1, ox2, oy2 = _clip(_grow(box, RING_OUT), w, h)
    ix1, iy1, ix2, iy2 = _clip(_grow(box, RING_IN), w, h)
    m = np.zeros((h, w), bool)
    m[oy1:oy2, ox1:ox2] = True
    m[iy1:iy2, ix1:ix2] = False
    ys, xs = np.nonzero(m)
    return xs, ys, arr[ys, xs].astype(np.float32)


def analyse_background(arr, box):
    """Robust background model from the ring: median colour, noise sigma, outlier fraction."""
    xs, ys, px = _ring_pixels(arr, box)
    med = np.median(px, axis=0)
    mad = np.median(np.abs(px - med), axis=0)
    sigma = float(np.max(1.4826 * mad))
    dev = np.max(np.abs(px - med), axis=1)
    outlier = dev > max(20.0, 4 * sigma)
    return {
        "xs": xs, "ys": ys, "px": px, "median": med, "sigma": sigma,
        "outlier_frac": float(outlier.mean()) if len(dev) else 0.0,
        # tight inliers for colour fitting: excludes anti-aliased edges of neighbouring text
        "inlier": dev <= max(6.0, 3 * sigma),
    }


def choose_level(bg):
    if bg["sigma"] < 4 and bg["outlier_frac"] < 0.03:
        return "fill"
    if bg["sigma"] < 12 and bg["outlier_frac"] < 0.20:
        return "cv"
    return "lama"


def ink_mask(arr, box, bg, dilate=2):
    """Pixels inside box(+pad) that differ from the background model -> the letters."""
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = _clip(_grow(box, INK_PAD), w, h)
    crop = arr[y1:y2, x1:x2].astype(np.float32)
    diff = np.max(np.abs(crop - bg["median"]), axis=2)
    thr = max(18.0, 3.5 * bg["sigma"])
    m = (diff > thr).astype(np.uint8)
    if dilate:
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=dilate)
    full = np.zeros((h, w), np.uint8)
    full[y1:y2, x1:x2] = m
    return full


def _fill(arr, mask, bg):
    """Fit colour = a*x + b*y + c on background inliers (handles scan gradients), add matched noise."""
    xs, ys, px = bg["xs"][bg["inlier"]], bg["ys"][bg["inlier"]], bg["px"][bg["inlier"]]
    ys_m, xs_m = np.nonzero(mask)
    if len(xs) < 10 or len(xs_m) == 0:
        arr[ys_m, xs_m] = bg["median"].astype(np.uint8)
        return
    A = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float32)
    coef, *_ = np.linalg.lstsq(A, px, rcond=None)
    pred = np.stack([xs_m, ys_m, np.ones_like(xs_m)], 1).astype(np.float32) @ coef
    if bg["sigma"] > 0.8:
        pred += np.random.normal(0, bg["sigma"], pred.shape[0])[:, None]
    arr[ys_m, xs_m] = np.clip(pred, 0, 255).astype(np.uint8)


def _cv(arr, mask, box):
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = _clip(_grow(box, 20), w, h)
    sub = np.ascontiguousarray(arr[y1:y2, x1:x2])
    out = cv2.inpaint(sub, mask[y1:y2, x1:x2], 3, cv2.INPAINT_TELEA)
    m = mask[y1:y2, x1:x2].astype(bool)
    arr[y1:y2, x1:x2][m] = out[m]


LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
_LAMA = None  # (model, device)


def _lama_model():
    """big-lama TorchScript (~200MB, cached in ~/.cache/torch). The file was traced on CUDA,
    so load it onto CPU first, then move to Apple GPU (MPS) if it works there."""
    global _LAMA
    if _LAMA is None:
        import torch
        from simple_lama_inpainting.utils import download_model
        model = torch.jit.load(download_model(LAMA_URL), map_location="cpu").eval()
        dev = torch.device("cpu")
        if torch.backends.mps.is_available():
            try:
                m = model.to("mps")
                with torch.inference_mode():
                    m(torch.zeros(1, 3, 64, 64, device="mps"), torch.zeros(1, 1, 64, 64, device="mps"))
                model, dev = m, torch.device("mps")
            except Exception:
                model = model.to("cpu")
        _LAMA = (model, dev)
    return _LAMA


def _lama(arr, mask, box):
    import torch
    model, dev = _lama_model()
    h, w = arr.shape[:2]
    bh = box[3] - box[1]
    margin = max(48, 2 * bh)
    x1, y1, x2, y2 = _clip(_grow(box, margin), w, h)
    sub = arr[y1:y2, x1:x2]
    m = mask[y1:y2, x1:x2]
    sh, sw = sub.shape[:2]
    ph, pw = (8 - sh % 8) % 8, (8 - sw % 8) % 8  # LaMa wants multiples of 8
    img_t = np.pad(sub, ((0, ph), (0, pw), (0, 0)), mode="reflect").astype(np.float32) / 255
    msk_t = np.pad(m, ((0, ph), (0, pw)), mode="reflect").astype(np.float32)
    with torch.inference_mode():
        out = model(torch.from_numpy(img_t).permute(2, 0, 1)[None].to(dev),
                    torch.from_numpy((msk_t > 0).astype(np.float32))[None, None].to(dev))
    out = out[0].permute(1, 2, 0).cpu().numpy()[:sh, :sw] * 255
    # LaMa shifts colours slightly (1-3 levels); re-anchor on the unmasked background around the box
    keep = ~cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=3).astype(bool)
    if keep.any():
        out += np.median(sub[keep].astype(np.float32), 0) - np.median(out[keep], 0)
    mb = m.astype(bool)
    arr[y1:y2, x1:x2][mb] = np.clip(out, 0, 255).astype(np.uint8)[mb]


def erase_box(arr, box, level="auto", dilate=2, src=None):
    """Erase in place. Background is read from the current `arr` (neighbours may already be
    erased); the ink mask is computed on `src` (the original) so re-runs still see the letters."""
    bg = analyse_background(arr, box)
    lvl = choose_level(bg) if level == "auto" else level
    mask = ink_mask(arr if src is None else src, box, bg, dilate=dilate)
    if lvl == "fill":
        _fill(arr, mask, bg)
    elif lvl == "cv":
        _cv(arr, mask, box)
    else:
        _lama(arr, mask, box)
    return {"level": lvl, "sigma": round(bg["sigma"], 2), "outlier_frac": round(bg["outlier_frac"], 3),
            "ink_px": int(mask.sum())}


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / area


def erase(img: Image.Image, boxes, level="auto", ghost_check=True):
    """Returns (erased PIL image, per-box info list, seconds). `level` may be one value or one per box."""
    t0 = time.time()
    original = np.array(img.convert("RGB"))
    arr = original.copy()
    levels = level if isinstance(level, (list, tuple)) else [level] * len(boxes)  # one level per box allowed
    info = [erase_box(arr, b, lv, src=original) for b, lv in zip(boxes, levels)]
    for d in info:
        d["ghost"] = False

    if ghost_check:
        import ocr as O
        for attempt in range(3):
            lines = O.ocr(Image.fromarray(arr))
            words = [wd["box"] for ln in lines for wd in ln["words"]]
            ghosts = [i for i, b in enumerate(boxes) if any(_overlap(b, wb) > 0.3 for wb in words)]
            if not ghosts or attempt == 2:
                for i in ghosts:
                    info[i]["ghost"] = True  # still readable after escalating
                break
            for i in ghosts:
                cur = info[i]["level"]
                nxt = LEVELS[min(LEVELS.index(cur) + 1, len(LEVELS) - 1)]
                info[i] = {**erase_box(arr, boxes[i], nxt, dilate=3 + attempt, src=original),
                           "escalated_from": info[i].get("escalated_from", cur), "ghost": False}
    return Image.fromarray(arr), info, time.time() - t0
