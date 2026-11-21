"""Write fake text into erased boxes so it looks like it was always there.

Style matching ("render & compare"): we know what the original text says (OCR),
so for every candidate font we render that exact text, fit size + horizontal
scale to the original ink, and score how well the glyph shapes overlap. The
winner gives font, size, stretch, baseline and colour; the fake is then drawn
with the same settings, picking the fake candidate whose width fits best.
"""
import glob
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import erase as E

SS = 4  # supersampling factor for rendering
HERE = Path(__file__).parent

SYS = "/System/Library/Fonts"
SUP = "/System/Library/Fonts/Supplemental"


def _catalog():
    """List of (label, path, index, variation-axes-or-None)."""
    c = []
    sf = f"{SYS}/SFNS.ttf"
    if os.path.exists(sf):  # SF Pro: very close to Inter / modern UI sans
        for w in (400, 500, 600, 700):
            c.append((f"SF Pro {w}", sf, 0, [100, 17, 400, w]))  # opsz 17 = "Text" cut, for body sizes
    for path, faces in [
        (f"{SYS}/Helvetica.ttc", {0: "Regular", 1: "Bold", 4: "Light"}),
        (f"{SYS}/HelveticaNeue.ttc", {0: "Regular", 1: "Bold", 10: "Medium", 7: "Light"}),
        (f"{SYS}/Avenir Next.ttc", {7: "Regular", 5: "Medium", 2: "Demi Bold", 0: "Bold"}),
    ]:
        if os.path.exists(path):
            fam = Path(path).stem
            for idx, style in faces.items():
                c.append((f"{fam} {style}", path, idx, None))
    for name in ["Arial", "Arial Bold", "Arial Narrow", "Times New Roman", "Times New Roman Bold",
                 "Georgia", "Georgia Bold", "Verdana", "Verdana Bold", "Tahoma", "Tahoma Bold",
                 "Trebuchet MS", "Trebuchet MS Bold", "Courier New", "Courier New Bold"]:
        p = f"{SUP}/{name}.ttf"
        if os.path.exists(p):
            c.append((name, p, 0, None))
    # user-supplied fonts (drop Inter, Roboto, ... into ./fonts)
    for p in sorted(glob.glob(str(HERE / "fonts" / "*.[ot]tf"))):
        c.append((Path(p).stem, p, 0, None))
    return c


CATALOG = _catalog()


@lru_cache(maxsize=4096)
def _font(ci: int, size_q: int):
    """size_q = size in 1/4 px (cache-friendly)."""
    label, path, idx, var = CATALOG[ci]
    f = ImageFont.truetype(path, size_q / 4, index=idx)
    if var:
        f.set_variation_by_axes(var)
    return f


def _render_alpha(ci, text, size):
    """Render text supersampled; return (alpha float32 [0..1] at 1x, baseline y, origin x) - tight-ish canvas."""
    f = _font(ci, max(4, round(size * SS * 4)))
    l, t, r, b = f.getbbox(text, anchor="ls")
    pad = 2 * SS
    W, H = r - l + 2 * pad, b - t + 2 * pad
    W += (-W) % SS
    H += (-H) % SS
    im = Image.new("L", (int(W), int(H)), 0)
    ox, oy = pad - l, pad - t  # origin (baseline-left) in supersampled px
    ImageDraw.Draw(im).text((ox, oy), text, font=f, fill=255, anchor="ls")
    im = im.resize((int(W) // SS, int(H) // SS), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255, oy / SS, ox / SS


def _tight(alpha, thr=0.35):
    ys, xs = np.nonzero(alpha > thr)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _resize(a, w, h):
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((max(1, w), max(1, h)), Image.BILINEAR),
                      np.float32) / 255


def original_style(orig: np.ndarray, box):
    """Measure the original text: soft ink alpha, tight ink box, colour, background."""
    bg = E.analyse_background(orig, box)
    h, w = orig.shape[:2]
    x1, y1, x2, y2 = E._clip(E._grow(box, E.INK_PAD), w, h)
    crop = orig[y1:y2, x1:x2].astype(np.float32)
    diff = np.max(np.abs(crop - bg["median"]), axis=2)
    full = diff.max() if diff.size else 0
    if full < 25:
        return None
    alpha = np.clip(diff / full, 0, 1)
    # keep only glyph blobs that mostly sit inside the OCR box: drops descenders/ascenders of the
    # neighbouring lines that poke into the padding (tight footers)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((alpha > 0.2).astype(np.uint8), connectivity=8)
    bx1, by1, bx2, by2 = box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1
    keep = np.zeros(n, bool)
    for c in range(1, n):
        cx, cy, cw, ch, _ = stats[c]
        iy = max(0, min(cy + ch, by2) - max(cy, by1))
        ix = max(0, min(cx + cw, bx2) - max(cx, bx1))
        keep[c] = iy >= 0.5 * ch and ix > 0
    soft = cv2.dilate(keep[lab].astype(np.uint8), np.ones((3, 3), np.uint8))  # keep AA fringe
    alpha = alpha * soft
    tb = _tight(alpha)
    if tb is None:
        return None
    strong = alpha > 0.85
    colour = np.median(crop[strong], axis=0) if strong.sum() >= 3 else crop.reshape(-1, 3)[np.argmax(diff)]
    tx1, ty1, tx2, ty2 = tb
    return {
        "alpha": alpha[ty1:ty2, tx1:tx2], "ink": [x1 + tx1, y1 + ty1, x1 + tx2, y1 + ty2],
        "colour": colour, "bg": bg,
    }


def family(ci):
    label, path, idx, var = CATALOG[ci]
    if var:
        return "SF Pro"
    if path.endswith(".ttc"):
        return Path(path).stem
    return label.replace(" Bold", "").replace(" Narrow", "").replace(" Italic", "")


def fit_all(orig_style, text):
    """Fit every font in the catalog. Returns {ci: fit}."""
    fits = {}
    for ci in range(len(CATALOG)):
        f = _fit_one(orig_style, text, ci)
        if f:
            fits[ci] = f
    return fits


def choose_fonts(fits_per_item):
    """Documents use 1-2 font families. Pick the family with the lowest total error over all
    boxes, then per box the best face (weight) inside that family."""
    fam_err = {}
    for fits in fits_per_item:
        if not fits:
            continue
        per_fam = {}
        for ci, f in fits.items():
            fam = family(ci)
            per_fam[fam] = min(per_fam.get(fam, 9), f["err"])
        worst = max(per_fam.values())
        for fam in {family(ci) for ci in range(len(CATALOG))}:
            fam_err[fam] = fam_err.get(fam, 0) + per_fam.get(fam, worst)
    doc_fam = min(fam_err, key=fam_err.get) if fam_err else None
    chosen = []
    for fits in fits_per_item:
        inside = {ci: f for ci, f in fits.items() if family(ci) == doc_fam}
        pool = inside or fits
        chosen.append(min(pool.values(), key=lambda f: f["err"]) if pool else None)
    return chosen, doc_fam, fam_err


def harmonise(chosen, fits_per_item, styles, override_margin=0.04):
    """Text that looks alike (same colour, similar size) should get the same weight and size.
    Group boxes, pick the face with the lowest summed error per group and use the group's median
    size - unless a box fits a different face clearly better (e.g. one bold line in a block)."""
    idx = [i for i, c in enumerate(chosen) if c is not None]
    groups = []
    for i in idx:
        col = styles[i]["colour"]
        for g in groups:
            j = g[0]
            if np.abs(styles[j]["colour"] - col).max() < 30 and abs(chosen[i]["size"] / chosen[j]["size"] - 1) < 0.12:
                g.append(i)
                break
        else:
            groups.append([i])
    out = list(chosen)
    for g in groups:
        faces = set.intersection(*[{ci for ci in fits_per_item[i] if family(ci) == family(chosen[i]["ci"])} for i in g])
        if not faces:
            continue
        face = min(faces, key=lambda ci: sum(fits_per_item[i][ci]["err"] for i in g))
        for i in g:
            own = chosen[i]
            grp = fits_per_item[i][face]
            pick = own if own["ci"] != face and own["err"] < grp["err"] - override_margin else grp
            out[i] = {**pick, "group": g[0]}
    return out


def _fit_one(orig_style, text, ci):
    """Fit one font: size from ink height, stretch from ink width, score glyph overlap."""
    A = orig_style["alpha"]
    Ht, Wt = A.shape
    best = None
    for ci in [ci]:
        a, by, ox = _render_alpha(ci, text, 100)
        tb = _tight(a)
        if tb is None:
            continue
        size = 100 * Ht / (tb[3] - tb[1])
        for _ in range(2):  # refine: thresholded ink height isn't perfectly linear in size
            a, by, ox = _render_alpha(ci, text, size)
            tb = _tight(a)
            if tb is None:
                break
            size *= Ht / max(1, tb[3] - tb[1])
        if tb is None:
            continue
        rw, rh = tb[2] - tb[0], tb[3] - tb[1]
        sx = Wt / rw
        if not 0.75 < sx < 1.33:
            continue
        r = _resize(a[tb[1]:tb[3], tb[0]:tb[2]], Wt, Ht)
        err = float(np.abs(r - A).mean()) + 0.15 * abs(sx - 1)
        if best is None or err < best["err"]:
            best = {"ci": ci, "font": CATALOG[ci][0], "size": size, "sx": sx, "err": err,
                    # offsets from ink box to the text origin, in output px
                    "base_from_top": (by - tb[1]) * (Ht / rh), "origin_from_left": (ox - tb[0]) * sx}
    return best


def _alignment(box, lines):
    """'right' if this full line shares its right edge with a neighbour line (and not its left)."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    right = left = 0
    for ln in lines:
        lx1, ly1, lx2, ly2 = ln["box"]
        if abs(ly1 - y1) < 2 or abs(ly1 - y1) > 6 * h:
            continue
        right += abs(lx2 - x2) <= 4
        left += abs(lx1 - x1) <= 4
    return "right" if right > left else "left"


def draw_text(canvas: Image.Image, text, font_ci, size, sx, colour, origin_xy):
    """Draw text with origin (baseline-left) at origin_xy, horizontally scaled by sx. Sub-pixel accurate."""
    f = _font(font_ci, max(4, round(size * SS * 4)))
    l, t, r, b = f.getbbox(text, anchor="ls")
    ox, oy = origin_xy
    pad = 2 * SS
    # integer top-left of the 1x layer + the fractional remainder, which we bake into the
    # supersampled drawing position so the origin lands exactly on (ox, oy)
    px_f, py_f = ox - (pad - l) / SS * sx, oy - (pad - t) / SS
    px, py = int(np.floor(px_f)), int(np.floor(py_f))
    frx, fry = px_f - px, py_f - py
    W = int(r - l + 2 * pad + 2 * SS)
    H = int(b - t + 2 * pad + 2 * SS)
    W += (-W) % SS
    H += (-H) % SS
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).text((pad - l + frx * SS / sx, pad - t + fry * SS), text, font=f, fill=255, anchor="ls")
    im = im.resize((max(1, round(W / SS * sx)), H // SS), Image.LANCZOS)
    col = tuple(int(c) for c in colour)
    layer = Image.new("RGBA", im.size, col + (0,))
    layer.putalpha(im)
    canvas.alpha_composite(layer, (px, py)) if canvas.mode == "RGBA" else canvas.paste(layer, (px, py), layer)
    return [px, py, px + im.width, py + im.height]


def text_width(ci, text, size, sx):
    a, _, _ = _render_alpha(ci, text, size)
    tb = _tight(a)
    return 0 if tb is None else (tb[2] - tb[0]) * sx


def replace(orig_img: Image.Image, erased_img: Image.Image, items, lines, faker):
    """items: dicts with box, label, text (original, from OCR), selected. Returns (image, per-item info)."""
    orig = np.asarray(orig_img.convert("RGB"))
    out = erased_img.convert("RGB").copy()
    styles = [original_style(orig, it["box"]) for it in items]
    # OCR word boxes often include punctuation glued to the span ("Sauret," / "11,") - match
    # against what's actually inside the box and re-append it to the fake
    items = [{**it, "trail": it.get("trail", "")} for it in items]
    fits = [fit_all(st, it["text"] + it["trail"]) if st else {} for st, it in zip(styles, items)]
    chosen, doc_fam, _ = choose_fonts(fits)
    chosen = harmonise(chosen, fits, styles)
    info = []
    for it, st, m in zip(items, styles, chosen):
        box, text, label = it["box"], it["text"], it["label"]
        if st is None:
            info.append({"error": "no ink found"})
            continue
        if m is None:
            info.append({"error": "no font matched"})
            continue
        cands = [c + it["trail"] for c in faker.candidates(label, text)]
        Wt = st["ink"][2] - st["ink"][0]
        widths = [text_width(m["ci"], c, m["size"], m["sx"]) for c in cands]
        k = int(np.argmin([abs(w - Wt) for w in widths]))
        fake, fw = cands[k], widths[k]
        faker.commit(label, text, fake[:len(fake) - len(it["trail"])] if it["trail"] else fake)
        sx = m["sx"]
        if fw > Wt * 1.06:  # a bit too long: squeeze slightly (max 8%)
            sx *= max(0.92, Wt * 1.06 / fw)
            fw = text_width(m["ci"], fake, m["size"], sx)
        align = _alignment(box, lines) if it.get("full_line") else "left"
        ink_x1, ink_y1, ink_x2, _ = st["ink"]
        baseline = ink_y1 + m["base_from_top"]
        # where the fake's own ink starts/ends relative to its origin (side bearings differ per glyph)
        fa, _, fox = _render_alpha(m["ci"], fake, m["size"])
        ftb = _tight(fa) or (fox, 0, fox, 0)
        if align == "right":
            origin_x = ink_x2 - (ftb[2] - fox) * sx  # right ink edge stays put
        else:
            origin_x = ink_x1 - (ftb[0] - fox) * sx  # left ink edge stays put
        draw_text(out, fake, m["ci"], m["size"], sx, st["colour"], (origin_x, baseline))
        info.append({"fake": fake, "font": m["font"], "doc_family": doc_fam, "size": round(m["size"], 1), "stretch": round(sx, 3),
                     "colour": "#%02x%02x%02x" % tuple(int(c) for c in st["colour"]), "align": align,
                     "fit_err": round(m["err"], 3), "width_ratio": round(fw / max(1, Wt), 2)})
    return out, info
