"""Pixel-exact word boxes using macOS Vision (on-device OCR)."""
import io
import re

import Quartz
import Vision
from Foundation import NSData
from PIL import Image


def _cgimage(img: Image.Image):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def _to_px(bb, w, h):
    # Vision: normalized, origin bottom-left
    x, y = bb.origin.x, bb.origin.y
    bw, bh = bb.size.width, bb.size.height
    return [round(x * w), round((1 - y - bh) * h), round((x + bw) * w), round((1 - y) * h)]


def ocr(img: Image.Image, langs=("de-DE", "en-US")):
    """Returns list of lines: {text, box:[x1,y1,x2,y2], words:[{text, box}]}"""
    w, h = img.size
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    req.setRecognitionLanguages_(list(langs))
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(_cgimage(img), None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(err)

    lines = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)[0]
        text = str(cand.string())
        words = []
        for m in re.finditer(r"\S+", text):
            word = m.group(0).rstrip(",;:|")
            if not word:
                continue
            rng = (m.start(), len(word))
            box_obs, _ = cand.boundingBoxForRange_error_(rng, None)
            if box_obs is None:
                continue
            words.append({"text": word, "start": m.start(), "end": m.start() + len(word), "box": _to_px(box_obs.boundingBox(), w, h)})
        lines.append({"text": text, "box": _to_px(obs.boundingBox(), w, h), "words": words})
    return lines


if __name__ == "__main__":
    import sys
    for ln in ocr(Image.open(sys.argv[1])):
        print(ln["box"], ln["text"])
