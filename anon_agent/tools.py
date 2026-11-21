"""Tools of the cloud coordinator. Each one works on the local vault and returns ONLY
placeholders / counts - never a raw value - so the coordinator stays clean.
Heavy work (OCR, inpainting, rendering) runs in a thread so the live trace keeps streaming.
"""
import asyncio
import io
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from google.adk.tools import ToolContext
from PIL import Image

import erase as E
import fake as F
import ocr as O
import pii as P
import render as R
from . import audit as A
from . import specialists as S
from .vault import VAULT

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_DIRS = [ROOT / "Docs", ROOT / "uploads"]  # least privilege: the only folders tools may read
(ROOT / "uploads").mkdir(exist_ok=True)


def _rid(tc):
    return tc.session.id if tc is not None else None


def _state(tc, stage, doc_id):
    """Tell the UI that the document state changed (it fetches images from the local API)."""
    A.emit("state", _rid(tc), agent="", zone="tool", name=stage, doc_id=doc_id)


def load_image(data: bytes, filename: str) -> Image.Image:
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        return Image.open(io.BytesIO(doc[0].get_pixmap(dpi=150).tobytes("png"))).convert("RGB")
    return Image.open(io.BytesIO(data)).convert("RGB")


# --------------------------------------------------------------------------------------------- load
async def load_document(path: str, tool_context: ToolContext = None) -> dict:
    """Load a document image/PDF (only from the Docs/ or uploads/ folder) and OCR it on-device.

    Args:
        path: file path relative to the project, e.g. "Docs/SAMPLE_Arztbrief.png" or "uploads/doc_ab12.png".
    Returns: doc_id and the number of text lines found.
    """
    p = (ROOT / path).resolve()
    if not any(p.is_relative_to(d.resolve()) for d in ALLOWED_DIRS) or not p.is_file():
        A.emit("policy", _rid(tool_context), agent="coordinator", zone="policy", name="path_guard",
               decision="BLOCK", detail=f"path outside the allowed folders: {path}")
        return {"error": "policy: only files in Docs/ or uploads/ may be opened"}
    img = load_image(p.read_bytes(), p.name)
    doc = VAULT.add(p.stem, img)
    doc.run_id = A.RUN_ID.get() or _rid(tool_context)
    doc.lines = await asyncio.to_thread(O.ocr, img)
    _state(tool_context, "loaded", doc.doc_id)
    words = sum(len(ln["text"].split()) for ln in doc.lines)
    if len(doc.lines) < 4 or words < 15:
        A.emit("policy", _rid(tool_context), agent="gatekeeper", zone="policy", name="content_gate", decision="BLOCK",
               detail=f"only {len(doc.lines)} text line(s) / {words} word(s) found - this is not a document (photo?). "
                      f"Nothing to anonymise, the agents stop here.")
        return {"error": "policy: no document text found (is this a photo?). Stop and tell the user.",
                "ocr_lines": len(doc.lines)}
    A.emit("policy", _rid(tool_context), agent="gatekeeper", zone="policy", name="content_gate", decision="ALLOW",
           detail=f"{len(doc.lines)} lines / {words} words of document text")
    return {"doc_id": doc.doc_id, "width": img.width, "height": img.height, "ocr_lines": len(doc.lines)}


# --------------------------------------------------------------------------------------------- analyse
def _validate_subject(doc, proposed, quarantined_ids):
    """Policy: the subject must be a private person named on a non-quarantined line, and should be
    the addressee if the document has a recipient block."""
    lines = doc.lines
    clean = [ln if i not in quarantined_ids else {**ln, "text": "", "words": []} for i, ln in enumerate(lines)]
    subject = P.pick_subject(clean, proposed, [])
    notes = []
    if subject != proposed:
        notes.append(f"Gemma's subject rejected ({'company/doctor' if P.COMPANY_RE.search(proposed or '') else 'not on a trusted line'}) "
                     f"→ using the addressee")
    blocks = P.block_names(clean)
    if blocks and subject and not any(P._compatible(b, subject) or P._n(b) == P._n(subject) for b in blocks):
        notes.append("subject is not the addressee → using the addressee")
        subject = blocks[0]
    return subject, notes


async def analyse_document(doc_id: str, tool_context: ToolContext = None) -> dict:
    """Run the on-device specialists (privacy analyst + injection sentinel, in parallel and isolated)
    and apply the anonymisation policy. Returns placeholders only - never the personal data itself.

    Args:
        doc_id: the id returned by load_document.
    """
    doc = VAULT.get(doc_id)
    rid = _rid(tool_context)
    text = P.layout_text(doc.lines, doc.img.size)
    wrapped = f"<document>\n{text}\n</document>"

    rule_hits = S.regex_scan(doc.lines)
    for q in rule_hits:
        A.emit("policy", rid, agent="sentinel", zone="policy", name="injection_rules", decision="QUARANTINE",
               detail=f"line L{q['line']}: {q['reason']}")

    analysis, sentinel = await asyncio.gather(S.run_isolated(S.analyst, wrapped),
                                              S.run_isolated(S.sentinel, wrapped))
    suspicious = {q["line"]: {**q, "by": "rules"} for q in rule_hits}
    for q in (sentinel or {}).get("suspicious", []):
        if not (0 <= int(q.get("line", -1)) < len(doc.lines)) or q["line"] in suspicious:
            continue
        if not S.plausible_injection(doc.lines[q["line"]]["text"]):
            A.emit("policy", rid, agent="sentinel", zone="policy", name="sentinel_check", decision="DROP",
                   detail=f"line L{q['line']} flagged by the model but has no instruction cues - claim rejected")
            continue
        suspicious[q["line"]] = {"line": q["line"], "reason": q.get("reason", ""), "by": "sentinel"}
        A.emit("policy", rid, agent="sentinel", zone="policy", name="injection_sentinel", decision="QUARANTINE",
               detail=f"line L{q['line']}: {q.get('reason', '')}")
    doc.quarantined = [{**q, "box": doc.lines[q["line"]]["box"]} for q in suspicious.values()]
    qids = set(suspicious)

    llm_items = [it for it in (analysis or {}).get("items", []) if int(it.get("line", -1)) not in qids]
    dropped = len((analysis or {}).get("items", [])) - len(llm_items)
    if dropped:
        A.emit("policy", rid, agent="analyst", zone="policy", name="quarantine_filter", decision="DROP",
               detail=f"{dropped} analyst item(s) came from quarantined lines")

    subject, notes = _validate_subject(doc, str((analysis or {}).get("subject", "")).strip(), qids)
    for n in notes:
        A.emit("policy", rid, agent="analyst", zone="policy", name="subject_check", decision="OVERRIDE", detail=n)
    if not subject:
        return {"error": "no data subject found - ask the user who the document belongs to"}

    take, keep = P.scope(doc.lines, doc.img.size, subject, llm_items)
    doc.subject, doc.items, doc.kept, doc.llm_items = subject, take, keep, llm_items
    doc.gender = P.gender_hint(doc.lines, take)
    for i, it in enumerate(doc.items):
        it["id"] = i
        it["ph"] = doc.placeholder(it["label"], it["text"])
    doc.subject_placeholder()
    A.emit("policy", rid, agent="analyst", zone="policy", name="scope_rules", decision="ALLOW",
           detail=f"{len(take)} item(s) belong to {doc.subject_placeholder()}, {len(keep)} kept as someone else's")
    _state(tool_context, "analysed", doc_id)

    kinds = {}
    for k in keep:
        kinds[k["label"]] = kinds.get(k["label"], 0) + 1
    out = {
        "subject": doc.subject_placeholder(),
        "to_replace": [{"id": it["id"], "placeholder": it["ph"], "type": it["label"], "rule": it["why"]} for it in take],
        "kept_someone_elses": kinds,
        "quarantined_lines": [{"line": q["line"], "reason": q["reason"], "found_by": q["by"]} for q in doc.quarantined],
    }
    return _scrubbed(out, doc)


def _scrubbed(obj, doc):
    import json
    return json.loads(doc.scrub(json.dumps(obj, ensure_ascii=False)))


# --------------------------------------------------------------------------------------------- redact
def _redact_sync(doc):
    items = [{"box": it["box"], "label": it["label"], "text": it["text"], "trail": it.get("trail", ""),
              "full_line": it.get("full_line", False)} for it in doc.items]
    levels = [doc.levels.get(it["id"], "auto") for it in doc.items]
    erased, einfo, _ = E.erase(doc.img, [it["box"] for it in items], level=levels, ghost_check=False)
    faker = F.Faker2(doc.name)
    faker.gender = doc.gender
    anon, rinfo = R.replace(doc.img, erased, items, doc.lines, faker)
    doc.erased, doc.anonymized, doc.faker = erased, anon, faker
    doc.render_info = [{**e, **r} for e, r in zip(einfo, rinfo)]
    for it, r in zip(doc.items, rinfo):
        if r.get("fake"):
            fake = r["fake"][:len(r["fake"]) - len(it.get("trail", ""))] if it.get("trail") else r["fake"]
            doc.fakes[it["text"]] = fake
            it["fake"] = fake
    return doc


async def redact_document(doc_id: str, stronger_for: Optional[List[int]] = None,
                          tool_context: ToolContext = None) -> dict:
    """Erase the subject's data and write believable fakes in the matching font (on-device).

    Args:
        doc_id: the document.
        stronger_for: item ids the verifier flagged; they are erased with a stronger method (inpainting).
    """
    doc = VAULT.get(doc_id)
    if not doc.items:
        return {"error": "nothing to redact - run analyse_document first"}
    for i in stronger_for or []:
        cur = doc.levels.get(i) or next((r.get("level") for r in doc.render_info[i:i + 1]), "fill")
        doc.levels[i] = E.LEVELS[min(E.LEVELS.index(cur) + 1, len(E.LEVELS) - 1)] if cur in E.LEVELS else "lama"
    await asyncio.to_thread(_redact_sync, doc)
    _state(tool_context, "redacted", doc_id)
    counts = {}
    for r in doc.render_info:
        counts[r.get("level", "?")] = counts.get(r.get("level", "?"), 0) + 1
    fam = next((r.get("doc_family") for r in doc.render_info if r.get("doc_family")), None)
    return {"replaced": sum(1 for r in doc.render_info if r.get("fake")), "erase_methods": counts, "font": fam,
            "items": [{"id": it["id"], "placeholder": it["ph"], "method": r.get("level")}
                      for it, r in zip(doc.items, doc.render_info)]}


# --------------------------------------------------------------------------------------------- verify
def _verify_sync(doc):
    W, H = doc.img.size
    # 1) leftover ink: OCR the erased image, any word inside an erased box = ghost
    er_lines = O.ocr(doc.erased)
    words = [w["box"] for ln in er_lines for w in ln["words"]]
    ghosts = [it["id"] for it in doc.items if any(E._overlap(it["box"], wb) > 0.3 for wb in words)]
    # 2) leaks: OCR the final image and look for every raw value
    an_lines = O.ocr(doc.anonymized)
    an_text = "\n".join(ln["text"] for ln in an_lines)
    leaked = doc.leaks(an_text)
    # 3) nothing else changed: pixels outside the (padded) boxes must be identical
    a, b = np.asarray(doc.img.convert("RGB")).astype(int), np.asarray(doc.anonymized.convert("RGB")).astype(int)
    mask = np.ones((H, W), bool)
    for it in doc.items:
        x1, y1, x2, y2 = it["box"]
        pad = max(12, (y2 - y1))
        mask[max(0, y1 - pad):y2 + pad, max(0, x1 - 2 * pad):x2 + 3 * pad] = False
    changed_outside = int((np.abs(a - b).sum(-1)[mask] > 0).sum())
    return ghosts, leaked, an_text, changed_outside


async def verify_document(doc_id: str, tool_context: ToolContext = None) -> dict:
    """Re-read the result on-device: leftover ink, leaked values (rules + local Gemma), and whether any
    pixel outside the replaced boxes changed.

    Args:
        doc_id: the document.
    """
    doc = VAULT.get(doc_id)
    if doc.anonymized is None:
        return {"error": "run redact_document first"}
    ghosts, leaked, an_text, changed_outside = await asyncio.to_thread(_verify_sync, doc)
    secrets = [it["text"] for it in doc.items]
    v = await S.run_isolated(S.verifier, "PRIVATE VALUES:\n" + "\n".join(f"- {s}" for s in secrets) +
                             f"\n\n<document>\n{an_text}\n</document>")
    # the model is advisory: only count a claimed leak if it really is in the text
    model_leaks = [x for x in (v or {}).get("leaked", []) if isinstance(x, str) and len(x) >= 3
                   and x.lower() in an_text.lower() and any(x.lower() in s.lower() or s.lower() in x.lower() for s in secrets)]
    rejected = len((v or {}).get("leaked", [])) - len(model_leaks)
    if rejected:
        A.emit("policy", _rid(tool_context), agent="verifier", zone="policy", name="verifier_check", decision="DROP",
               detail=f"{rejected} leak claim(s) by the verifier model not found in the text")
    leaked_ids = sorted({it["id"] for it in doc.items for x in leaked + model_leaks
                         if x.lower() in it["text"].lower() or it["text"].lower() in x.lower()})
    doc.verify = {"ghost_items": ghosts, "leaked_items": leaked_ids, "pixels_changed_outside_boxes": changed_outside}
    clean = not ghosts and not leaked_ids
    _state(tool_context, "verified", doc_id)
    return {"clean": clean,
            "still_readable_item_ids": ghosts,
            "leaked_item_ids": leaked_ids,
            "leaked_placeholders": [doc.items[i]["ph"] for i in leaked_ids],
            "pixels_changed_outside_boxes": changed_outside}


# --------------------------------------------------------------------------------------------- finalize
async def finalize(doc_id: str, tool_context: ToolContext = None) -> dict:
    """Mark the document as finished and publish the anonymised version to the user (locally).

    Args:
        doc_id: the document.
    """
    doc = VAULT.get(doc_id)
    out = ROOT / "uploads" / f"{doc_id}_anonymised.pdf"
    doc.anonymized.convert("RGB").save(out, format="PDF", resolution=150)
    _state(tool_context, "done", doc_id)
    return {"status": "done", "saved_as": str(out.relative_to(ROOT)), "replaced": len(doc.items)}


# --------------------------------------------------------------------------------------------- naive (for the demo)
async def read_document_text(doc_id: str, tool_context: ToolContext = None) -> dict:
    """NAIVE PIPELINE ONLY: returns the raw document text (this is what a typical agent would do).

    Args:
        doc_id: the document.
    """
    doc = VAULT.get(doc_id)
    return {"text": "\n".join(f"L{i}: {ln['text']}" for i, ln in enumerate(doc.lines))}
