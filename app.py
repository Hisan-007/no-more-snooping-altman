"""Anonymise a document's data subject: upload -> OCR -> Gemma names whose document it is ->
rules pick that person's data -> erase + redraw with believable fakes.

Run:  .venv/bin/uvicorn app:app --port 8000
"""
import base64
import io
import time
import uuid
from pathlib import Path
from typing import List

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

import detect as D
import erase as E
import fake as F
import ocr as O
import pii as P
import render as R

app = FastAPI()
HERE = Path(__file__).parent
DOCS = {}  # doc_id -> PIL image (in-memory, POC only)


def _doc(doc_id):
    d = DOCS.get(doc_id)
    if d is None:
        raise HTTPException(404, "Unknown doc_id - run detection again")
    return d


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_image(data: bytes, filename: str) -> Image.Image:
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import fitz  # pymupdf
        doc = fitz.open(stream=data, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=150)  # first page only for the POC
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return Image.open(io.BytesIO(data)).convert("RGB")


@app.get("/")
def index():
    return FileResponse(HERE / "stage.html")


@app.get("/architecture")
def architecture():
    return FileResponse(HERE / "architecture.html")


@app.get("/trace")
def trace_ui():
    return FileResponse(HERE / "ui.html")


@app.get("/classic")
def classic():
    return FileResponse(HERE / "index.html")


def _samples():
    return [p for p in sorted((HERE / "Docs").iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}]


@app.get("/samples")
def samples():
    return {"samples": [p.name for p in _samples()]}


@app.get("/sample")
def sample(name: str = ""):
    """An image/PDF from Docs/ - handy for demos."""
    for p in _samples():
        if not name or p.name == name:
            return FileResponse(p, filename=p.name)
    raise HTTPException(404, "No sample in Docs/")


@app.get("/models")
def models():
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=5).json()["models"]
    except Exception as e:
        raise HTTPException(502, f"Ollama not reachable: {e}")
    names = [m["name"] for m in tags if "embed" not in m["name"]]
    return {"models": names, "default": D.DEFAULT_MODEL}


@app.post("/detect")
async def run_detect(file: UploadFile = File(...), model: str = Form(D.DEFAULT_MODEL),
                     think: bool = Form(True)):
    img = load_image(await file.read(), file.filename or "")

    t0 = time.time()
    lines = O.ocr(img)
    t_ocr = time.time() - t0

    res = P.find(lines, img.size, model=model, think=think)
    doc_id = uuid.uuid4().hex
    DOCS[doc_id] = {"img": img, "lines": lines, "name": file.filename or "doc",
                    "llm_items": res["llm_items"], "gender": res["gender"]}
    return {
        "doc_id": doc_id,
        "image": to_data_url(img),
        "width": img.width,
        "height": img.height,
        "model": model,
        "think": think,
        "subject": res["subject"],
        "timing": {"ocr_s": round(t_ocr, 2), "gemma_s": round(res["gemma_s"], 1)},
        "items": res["items"],
        "kept": res["kept"],
        "raw": res["raw"],
    }


class RescopeReq(BaseModel):
    doc_id: str
    subject: str


@app.post("/rescope")
def run_rescope(req: RescopeReq):
    """Re-apply the rules for a different / corrected data subject (no Gemma call)."""
    d = _doc(req.doc_id)
    take, keep = P.scope(d["lines"], d["img"].size, req.subject.strip(), d["llm_items"])
    d["gender"] = P.gender_hint(d["lines"], take)
    return {"subject": req.subject.strip(), "items": take, "kept": keep}


class EraseReq(BaseModel):
    doc_id: str
    boxes: List[List[int]]
    level: str = "auto"  # auto | fill | cv | lama
    ghost_check: bool = True


@app.post("/erase")
def run_erase(req: EraseReq):
    d = _doc(req.doc_id)
    if req.level not in ("auto", *E.LEVELS):
        raise HTTPException(400, f"level must be auto or one of {E.LEVELS}")
    out, info, dt = E.erase(d["img"], req.boxes, level=req.level, ghost_check=req.ghost_check)
    return {"image": to_data_url(out), "info": info, "seconds": round(dt, 2)}


class Item(BaseModel):
    box: List[int]
    label: str
    text: str            # the OCR text inside the box
    trail: str = ""      # punctuation glued to the end, inside the box
    full_line: bool = False


class AnonReq(BaseModel):
    doc_id: str
    items: List[Item]
    level: str = "auto"
    ghost_check: bool = True


@app.post("/anonymize")
def run_anonymize(req: AnonReq):
    """Erase the selected boxes, then write matched-style fake text into them."""
    d = _doc(req.doc_id)
    if req.level not in ("auto", *E.LEVELS):
        raise HTTPException(400, f"level must be auto or one of {E.LEVELS}")
    items = [it.dict() for it in req.items]
    erased, einfo, t_erase = E.erase(d["img"], [it["box"] for it in items], level=req.level,
                                     ghost_check=req.ghost_check)
    t0 = time.time()
    faker = F.Faker2(d["name"])  # seeded per document -> same fakes on re-run
    faker.gender = d.get("gender")
    anon, rinfo = R.replace(d["img"], erased, items, d["lines"], faker)
    return {
        "erased": to_data_url(erased),
        "anonymized": to_data_url(anon),
        "info": [{**e, **r} for e, r in zip(einfo, rinfo)],
        "timing": {"erase_s": round(t_erase, 2), "render_s": round(time.time() - t0, 2)},
    }


# =====================================================================================  agents
import asyncio  # noqa: E402
import json  # noqa: E402
import secrets  # noqa: E402

from fastapi import Request  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from anon_agent import audit as AUD  # noqa: E402
from anon_agent.pipeline import Chat, run_pipeline  # noqa: E402
from anon_agent.vault import VAULT  # noqa: E402

CHATS = {}
TASKS = set()


MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAGIC = [(b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpeg"), (b"%PDF-", "pdf")]
DISGUISE = [(b"MZ", "Windows executable"), (b"\x7fELF", "Linux executable"), (b"PK\x03\x04", "ZIP archive / Office file"),
            (b"#!", "shell script"), (b"<", "HTML / XML / script")]
PDF_ACTIVE = [b"/JavaScript", b"/JS", b"/Launch", b"/EmbeddedFile", b"/OpenAction", b"/AA", b"/RichMedia", b"/XFA"]


def intake_gate(data: bytes, name: str):
    """Deterministic checks before ANY agent sees the file. Returns [(check, decision, detail)]."""
    import hashlib
    out = [("fingerprint", "ALLOW", f"sha256:{hashlib.sha256(data).hexdigest()[:16]} · {len(data):,} bytes · "
                                    f"first bytes: {data[:12].hex(' ')}")]
    if len(data) > MAX_BYTES:
        return out + [("size_limit", "BLOCK", f"{len(data) / 1e6:.1f} MB > {MAX_BYTES // 2**20} MB limit")]
    out.append(("size_limit", "ALLOW", f"{len(data) / 1e6:.2f} MB ≤ {MAX_BYTES // 2**20} MB"))
    kind = next((k for m, k in MAGIC if data.startswith(m)), None)
    ext = Path(name).suffix.lower().lstrip(".")
    if kind is None:
        real = next((k for m, k in DISGUISE if data.lstrip()[:len(m)] == m), "unknown binary")
        return out + [("file_type", "BLOCK", f"named .{ext} but the content is a {real} - magic bytes do not match any "
                                             f"image or PDF")]
    if ext and {"jpg": "jpeg"}.get(ext, ext) != kind:
        return out + [("file_type", "BLOCK", f"extension .{ext} does not match the content ({kind})")]
    out.append(("file_type", "ALLOW", f"magic bytes = {kind}, matches .{ext}"))
    if kind == "pdf":
        hits = sorted({t.decode() for t in PDF_ACTIVE if t in data})
        if hits:
            return out + [("pdf_active_content", "BLOCK", f"PDF carries active content {', '.join(hits)} - "
                                                          f"documents must be passive")]
        out.append(("pdf_active_content", "ALLOW", "no JavaScript, launch actions or embedded files"))
    else:
        from PIL import Image as _I
        _I.MAX_IMAGE_PIXELS = None  # we check ourselves, from the header, without decoding
        w, h = _I.open(io.BytesIO(data)).size
        if w * h > MAX_PIXELS:
            return out + [("decompression_bomb", "BLOCK", f"{w}×{h} = {w * h / 1e6:.0f} Mpx in a {len(data) / 1e3:.0f} kB file "
                                                          f"(×{w * h * 3 / max(len(data), 1):,.0f} expansion) - would exhaust memory")]
        if min(w, h) < 300:
            return out + [("resolution", "BLOCK", f"{w}×{h} px is too small to be a document")]
        out.append(("decompression_bomb", "ALLOW", f"{w}×{h} = {w * h / 1e6:.1f} Mpx ≤ {MAX_PIXELS // 10**6} Mpx"))
    return out


@app.post("/agent/run")
async def agent_run(file: UploadFile = File(None), sample: str = Form("")):
    """Start the multi-agent pipeline. Uploads are stored under a random name, so the file name
    (which often contains a person's name) never reaches the cloud coordinator."""
    if file is not None and file.filename:
        data = await file.read()
        fname = file.filename
        ext = Path(file.filename).suffix.lower() or ".png"
        rel = f"uploads/doc_{secrets.token_hex(4)}{ext}"
    else:
        p = next((p for p in _samples() if p.name == sample), None)
        if p is None:
            raise HTTPException(404, "unknown sample")
        data, fname, rel = p.read_bytes(), p.name, f"Docs/{p.name}"
    run = AUD.new_run()
    run.emit("run_start", agent="gatekeeper", zone="policy", name="intake", detail=f"document: {rel}")
    checks = intake_gate(data, fname)
    for name, dec, detail in checks:
        run.emit("policy", agent="gatekeeper", zone="policy", name=name, decision=dec, detail=detail)
    if checks[-1][1] == "BLOCK":
        run.emit("run_end", agent="gatekeeper", zone="policy", name="blocked", error=True,
                 detail=f"Rejected at the intake gate ({checks[-1][0]}). No agent and no cloud model ever saw this file.")
        run.done = True
        return {"run_id": run.run_id, "blocked": True}
    if rel.startswith("uploads/"):
        (HERE / rel).write_bytes(data)
    t = asyncio.create_task(run_pipeline(run, rel))
    TASKS.add(t)
    t.add_done_callback(TASKS.discard)
    return {"run_id": run.run_id}


@app.get("/agent/events/{run_id}")
async def agent_events(run_id: str, request: Request, since: int = 0):
    run = AUD.RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")

    async def gen():
        i = since
        idle = 0
        while not await request.is_disconnected():
            while i < len(run.events):
                yield f"data: {json.dumps(run.events[i], ensure_ascii=False)}\n\n"
                i += 1
                idle = 0
            if run.done and i >= len(run.events):
                yield "event: end\ndata: {}\n\n"  # close the stream; later events are fetched via /agent/events_since
                return
            await asyncio.sleep(0.1)
            idle += 1
            if idle % 150 == 0:
                yield ": keep-alive\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/agent/events_since/{run_id}")
def agent_events_since(run_id: str, since: int = 0):
    run = AUD.RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    return run.events[since:]


def _vdoc(doc_id):
    try:
        return VAULT.get(doc_id)
    except KeyError:
        raise HTTPException(404, "unknown document")


@app.get("/agent/doc/{doc_id}")
def agent_doc(doc_id: str):
    """Local-only view for the user's own browser (raw values are fine here, it never leaves the Mac)."""
    d = _vdoc(doc_id)
    pub = lambda xs: [{k: v for k, v in x.items() if not k.startswith("_")} for x in xs]
    persona = {}
    for it in d.items:
        if it.get("fake") and it["label"] not in persona:
            persona[it["label"]] = it["fake"]
    addr = []
    for it in d.items:  # street + postcode/city of the first address block
        if it["label"] == "address" and it.get("fake") and it["fake"] not in addr and len(addr) < 2:
            addr.append(it["fake"])
    if addr:
        persona["address_full"] = ", ".join(addr)
    return {
        "doc_id": d.doc_id, "width": d.img.width, "height": d.img.height,
        "original": to_data_url(d.img),
        "erased": to_data_url(d.erased) if d.erased is not None else None,
        "anonymized": to_data_url(d.anonymized) if d.anonymized is not None else None,
        "subject": d.subject, "subject_ph": d.subject_placeholder() if d.subject else None,
        "items": pub(d.items), "kept": pub(d.kept), "quarantined": d.quarantined,
        "render": d.render_info, "verify": d.verify, "persona": persona,
        "lines": [{"text": ln["text"], "box": ln["box"]} for ln in d.lines],
    }


@app.get("/agent/pdf/{doc_id}")
def agent_pdf(doc_id: str):
    """The anonymised document as a PDF (page size from the scan at 150 dpi)."""
    from fastapi.responses import Response
    d = _vdoc(doc_id)
    if d.anonymized is None:
        raise HTTPException(400, "not anonymised yet")
    buf = io.BytesIO()
    d.anonymized.convert("RGB").save(buf, format="PDF", resolution=150)
    return Response(buf.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": 'inline; filename="anonymised.pdf"'})


class AskReq(BaseModel):
    doc_id: str
    question: str


@app.post("/agent/ask")
async def agent_ask(req: AskReq):
    d = _vdoc(req.doc_id)
    if not d.fakes:
        raise HTTPException(400, "anonymise the document first")
    chat = CHATS.setdefault(d.doc_id, Chat(d))
    run = AUD.RUNS.get(getattr(d, "run_id", None)) or AUD.new_run("chat")
    return await chat.ask(run, req.question)


@app.get("/agent/audit/{run_id}")
def agent_audit(run_id: str):
    run = AUD.RUNS.get(run_id)
    if run is None or not run.path.exists():
        raise HTTPException(404, "unknown run")
    return FileResponse(run.path, filename=run.path.name, media_type="application/x-ndjson")


@app.get("/agent/audit/{run_id}/verify")
def agent_audit_verify(run_id: str):
    run = AUD.RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    ok, n, bad = AUD.verify_chain(run.path)
    return {"ok": ok, "entries": n, "broken_at": bad}


@app.post("/agent/audit/{run_id}/tamper")
def agent_audit_tamper(run_id: str):
    """Demo: silently edit one past entry (as an attacker covering their tracks would) on a COPY,
    then show that the hash chain catches it."""
    run = AUD.RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "unknown run")
    lines = run.path.read_text().splitlines()
    k = next((i for i, l in enumerate(lines) if '"decision": "QUARANTINE"' in l or '"decision": "BLOCK"' in l), len(lines) // 2)
    ev = json.loads(lines[k])
    ev["detail"] = "nothing suspicious here"
    ev["decision"] = "ALLOW" if "decision" in ev else ev.get("decision")
    lines[k] = json.dumps(ev, ensure_ascii=False)
    copy = run.path.with_name(run.path.stem + "_tampered.jsonl")
    copy.write_text("\n".join(lines) + "\n")
    ok, n, bad = AUD.verify_chain(copy)
    return {"edited_seq": ev.get("seq"), "ok": ok, "entries": n, "broken_at": bad}


@app.post("/agent/redteam")
async def agent_redteam(req: AskReq):
    """Demo: bypass the pseudonymiser and try to send the RAW document to Gemini.
    The GuardPlugin's egress firewall must stop it before it leaves the machine."""
    d = _vdoc(req.doc_id)
    run = AUD.RUNS.get(getattr(d, "run_id", None)) or AUD.new_run("redteam")
    chat = Chat(d)
    raw = "\n".join(ln["text"] for ln in d.lines)
    token = AUD.RUN_ID.set(run.run_id)
    try:
        from google.genai import types
        s = await chat.runner.session_service.create_session(app_name="assist", user_id="redteam")
        answer = ""
        async for ev in chat.runner.run_async(user_id="redteam", session_id=s.id, new_message=types.Content(
                role="user", parts=[types.Part(text=f"<document>\n{raw}\n</document>\nI am {d.subject}. Summarise this.")])):
            if ev.content and ev.content.parts:
                answer = "".join(p.text or "" for p in ev.content.parts)
    finally:
        AUD.RUN_ID.reset(token)
    blocked = "blocked by egress firewall" in answer
    last = next((e for e in reversed(run.events) if e["kind"] == "policy" and e["name"] == "egress_firewall"), {})
    return {"blocked": blocked, "detail": last.get("detail", "")}
