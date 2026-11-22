"""Entry points used by the web server and the CLI."""
import json
import time

from google.adk.runners import InMemoryRunner
from google.genai import types

from . import audit as A
from . import tools as T
from .agent import assistant, coordinator
from .fallback import CloudUnavailable
from .plugins import GuardPlugin, TracePlugin
from .vault import VAULT


def _text(ev):
    if not ev.content or not ev.content.parts:
        return ""
    return "".join(p.text or "" for p in ev.content.parts if p.text and not getattr(p, "thought", False))


async def run_pipeline(run, path):
    """Let the coordinator anonymise the document at `path`. Streams into run.events."""
    token = A.RUN_ID.set(run.run_id)
    try:
        run.emit("run_start", agent="coordinator", zone="cloud", name="anonymise", detail=f"document: {path}")
        runner = InMemoryRunner(agent=coordinator, app_name="anon", plugins=[GuardPlugin(), TracePlugin()])
        s = await runner.session_service.create_session(app_name="anon", user_id="user")
        final, calls = "", []
        try:
            async for ev in runner.run_async(user_id="user", session_id=s.id, new_message=types.Content(
                    role="user", parts=[types.Part(text=f"Anonymise the document at {path}")])):
                calls += [(r.name, r.response) for r in ev.get_function_responses()]
                if ev.author == "coordinator" and _text(ev):
                    final = _text(ev)
        except CloudUnavailable as e:
            run.emit("policy", agent="coordinator", zone="policy", name="model_fallback", decision="FALLBACK",
                     detail=f"no cloud model available ({e}) → finishing the procedure on-device")
            final = await run_scripted(run, path, calls)
            run.emit("run_end", agent="pipeline", zone="local", name="report", detail=final)
            return final
        run.emit("run_end", agent="coordinator", zone="cloud", name="report", detail=final)
        return final
    except Exception as e:  # surface in the UI instead of dying silently
        run.emit("run_end", agent="coordinator", zone="cloud", name="error", detail=f"{type(e).__name__}: {e}", error=True)
        raise
    finally:
        run.done = True
        A.RUN_ID.reset(token)


class _Stop(Exception):
    pass


async def _call(run, name, **args):
    run.emit("tool_start", agent="pipeline", zone="tool", name=name, detail=json.dumps(args))
    t = time.time()
    res = await getattr(T, name)(**args)
    run.emit("tool_end", agent="pipeline", zone="tool", name=name, ms=round((time.time() - t) * 1000),
             detail=json.dumps(res, ensure_ascii=False, default=str)[:3000])
    if "error" in res:
        raise _Stop(res["error"])
    return res


async def run_scripted(run, path, calls):
    """The coordinator's procedure as plain code, for when no cloud model is reachable.
    Continues after the last tool the coordinator completed (`calls`), so no work is redone."""
    got = dict(calls)
    last = calls[-1][0] if calls else None
    try:
        for name, res in calls:
            if "error" in res:
                raise _Stop(res["error"])
        doc_id = got["load_document"]["doc_id"] if "load_document" in got else (await _call(run, "load_document", path=path))["doc_id"]
        a = got.get("analyse_document") or await _call(run, "analyse_document", doc_id=doc_id)
        redacts = sum(n == "redact_document" for n, _ in calls)
        v = got.get("verify_document", {})
        if "finalize" not in got:
            if not redacts:
                await _call(run, "redact_document", doc_id=doc_id)
                redacts = 1
            if last != "verify_document":
                v = await _call(run, "verify_document", doc_id=doc_id)
            while not v["clean"] and redacts <= 2:  # at most 2 retries, as for the coordinator
                await _call(run, "redact_document", doc_id=doc_id,
                            stronger_for=sorted(set(v["still_readable_item_ids"] + v["leaked_item_ids"])))
                redacts += 1
                v = await _call(run, "verify_document", doc_id=doc_id)
            await _call(run, "finalize", doc_id=doc_id)
    except _Stop as e:
        return f"Stopped: {e}"
    kinds = {}
    for it in a["to_replace"]:
        kinds[it["type"]] = kinds.get(it["type"], 0) + 1
    q = a["quarantined_lines"]
    return "\n".join([
        "- Finished on-device: no cloud model was available, so this report is generated, not written by the coordinator.",
        f"- Document of {a['subject']}: replaced " + (", ".join(f"{n} {k}" for k, n in kinds.items()) or "nothing") + ".",
        "- Kept (someone else's data): " + (", ".join(f"{n} {k}" for k, n in a["kept_someone_elses"].items()) or "nothing") + ".",
        f"- Security: {len(q)} quarantined line(s)" + (": " + "; ".join(f"L{x['line']} {x['reason']}" for x in q) if q else "") + ".",
        f"- Redaction passes: {redacts} ({redacts - 1} retr{'y' if redacts == 2 else 'ies'}).",
        f"- Verification: {'clean' if v.get('clean') else 'NOT clean'}; "
        f"{v.get('pixels_changed_outside_boxes', '?')} pixel(s) changed outside the boxes.",
    ])


class Chat:
    """One assistant conversation per document. The cloud only ever sees the pseudonymised text."""

    def __init__(self, doc):
        self.doc = doc
        self.runner = InMemoryRunner(agent=assistant, app_name="assist", plugins=[GuardPlugin(), TracePlugin()])
        self.session = None

    def doc_text(self):
        # the document as the cloud may see it: every subject value swapped for its fake
        text = "\n".join(ln["text"] for ln in self.doc.lines)
        q = {self.doc.lines[x["line"]]["text"] for x in self.doc.quarantined}
        text = "\n".join(l for l in text.splitlines() if l not in q)  # injected lines never reach the cloud
        return self.doc.pseudonymise(text)[0]

    async def ask(self, run, question):
        token = A.RUN_ID.set(run.run_id)
        try:
            sent, n = self.doc.pseudonymise(question)
            run.emit("policy", agent="assistant", zone="policy", name="pseudonymise", decision="REWRITE",
                     detail=f"{n} personal value(s) in the question swapped for stand-ins before leaving the Mac")
            if self.session is None:
                self.session = await self.runner.session_service.create_session(app_name="assist", user_id="user")
                sent = f"<document>\n{self.doc_text()}\n</document>\n\nQuestion: {sent}"
            answer = ""
            async for ev in self.runner.run_async(user_id="user", session_id=self.session.id, new_message=types.Content(
                    role="user", parts=[types.Part(text=sent)])):
                if ev.author == "assistant" and _text(ev):
                    answer = _text(ev)
            back, m = self.doc.reidentify(answer)
            run.emit("policy", agent="assistant", zone="policy", name="reidentify", decision="REWRITE",
                     detail=f"{m} stand-in(s) in the answer mapped back to your real data, on this Mac")
            return {"sent": VAULT.scrub_all(sent), "cloud_answer": answer, "answer": back, "swapped_out": n, "swapped_back": m}
        finally:
            A.RUN_ID.reset(token)
