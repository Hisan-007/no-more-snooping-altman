"""Entry points used by the web server and the CLI."""
from google.adk.runners import InMemoryRunner
from google.genai import types

from . import audit as A
from .agent import assistant, coordinator
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
        final = ""
        async for ev in runner.run_async(user_id="user", session_id=s.id, new_message=types.Content(
                role="user", parts=[types.Part(text=f"Anonymise the document at {path}")])):
            if ev.author == "coordinator" and _text(ev):
                final = _text(ev)
        run.emit("run_end", agent="coordinator", zone="cloud", name="report", detail=final)
        return final
    except Exception as e:  # surface in the UI instead of dying silently
        run.emit("run_end", agent="coordinator", zone="cloud", name="error", detail=f"{type(e).__name__}: {e}", error=True)
        raise
    finally:
        run.done = True
        A.RUN_ID.reset(token)


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
