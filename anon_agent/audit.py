"""Trace bus + tamper-evident audit log.

Every agent / model / tool / policy event becomes one JSON line in audit/<run>.jsonl.
Each line carries the hash of the previous line, so editing or deleting any entry
breaks the chain (`verify_chain`). The same events are streamed live to the UI.
Text fields are scrubbed through the vault before they are written: no raw values.
"""
import contextvars
import hashlib
import json
import time
import uuid
from pathlib import Path

from .vault import VAULT

AUDIT_DIR = Path(__file__).resolve().parent.parent / "audit"
AUDIT_DIR.mkdir(exist_ok=True)

RUN_ID = contextvars.ContextVar("run_id", default=None)

GENESIS = "0" * 64


class Run:
    def __init__(self, run_id):
        self.run_id = run_id
        self.t0 = time.time()
        self.events = []
        self.prev = GENESIS
        self.path = AUDIT_DIR / f"{run_id}.jsonl"
        self.done = False

    def emit(self, kind, agent="", zone="", name="", detail="", **extra):
        ev = {"seq": len(self.events), "ts": round(time.time() - self.t0, 3), "run": self.run_id, "kind": kind,
              "agent": agent, "zone": zone, "name": name, "detail": VAULT.scrub_all(str(detail))[:2000]}
        for k, v in extra.items():
            ev[k] = VAULT.scrub_all(v) if isinstance(v, str) else v
        ev["prev"] = self.prev
        ev["hash"] = hashlib.sha256((self.prev + json.dumps(ev, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
        self.prev = ev["hash"]
        self.events.append(ev)
        with self.path.open("a") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return ev


RUNS = {}


def new_run(prefix="run"):
    rid = f"{prefix}_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}"
    RUNS[rid] = Run(rid)
    return RUNS[rid]


def current(fallback_id=None):
    rid = RUN_ID.get() or fallback_id
    if rid is None:
        return None
    if rid not in RUNS:
        RUNS[rid] = Run(rid)  # e.g. runs started from `adk web`
    return RUNS[rid]


def emit(kind, fallback_id=None, **kw):
    r = current(fallback_id)
    return r.emit(kind, **kw) if r else None


def new_span():
    return uuid.uuid4().hex[:8]


def verify_chain(path):
    """Recompute every hash. Returns (ok, n_entries, first_bad_seq)."""
    prev = GENESIS
    n = 0
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        ev = json.loads(line)
        claimed = ev.pop("hash")
        if ev["prev"] != prev:
            return False, n, ev.get("seq")
        if hashlib.sha256((prev + json.dumps(ev, sort_keys=True, ensure_ascii=False)).encode()).hexdigest() != claimed:
            return False, n, ev.get("seq")
        prev = claimed
    return True, n, None


if __name__ == "__main__":
    import sys
    ok, n, bad = verify_chain(sys.argv[1])
    print(f"{'OK' if ok else 'TAMPERED'} - {n} entries" + ("" if ok else f", chain breaks at seq {bad}"))
    sys.exit(0 if ok else 1)
