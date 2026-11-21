"""Terminal demo:  .venv/bin/python cli.py Docs/SAMPLE_Arztbrief.png ["question for the assistant"]

Prints the live agent trace, the coordinator's report, and the audit-chain check.
"""
import asyncio
import sys
import warnings

warnings.filterwarnings("ignore")

from anon_agent import audit as A  # noqa: E402
from anon_agent.pipeline import Chat, run_pipeline  # noqa: E402
from anon_agent.vault import VAULT  # noqa: E402

C = {"cloud": "\033[34m", "local": "\033[32m", "tool": "\033[33m", "policy": "\033[35m", "": "\033[0m"}
R = "\033[0m"


def show(ev):
    k = ev["kind"]
    if k in ("model_end", "tool_end", "agent_end", "state"):
        if k == "tool_end":
            print(f"{C['tool']}{ev['ts']:6.1f}s  [tool] {ev['name']:18} ← {ev['detail'][:150]}{R}")
        elif k == "model_end" and ev["detail"]:
            print(f"{C.get(ev['zone'], '')}{ev['ts']:6.1f}s  {ev['agent']:12} ⟵ {ev['detail'][:150].replace(chr(10), ' ')}{R}")
        return
    if k == "policy":
        print(f"{C['policy']}{ev['ts']:6.1f}s  policy {ev.get('decision', ''):10} {ev['name']}: {ev['detail'][:140]}{R}")
    elif k == "model_start":
        print(f"{C.get(ev['zone'], '')}{ev['ts']:6.1f}s  {ev['agent']:12} → {ev['name']} ({ev['zone']}, {ev.get('chars')} chars){R}")
    elif k == "tool_start":
        print(f"{C['tool']}{ev['ts']:6.1f}s  [tool] {ev['name']:18} ({ev['agent']}) {ev['detail'][:100]}{R}")
    elif k in ("run_start", "run_end"):
        print(f"\n{ev['ts']:6.1f}s  == {k} {ev['name']} ==\n{ev['detail']}\n")


async def main(path, question=None):
    run = A.new_run()
    seen = 0

    async def tail():
        nonlocal seen
        while not run.done or seen < len(run.events):
            while seen < len(run.events):
                show(run.events[seen])
                seen += 1
            await asyncio.sleep(0.1)

    t = asyncio.create_task(tail())
    await run_pipeline(run, path)
    await t
    if question:
        doc = list(VAULT.docs.values())[-1]
        res = await Chat(doc).ask(run, question)
        for ev in run.events[seen:]:
            show(ev)
        print("\nSENT TO GEMINI:\n" + res["sent"][-400:] + "\n\nGEMINI SAID:\n" + res["cloud_answer"] + "\n\nYOU SEE:\n" + res["answer"])
    ok, n, bad = A.verify_chain(run.path)
    print(f"\naudit log {run.path.name}: {n} entries, hash chain {'OK' if ok else 'BROKEN at ' + str(bad)}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
