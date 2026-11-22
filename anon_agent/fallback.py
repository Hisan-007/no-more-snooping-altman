"""Model failover for the cloud agents.

A 503 / 429 from Gemini comes back fast, so instead of retrying with backoff we switch to the
next model at once and skip the failed one for a while (one run makes several model calls).
An overloaded model can also just hang, so a cloud call that takes longer than MODEL_TIMEOUT
counts as a failure too.
The GuardPlugin checks the request before this runs; every later model in the chain is either
cloud (same, already checked request) or local, so failover never weakens the egress firewall.
"""
import asyncio
import os
import time
from typing import AsyncGenerator, List

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import errors

from . import audit as A

OVERLOADED = {429, 500, 503, 504}
COOLDOWN = float(os.environ.get("MODEL_COOLDOWN", "60"))  # seconds a failed model is skipped
TIMEOUT = float(os.environ.get("MODEL_TIMEOUT", "30"))  # seconds per cloud call; local models may take their time
_down = {}  # model name -> time it may be tried again (shared by all agents)


class CloudUnavailable(Exception):
    """Every model in the chain is overloaded or cooling down."""


async def _collect(m, llm_request, stream):
    return [r async for r in m.generate_content_async(llm_request, stream)]


class FallbackLlm(BaseLlm):
    chain: List[BaseLlm]
    agent: str = ""  # only for the trace

    def __init__(self, chain, agent=""):
        super().__init__(model=chain[0].model, chain=chain, agent=agent)

    @property
    def capabilities(self):
        return self.chain[0].capabilities

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        live = [m for m in self.chain if _down.get(m.model, 0) <= time.time()]
        skipped = [m.model for m in self.chain if m not in live]
        failed = []
        for i, m in enumerate(live):
            llm_request.model = m.model
            if i or skipped:
                A.emit("policy", agent=self.agent, zone="policy", name="model_fallback", decision="FALLBACK",
                       detail=f"{', '.join(failed + [f'{s}: cooling down' for s in skipped])} → {m.model}")
            try:
                # collected before yielding: the timeout must not cover what the caller does with
                # the answer (e.g. run a tool), and a failed model must not have yielded anything
                responses = await asyncio.wait_for(_collect(m, llm_request, stream),
                                                   None if "ollama" in m.model else TIMEOUT)
            except asyncio.TimeoutError:
                failed.append(f"{m.model}: no answer after {TIMEOUT:g}s")
            except errors.APIError as e:
                if e.code not in OVERLOADED:
                    raise
                failed.append(f"{m.model}: {e.code} {e.status}")
            else:
                for r in responses:
                    r.custom_metadata = {**(r.custom_metadata or {}), "served_by": m.model}
                    yield r
                return
            _down[m.model] = time.time() + COOLDOWN
        raise CloudUnavailable("; ".join(failed + [f"{s}: cooling down" for s in skipped]))
