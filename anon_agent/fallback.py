"""Model failover for the cloud agents.

A 503 / 429 from Gemini comes back fast, so instead of retrying with backoff we switch to the
next model at once and skip the failed one for a while (one run makes several model calls).
An overloaded model can also just hang, so a cloud call that takes longer than MODEL_TIMEOUT
counts as a failure too. So does a missing or rejected API key: the run continues on-device
instead of failing (the trace says why).
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
    """Every model in the chain is overloaded, hanging, cooling down or lacks a valid API key."""


def _has_credentials():
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"))


def _key_rejected(e):
    # an invalid key is a 400 "API key not valid"; a key without access to the model is 401 / 403
    return e.code in (401, 403) or (e.code == 400 and "api key" in str(e.message).lower())


async def _collect(m, llm_request, stream):
    return [r async for r in m.generate_content_async(llm_request, stream)]


class FallbackLlm(BaseLlm):
    chain: List[BaseLlm]
    agent: str = ""  # only for the trace

    def __init__(self, chain, agent=""):
        # e.g. CLOUD_MODEL=gemini-flash-lite-latest equals the default fallback: try it once, not twice
        chain = [m for i, m in enumerate(chain) if m.model not in {c.model for c in chain[:i]}]
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
            if "ollama" not in m.model and not _has_credentials():  # skipped before anything is sent
                failed.append(f"{m.model}: no API key")
                continue
            try:
                # collected before yielding: the timeout must not cover what the caller does with
                # the answer (e.g. run a tool), and a failed model must not have yielded anything
                responses = await asyncio.wait_for(_collect(m, llm_request, stream),
                                                   None if "ollama" in m.model else TIMEOUT)
            except asyncio.TimeoutError:
                failed.append(f"{m.model}: no answer after {TIMEOUT:g}s")
            except errors.APIError as e:
                if _key_rejected(e):
                    failed.append(f"{m.model}: API key rejected ({e.code} {e.status})")
                elif e.code in OVERLOADED:
                    failed.append(f"{m.model}: {e.code} {e.status}")
                else:
                    raise
            else:
                for r in responses:
                    r.custom_metadata = {**(r.custom_metadata or {}), "served_by": m.model}
                    yield r
                return
            _down[m.model] = time.time() + COOLDOWN
        raise CloudUnavailable("; ".join(failed + [f"{s}: cooling down" for s in skipped]))
