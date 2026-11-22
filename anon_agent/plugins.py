"""Two ADK plugins attached to every Runner:

TracePlugin  - turns every agent / model / tool callback into a trace span (audit log + live UI).
GuardPlugin  - the policy layer, enforced in code (not by asking a model nicely):
    * tool allow-list per agent (least privilege)
    * egress firewall: a request to a CLOUD model that contains any raw value from the
      vault is blocked before it leaves the machine
    * local "quarantined" agents get no tools at all
"""
import json
import time
from typing import Any, Optional

from google.adk.models import LlmResponse
from google.adk.plugins import BasePlugin
from google.genai import types

from . import audit as A
from .fallback import FALLBACKS
from .vault import VAULT

ZONE = {"coordinator": "cloud", "assistant": "cloud", "naive_agent": "cloud",
        "analyst": "local", "sentinel": "local", "verifier": "local"}

# least privilege: which agent may call which tool
ALLOW = {
    "coordinator": {"load_document", "analyse_document", "redact_document", "verify_document", "finalize"},
    "assistant": set(),
    "analyst": set(), "sentinel": set(), "verifier": set(),
    "naive_agent": {"load_document", "read_document_text", "redact_document", "finalize"},
}


def zone_of(agent_name, model=""):
    # the model decides when known: a cloud agent that fell back to local Gemma runs locally
    if "ollama" in str(model):
        return "local"
    return ZONE.get(agent_name, "cloud")


def _request_text(req) -> str:
    out = []
    if req.config and req.config.system_instruction:
        si = req.config.system_instruction
        out.append(si if isinstance(si, str) else " ".join(p.text or "" for p in getattr(si, "parts", []) or []))
    for c in req.contents or []:
        for p in c.parts or []:
            if p.text:
                out.append(f"[{c.role}] {p.text}")
            elif p.function_call:
                out.append(f"[{c.role} → tool] {p.function_call.name}({json.dumps(p.function_call.args, ensure_ascii=False)})")
            elif p.function_response:
                out.append(f"[tool → {c.role}] {p.function_response.name}: {json.dumps(p.function_response.response, ensure_ascii=False, default=str)}")
            elif p.inline_data:
                out.append(f"[{c.role}] <{p.inline_data.mime_type}, {len(p.inline_data.data or b'')} bytes>")
    return "\n".join(out)


def _response_summary(resp) -> str:
    if not resp.content or not resp.content.parts:
        return resp.error_message or ""
    out = []
    for p in resp.content.parts:
        if p.function_call:
            out.append(f"→ {p.function_call.name}({json.dumps(p.function_call.args, ensure_ascii=False)})")
        elif p.text and not getattr(p, "thought", False):
            out.append(p.text.strip())
    return "\n".join(out)


class TracePlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="trace")
        self.open = {}

    def _rid(self, ctx):
        try:
            return ctx.session.id
        except Exception:
            return None

    async def before_agent_callback(self, *, agent, callback_context):
        sid = A.new_span()
        self.open[("agent", callback_context.invocation_id, agent.name)] = (sid, time.time())
        A.emit("agent_start", self._rid(callback_context), agent=agent.name, zone=zone_of(agent.name),
               name=agent.name, span=sid, detail=(agent.description or "")[:200])

    async def after_agent_callback(self, *, agent, callback_context):
        sid, t = self.open.pop(("agent", callback_context.invocation_id, agent.name), (None, time.time()))
        A.emit("agent_end", self._rid(callback_context), agent=agent.name, zone=zone_of(agent.name),
               name=agent.name, span=sid, ms=round((time.time() - t) * 1000))

    async def before_model_callback(self, *, callback_context, llm_request):
        name = callback_context.agent_name
        model = llm_request.model
        if name in FALLBACKS:  # label the call with the model it really goes to first, not the configured one
            model = FALLBACKS[name].first_choice()
        zone = zone_of(name, model) if model else "none"  # none: no model available, nothing is sent
        sid = A.new_span()
        self.open[("model", callback_context.invocation_id, name)] = (sid, time.time())
        text = _request_text(llm_request)
        # cloud payloads are shown verbatim (they must be clean - the GuardPlugin enforces it);
        # local payloads hold raw document text, so only their size is logged
        payload = text if zone == "cloud" else f"<{len(text)} chars of document text - stays on this Mac>"
        A.emit("model_start", self._rid(callback_context), agent=name, zone=zone, name=str(model or "no model available"),
               span=sid, payload=payload[:20000], chars=len(text))

    async def after_model_callback(self, *, callback_context, llm_response):
        name = callback_context.agent_name
        if llm_response.partial:
            return None
        sid, t = self.open.pop(("model", callback_context.invocation_id, name), (None, time.time()))
        u = llm_response.usage_metadata
        served = (llm_response.custom_metadata or {}).get("served_by", "")  # set by FallbackLlm
        A.emit("model_end", self._rid(callback_context), agent=name, zone=zone_of(name, served), span=sid,
               ms=round((time.time() - t) * 1000), detail=_response_summary(llm_response)[:1500],
               tokens={"in": getattr(u, "prompt_token_count", None), "out": getattr(u, "candidates_token_count", None)} if u else None)
        return None

    async def on_model_error_callback(self, *, callback_context, llm_request, error):
        name = callback_context.agent_name
        sid, t = self.open.pop(("model", callback_context.invocation_id, name), (None, time.time()))
        A.emit("model_end", self._rid(callback_context), agent=name, zone=zone_of(name, llm_request.model), span=sid,
               ms=round((time.time() - t) * 1000), detail=f"error: {error}", error=True)
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        sid = A.new_span()
        self.open[("tool", tool_context.function_call_id)] = (sid, time.time())
        A.emit("tool_start", self._rid(tool_context), agent=tool_context.agent_name, zone="tool", name=tool.name,
               span=sid, detail=json.dumps(tool_args, ensure_ascii=False, default=str))

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        sid, t = self.open.pop(("tool", tool_context.function_call_id), (None, time.time()))
        A.emit("tool_end", self._rid(tool_context), agent=tool_context.agent_name, zone="tool", name=tool.name,
               span=sid, ms=round((time.time() - t) * 1000),
               detail=json.dumps(result, ensure_ascii=False, default=str)[:3000])

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        sid, t = self.open.pop(("tool", tool_context.function_call_id), (None, time.time()))
        A.emit("tool_end", self._rid(tool_context), agent=tool_context.agent_name, zone="tool", name=tool.name,
               span=sid, ms=round((time.time() - t) * 1000), detail=f"error: {error}", error=True)
        return {"error": str(error)}


class GuardPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="guard")

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> Optional[dict]:
        agent = tool_context.agent_name
        allowed = ALLOW.get(agent, set())
        if tool.name not in allowed:
            A.emit("policy", tool_context.session.id, agent=agent, zone="policy", name="tool_allowlist",
                   decision="BLOCK", detail=f"{agent} may not call {tool.name} (allowed: {sorted(allowed) or 'none'})")
            return {"error": f"policy: {agent} is not allowed to call {tool.name}"}
        A.emit("policy", tool_context.session.id, agent=agent, zone="policy", name="tool_allowlist",
               decision="ALLOW", detail=f"{agent} → {tool.name}")
        return None

    async def before_model_callback(self, *, callback_context, llm_request) -> Optional[LlmResponse]:
        agent = callback_context.agent_name
        if zone_of(agent, llm_request.model) != "cloud":
            return None
        leaks = VAULT.leaks_all(_request_text(llm_request))
        if leaks:
            A.emit("policy", callback_context.session.id, agent=agent, zone="policy", name="egress_firewall",
                   decision="BLOCK", detail=f"request to the cloud contained {len(leaks)} raw value(s): "
                                            + ", ".join(VAULT.scrub_all(v) for v in leaks[:5]))
            return LlmResponse(content=types.Content(role="model", parts=[types.Part(
                text="[blocked by egress firewall: the request contained personal data]")]))
        A.emit("policy", callback_context.session.id, agent=agent, zone="policy", name="egress_firewall",
               decision="ALLOW", detail="no raw personal data in the outgoing request")
        return None
