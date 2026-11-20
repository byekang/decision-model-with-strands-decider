"""Decision backends: Haiku 4.5 via Bedrock (Arm A) and the strands-decider
/v1/systemone server (Arm B).

Both receive the same question text and label set. Invalid outputs fall back to a
per-decision default so the graph always proceeds; invalid rate is a reported metric.

The /v1/systemone wire format is a stable contract: any server that speaks it
(strands-decider, or other decision-model servers) can be plugged into Arm B by
changing only host/port/model — the graph code never changes.
"""

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass

from .prompts import HAIKU_DECISION_SYSTEM

FALLBACKS = {"d1": "in_scope", "d2": "general_faq", "d4": "clarify"}  # d3 -> first tool

DECIDER_MODEL = "StrandsAgents/strands-decider-E4B-gemma4-v1-2610"


@dataclass
class DecisionResult:
    label: str
    confidence: float | None
    latency_ms: float
    valid: bool
    usage: dict
    cost_usd: float
    raw: str


def _fallback(decision_id: str, choices: list[str]) -> str:
    return FALLBACKS.get(decision_id, choices[0])


class HaikuDecision:
    name = "haiku-4-5"

    def __init__(self, client, model_id: str, price: dict):
        self.client = client
        self.model_id = model_id
        self.price = price

    def decide(self, *, state_text: str, question: str, choices: list[str],
               decision_id: str) -> DecisionResult:
        labels = "\n".join(f"- {c}" for c in choices)
        user = f"{question}\n\nInput:\n{state_text}\n\nLabels:\n{labels}"
        t0 = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model_id, max_tokens=64,
            system=HAIKU_DECISION_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        ms = (time.perf_counter() - t0) * 1000.0
        text = resp.content[0].text if resp.content else ""
        label, valid = None, False
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            label = json.loads(m.group(0))["label"]
            valid = label in choices
        except Exception:
            pass
        if not valid:
            label = _fallback(decision_id, choices)
        usage = {"input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens}
        cost = (usage["input_tokens"] * self.price["input_per_mtok"]
                + usage["output_tokens"] * self.price["output_per_mtok"]) / 1e6
        return DecisionResult(label=label, confidence=None, latency_ms=round(ms, 2),
                              valid=valid, usage=usage, cost_usd=round(cost, 8),
                              raw=text[:200])


class SystemOneDecision:
    """Typed decision via a /v1/systemone server (default: the local
    strands-decider E4B server launched by setup/onstart.sh on :8004)."""

    name = "strands-decider-e4b"

    def __init__(self, model: str = DECIDER_MODEL, port: int | None = None,
                 host: str | None = None):
        self.model = model
        host = host or os.environ.get("STRANDS_HOST", "localhost")
        port = port or int(os.environ.get("STRANDS_PORT", "8004"))
        self.url = f"http://{host}:{port}/v1/systemone"

    def decide(self, *, state_text: str, question: str, choices: list[str],
               decision_id: str) -> DecisionResult:
        body = json.dumps({
            "state": state_text,
            "model": self.model,
            "questions": {decision_id: {
                "type": "choice",
                "instructions": question,
                "criteria": {c: None for c in choices},
            }},
        }).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        resp = json.load(urllib.request.urlopen(req, timeout=120))
        ms = (time.perf_counter() - t0) * 1000.0
        ans = resp["answers"][decision_id]
        label, conf = ans["choice"], ans["confidence"]
        valid = label in choices
        if not valid:
            label = _fallback(decision_id, choices)
        return DecisionResult(label=label, confidence=conf, latency_ms=round(ms, 2),
                              valid=valid, usage=resp.get("usage", {}), cost_usd=0.0,
                              raw=json.dumps(ans)[:200])
