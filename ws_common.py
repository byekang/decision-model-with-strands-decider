"""Workshop 공용 helper. / Shared workshop helpers.

- make_clients()/make_graphs(): Bedrock client와 두 arm(Haiku / strands-decider)의
  v2 graph 생성
- run_verbose(): graph.stream(stream_mode="updates")로 노드 실행을 실시간 수신,
  "STEP n" 형식으로 각 노드의 결정(label/confidence/latency)·생성 텍스트·tool 결과를
  출력하고 수행시간 집계를 반환
- compare(): 두 arm 실행 결과의 수행시간 비교표(markdown) 출력

언어: DMWS_LANG 환경변수 ('ko' 기본 | 'en') — import 전에 설정해야 합니다.
Language: set env DMWS_LANG ('ko' default | 'en') BEFORE importing this module;
it localizes step output, comparison tables and graph captions.
"""

import os
import re
import sys
import time
from pathlib import Path

# 프로젝트 루트: DMWS_ROOT 환경변수 > agent/ 패키지가 있는 가장 가까운 상위 폴더
# (repo 루트에 ws_common이 있는 workshop 배포 레이아웃과, workshop/ 하위에 있는
#  개발 레이아웃을 모두 자동 지원)
def _find_root():
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "agent").is_dir():
            return p
    return here.parents[1]

ROOT = Path(os.environ.get("DMWS_ROOT") or _find_root())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# decision 서버: strands-decider E4B가 notebook instance의 GPU에서 localhost로
# 서빙됩니다 (setup/onstart.sh). 다른 호스트/포트로 옮기면 env로 지정하세요.
STRANDS_HOST = os.environ.get("STRANDS_HOST", "localhost")
STRANDS_PORT = int(os.environ.get("STRANDS_PORT", "8004"))

# 출력/그래프 캡션 언어 ('ko' | 'en')
LANG = os.environ.get("DMWS_LANG", "ko")


# graphviz용 폰트: 시스템에서 한글 렌더링 가능한 폰트를 동적으로 선택
# (SageMaker notebook 등 Noto CJK가 없는 환경에서 자모 분리 렌더링 방지)
def _pick_font():
    import subprocess
    try:
        fam = subprocess.run(["fc-match", "-f", "%{family}", ":lang=ko"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        if fam:
            return fam.split(",")[0]
    except Exception:
        pass
    return "Noto Sans CJK KR"

FONT = _pick_font()

if LANG not in ("ko", "en"):
    LANG = "ko"

from common import BEDROCK_MODEL_IDS, PRICING                      # noqa: E402
from agent.bedrock_llm import BedrockLLM                           # noqa: E402
from agent.decisions import HaikuDecision, SystemOneDecision       # noqa: E402
from agent.graph_v2 import build_graph_v2                          # noqa: E402

DECIDER_MODEL = "StrandsAgents/strands-decider-E4B-gemma4-v1-2610"
NO_INFO = "지금 알려드릴 수 있는 정보는 그게 전부예요."
NO_INFO_EN = "That's all the information I have right now."

# scripts/94에서 검증된 상담 시나리오 3종 (고객 발화는 한국어 능력 시연을 위해 항상 한국어;
# *_en 필드는 EN 노트북 표시용 번역)
SCENARIOS = {
    1: {
        "title": "모호한 문의 -> clarification loop",
        "title_en": "Vague inquiry -> clarification loop",
        "question": "카드가 안 긁히는데 어떻게 해야 하나요?",
        "question_en": '"My card is not swiping. What should I do?"',
        "script": {"vague": True,
                   "clarify_reply": ("어제부터 오프라인 매장 단말기에서만 승인이 거절돼요. "
                                     "관련 정보: Visa 체크카드(끝자리 7712), 마지막 시도 "
                                     "2026-10-27 TechMart 강남점, 금액 45,000원입니다."),
                   "no_info_reply": NO_INFO,
                   "tool_outcomes": ["success"], "risk_flag": False, "kyc_result": None},
    },
    2: {
        "title": "시스템 오류 -> tool retry loop",
        "title_en": "System error -> tool retry loop",
        "question": ("어제 오후에 보낸 해외송금이 아직 수취인에게 도착하지 않았어요. "
                     "송금 번호는 TR-20261026-8841, 금액은 USD 2,000입니다. 확인해 주세요."),
        "question_en": ('"The overseas transfer I sent yesterday has not reached the '
                        'recipient. Reference TR-20261026-8841, USD 2,000. Please check."'),
        "script": {"vague": False, "clarify_reply": None, "no_info_reply": NO_INFO,
                   "tool_outcomes": ["tool_error", "success"],
                   "risk_flag": False, "kyc_result": None},
    },
    3: {
        "title": "이상거래 모니터링 flag -> KYC sub-flow",
        "title_en": "Fraud-monitoring flag -> KYC sub-flow",
        "question": ("카드 끝자리 7712로 해외 결제(USD 1,284, LUXE GADGETS) 승인이 "
                     "됐다는 알림을 받았는데, 이 거래를 확인하고 싶어요."),
        "question_en": ('"I got an approval alert for an overseas charge (USD 1,284, '
                        'LUXE GADGETS) on my card ending 7712. I would like to verify '
                        'this transaction."'),
        "script": {"vague": False, "clarify_reply": None, "no_info_reply": NO_INFO,
                   "tool_outcomes": ["success"], "risk_flag": True,
                   "kyc_result": "verified"},
    },
}

# EN 워크샵용 영어 고객 발화 (KR과 동일한 시나리오 구조, script 분기 동일)
SCENARIOS_EN = {
    1: {
        "title": "Vague inquiry -> clarification loop",
        "question": "My card doesn't swipe. What should I do?",
        "script": {"vague": True,
                   "clarify_reply": ("Since yesterday the card won't go through at "
                                     "in-store terminals only. Details: Visa debit "
                                     "card ending 7712, last attempt on 2026-10-27 at "
                                     "TechMart Gangnam, amount USD 34.60."),
                   "no_info_reply": NO_INFO_EN,
                   "tool_outcomes": ["success"], "risk_flag": False, "kyc_result": None},
    },
    2: {
        "title": "System error -> tool retry loop",
        "question": ("The overseas transfer I sent yesterday afternoon has not reached "
                     "the recipient yet. Reference number TR-20261026-8841, amount "
                     "USD 2,000. Please check what happened."),
        "script": {"vague": False, "clarify_reply": None, "no_info_reply": NO_INFO_EN,
                   "tool_outcomes": ["tool_error", "success"],
                   "risk_flag": False, "kyc_result": None},
    },
    3: {
        "title": "Fraud-monitoring flag -> KYC sub-flow",
        "question": ("I received an approval alert for an overseas charge "
                     "(USD 1,284, LUXE GADGETS) on my card ending 7712, and I would "
                     "like to verify this transaction."),
        "script": {"vague": False, "clarify_reply": None, "no_info_reply": NO_INFO_EN,
                   "tool_outcomes": ["success"], "risk_flag": True,
                   "kyc_result": "verified"},
    },
}


def scenario(n, lang=None):
    """언어에 맞는 시나리오 반환 (lang 기본: DMWS_LANG). EN은 영어 고객 발화 사용."""
    return (SCENARIOS_EN if (lang or LANG) == "en" else SCENARIOS)[n]

_NODE_DESC = {
    "ko": {
        "guardrail_check": "D1 접수 심사: 정상 문의 / 범위 밖 / compliance 위반",
        "intent_router": "D2 업무 분류: 12개 전문 상담 category 중 선택",
        "clarify_gate": "D5 정보 충분성 판정: 바로 처리 가능한가, 되물어야 하는가",
        "ask_clarify": "되묻기: Sonnet이 질문 생성 -> 고객 답변 수신",
        "tool_step": "D3 tool 선택 -> tool 실행 -> D6 결과 검토(성공/재시도/포기)",
        "give_up": "tool 처리 포기 -> 상담사 이관 경로",
        "summarize": "Sonnet이 tool 결과를 고객 응대용 사실 요약으로 정리",
        "risk_check": "D7 리스크 점검: fraud/AML 모니터링 flag 유무",
        "kyc_verify": "KYC 재검증 tool 실행 -> D8 통과 여부 판정",
        "escalation_gate": "D4 처리 방침: 종결 / 추가 질문 / 상담사 이관",
        "respond": "Sonnet이 최종 고객 응답 작성",
    },
    "en": {
        "guardrail_check": "D1 intake screening: in-scope / out-of-scope / compliance",
        "intent_router": "D2 routing: pick one of 12 specialist teams",
        "clarify_gate": "D5 sufficiency check: can we act now, or must we ask back?",
        "ask_clarify": "Clarify: Sonnet writes a question -> customer reply received",
        "tool_step": "D3 tool choice -> run tool -> D6 result review (done/retry/give up)",
        "give_up": "Give up automated handling -> human handoff path",
        "summarize": "Sonnet condenses the tool result into customer-facing facts",
        "risk_check": "D7 risk check: fraud/AML monitoring flag present?",
        "kyc_verify": "Run KYC re-verification tool -> D8 pass/fail decision",
        "escalation_gate": "D4 final action: resolve / clarify / human handoff",
        "respond": "Sonnet writes the final customer reply",
    },
}
NODE_DESC = _NODE_DESC[LANG]

# run_verbose/compare/compare_image 출력 문자열
_UI = {
    "ko": {
        "customer": "고객", "agent": "상담 agent",
        "decide": "판정", "gen": "생성(Sonnet 4.5)", "gen_prefix": "생성",
        "tool": "tool 실행",
        "auto": "tool 자동 선택(남은 후보 1개)",
        "summary": "요약(첫 줄)", "final": "--- 최종 응답 ---",
        "totals": ">>> [{arm}] 결정 {n}회 합계 {dec:,.0f}ms | 생성 {llm:,.0f}ms | e2e {e2e:,.0f}ms",
        "h_metric": "지표", "h_ratio": "배수", "h_ndec": "결정 횟수",
        "h_decsum": "**결정 latency 합계**", "h_llm": "생성(LLM) latency",
        "speedup": "**결정 계층 속도 차이: {r:,.1f}x** (e2e 단축 {d:,.0f}ms)",
        "per_dec": "**결정별 latency**",
        "h_node": "노드", "h_dec": "결정", "h_label": "판정 label",
        "same": "{l} (동일)",
        "team_cluster": "12개 전문 상담팀 (D2 선택지)", "team_assign": "1개 팀 배정",
    },
    "en": {
        "customer": "customer", "agent": "support agent",
        "decide": "decision", "gen": "generate (Sonnet 4.5)",
        "gen_prefix": "generate", "tool": "tool run",
        "auto": "tool auto-selected (single remaining option)",
        "summary": "summary (first line)", "final": "--- final reply ---",
        "totals": ">>> [{arm}] {n} decisions, total {dec:,.0f}ms | generation {llm:,.0f}ms | e2e {e2e:,.0f}ms",
        "h_metric": "metric", "h_ratio": "ratio", "h_ndec": "decisions",
        "h_decsum": "**decision latency total**", "h_llm": "generation (LLM) latency",
        "speedup": "**decision-layer speedup: {r:,.1f}x** (e2e saved {d:,.0f}ms)",
        "per_dec": "**per-decision latency**",
        "h_node": "node", "h_dec": "decision", "h_label": "label",
        "same": "{l} (same)",
        "team_cluster": "12 specialist teams (D2 choices)", "team_assign": "assign 1 team",
    },
}
UI = _UI[LANG]


def make_clients():
    from anthropic import AnthropicBedrock
    client = AnthropicBedrock(aws_region="us-west-2", max_retries=8)
    llm = BedrockLLM(client, BEDROCK_MODEL_IDS["sonnet-4-5"], PRICING["sonnet-4-5"])
    return client, llm


def make_graphs():
    client, llm = make_clients()
    return {
        "Haiku": build_graph_v2(
            HaikuDecision(client, BEDROCK_MODEL_IDS["haiku-4-5"],
                          PRICING["haiku-4-5"]), llm),
        "StrandsDecider": build_graph_v2(SystemOneDecision(), llm),
    }, client


def _print_event(ev):
    kind = ev["kind"]
    if kind == "decision":
        conf = f", confidence {ev['confidence']:.3f}" if ev.get("confidence") is not None else ""
        print(f"      {UI['decide']}[{ev.get('decision','?').upper()}] -> {ev['label']}"
              f"   ({ev['latency_ms']:,.0f}ms{conf})")
    elif kind == "llm":
        backend = ev.get("backend") or "Sonnet 4.5"
        print(f"      {UI['gen_prefix']}({backend})   ({ev['latency_ms']:,.0f}ms)")
    elif kind == "tool":
        print(f"      {UI['tool']}: {ev['node'].replace('tool:', '')}"
              f"   ({ev['latency_ms']:,.1f}ms)")
    elif kind == "auto":
        print(f"      {UI['auto']}: {ev['label']}")


def run_verbose(graph, question, script, arm_name):
    print(f"\n{'=' * 86}\n[{arm_name}] {UI['customer']}: {question}\n{'=' * 86}")
    t0 = time.perf_counter()
    step = 0
    events = []
    stream = graph.stream(
        {"conv_id": f"ws-{arm_name}-{int(t0)}", "arm": arm_name, "lang": LANG,
         "script": script, "trace": [],
         "dialogue": [{"role": "customer", "text": question}]},
        config={"recursion_limit": 60}, stream_mode="updates")
    for update in stream:
        for node, delta in update.items():
            delta = delta or {}
            step += 1
            print(f"\nSTEP {step} ▶ {node} — {NODE_DESC.get(node, '')}")
            for ev in delta.get("trace", []):
                events.append(ev)
                _print_event(ev)
            for turn in delta.get("dialogue", []):
                who = UI["agent"] if turn["role"] == "agent" else UI["customer"]
                print(f"      {who}: {turn['text']}")
            if delta.get("specialist_summary"):
                head = delta["specialist_summary"].strip().splitlines()[0]
                print(f"      {UI['summary']}: {head[:80]}")
            if delta.get("final_response"):
                print(f"      {UI['final']}\n      "
                      + delta["final_response"].strip().replace("\n", "\n      "))
    e2e = (time.perf_counter() - t0) * 1000.0
    dec = [e for e in events if e["kind"] == "decision"]
    res = {
        "arm": arm_name,
        "n_decisions": len(dec),
        "decision_ms": sum(e["latency_ms"] for e in dec),
        "llm_ms": sum(e["latency_ms"] for e in events if e["kind"] == "llm"),
        "e2e_ms": e2e,
        "decisions": [(e["node"], e.get("decision"), e["label"], e["latency_ms"])
                      for e in dec],
    }
    print("\n" + UI["totals"].format(arm=arm_name, n=res["n_decisions"],
                                     dec=res["decision_ms"], llm=res["llm_ms"],
                                     e2e=res["e2e_ms"]))
    return res


def compare(res_a, res_b):
    """두 arm 수행시간 비교를 markdown 표로 display (res_a=Haiku, res_b=StrandsDecider 권장)."""
    from IPython.display import Markdown, display
    a, b = res_a, res_b
    ratio = a["decision_ms"] / b["decision_ms"] if b["decision_ms"] else float("nan")
    def x(va, vb):
        return f"**{va / vb:,.1f}x**" if vb else "-"
    md = [f"| {UI['h_metric']} | {a['arm']} | {b['arm']} | {UI['h_ratio']}({a['arm']}/{b['arm']}) |",
          "|---|---:|---:|---:|",
          f"| {UI['h_ndec']} | {a['n_decisions']} | {b['n_decisions']} | |",
          f"| {UI['h_decsum']} | **{a['decision_ms']:,.0f}ms** | **{b['decision_ms']:,.0f}ms** "
          f"| {x(a['decision_ms'], b['decision_ms'])} |",
          f"| {UI['h_llm']} | {a['llm_ms']:,.0f}ms | {b['llm_ms']:,.0f}ms "
          f"| {x(a['llm_ms'], b['llm_ms'])} |",
          f"| e2e | {a['e2e_ms']:,.0f}ms | {b['e2e_ms']:,.0f}ms "
          f"| {x(a['e2e_ms'], b['e2e_ms'])} |",
          "",
          UI["speedup"].format(r=ratio, d=a["e2e_ms"] - b["e2e_ms"]),
          "",
          UI["per_dec"],
          "",
          f"| {UI['h_node']} | {UI['h_dec']} | {a['arm']} | {b['arm']} "
          f"| {UI['h_ratio']} | {UI['h_label']} |",
          "|---|---|---:|---:|---:|---|"]
    for (na, da, la, ma), (nb, db, lb, mb) in zip(a["decisions"], b["decisions"]):
        same = UI["same"].format(l=la) if la == lb \
            else f"{a['arm']}={la} / {b['arm']}={lb}"
        md.append(f"| {na} | {(da or '?').upper()} | {ma:,.0f}ms | {mb:,.0f}ms "
                  f"| {x(ma, mb)} | {same} |")
    display(Markdown("\n".join(md)))


# ------------------------------------------------------------ graph drawing --
# 노드 캡션 (그래프 시각화 전용, 언어별)
_NODES = {
    "ko": {
        "START": None, "END": None,
        "guardrail_check": "guardrail_check\n[D1] 접수 심사\n정상 / 범위밖 / 컴플라이언스",
        "intent_router": "intent_router\n[D2] 업무 분류\n(12개 전문 상담팀)",
        "clarify_gate": "clarify_gate\n[D5] 정보 충분성 판정",
        "ask_clarify": "ask_clarify\n되묻기 질문 생성(Sonnet)\n+ 고객 답변 수신",
        "tool_step": "tool_step\n[D3] tool 선택 -> 실행\n[D6] 결과 검토",
        "give_up": "give_up\n자동 처리 포기",
        "summarize": "summarize\n사실 요약 (Sonnet)",
        "risk_check": "risk_check\n[D7] 리스크 점검\n(fraud/AML flag)",
        "kyc_verify": "kyc_verify\nKYC 재검증 tool\n[D8] 통과 판정",
        "escalation_gate": "escalation_gate\n[D4] 처리 방침\n종결 / 추가질문 / 이관",
        "respond": "respond\n최종 응답 작성 (Sonnet)",
    },
    "en": {
        "START": None, "END": None,
        "guardrail_check": "guardrail_check\n[D1] intake screening\nin-scope / out-of-scope / compliance",
        "intent_router": "intent_router\n[D2] routing\n(12 specialist teams)",
        "clarify_gate": "clarify_gate\n[D5] information sufficiency",
        "ask_clarify": "ask_clarify\nwrite clarifying question (Sonnet)\n+ receive customer reply",
        "tool_step": "tool_step\n[D3] pick tool -> run\n[D6] review result",
        "give_up": "give_up\nabandon automated handling",
        "summarize": "summarize\nfact summary (Sonnet)",
        "risk_check": "risk_check\n[D7] risk check\n(fraud/AML flag)",
        "kyc_verify": "kyc_verify\nKYC re-verification tool\n[D8] pass decision",
        "escalation_gate": "escalation_gate\n[D4] final action\nresolve / clarify / handoff",
        "respond": "respond\nwrite final reply (Sonnet)",
    },
}
_DECISION_NODES = {"guardrail_check", "intent_router", "clarify_gate", "tool_step",
                   "risk_check", "kyc_verify", "escalation_gate"}
_LLM_NODES = {"ask_clarify", "summarize", "respond"}

# (src, dst, edge label, style) — style: "", "dashed", "loop"(빨간 굵은선)
_EDGES = {
    "ko": [
        ("START", "guardrail_check", "", ""),
        ("guardrail_check", "intent_router", "정상 문의", ""),
        ("guardrail_check", "respond", "범위밖/민감\n(거절 응답 직행)", "dashed"),
        ("intent_router", "clarify_gate", "", ""),
        ("clarify_gate", "ask_clarify", "정보 부족", "loop"),
        ("ask_clarify", "clarify_gate", "재판정 (loop, 최대 2회)", "loop"),
        ("clarify_gate", "tool_step", "정보 충분 /\n되묻기 상한 도달", ""),
        ("tool_step", "tool_step", "재시도 (loop,\n대체 tool, 최대 2회)", "loop"),
        ("tool_step", "summarize", "tool 성공", ""),
        ("tool_step", "give_up", "포기", "dashed"),
        ("give_up", "escalation_gate", "", "dashed"),
        ("summarize", "risk_check", "", ""),
        ("risk_check", "kyc_verify", "검증 필요", ""),
        ("kyc_verify", "escalation_gate", "", ""),
        ("risk_check", "escalation_gate", "이상 없음", ""),
        ("escalation_gate", "respond", "", ""),
        ("respond", "END", "", ""),
    ],
    "en": [
        ("START", "guardrail_check", "", ""),
        ("guardrail_check", "intent_router", "in-scope", ""),
        ("guardrail_check", "respond", "out-of-scope/sensitive\n(straight to refusal)", "dashed"),
        ("intent_router", "clarify_gate", "", ""),
        ("clarify_gate", "ask_clarify", "info missing", "loop"),
        ("ask_clarify", "clarify_gate", "re-check (loop, max 2)", "loop"),
        ("clarify_gate", "tool_step", "sufficient /\nask budget exhausted", ""),
        ("tool_step", "tool_step", "retry (loop,\nalternative tool, max 2)", "loop"),
        ("tool_step", "summarize", "tool success", ""),
        ("tool_step", "give_up", "give up", "dashed"),
        ("give_up", "escalation_gate", "", "dashed"),
        ("summarize", "risk_check", "", ""),
        ("risk_check", "kyc_verify", "verification needed", ""),
        ("kyc_verify", "escalation_gate", "", ""),
        ("risk_check", "escalation_gate", "no risk", ""),
        ("escalation_gate", "respond", "", ""),
        ("respond", "END", "", ""),
    ],
}

# 12개 전문 상담팀 캡션 (언어별)
_TEAMS = {
    "ko": {
        "cards": "카드 발급/배송", "card_security": "분실/도난/보안",
        "disputes_fees": "분쟁/수수료 이의", "transfers": "이체/송금",
        "accounts_balance": "잔액/입금 반영", "top_up": "계좌 충전",
        "exchange_fx": "환전/환율", "refunds": "환불",
        "declined_pay": "승인 거절/보류", "identity_verif": "신원확인(KYC)",
        "account_mgmt": "계정 관리", "general_faq": "일반 FAQ",
    },
    "en": {
        "cards": "card issuing/delivery", "card_security": "lost/stolen/security",
        "disputes_fees": "disputes/fees", "transfers": "transfers/remittance",
        "accounts_balance": "balance/deposits", "top_up": "account top-up",
        "exchange_fx": "FX/exchange rate", "refunds": "refunds",
        "declined_pay": "declined/pending pay", "identity_verif": "identity (KYC)",
        "account_mgmt": "account management", "general_faq": "general FAQ",
    },
}

# 하위 호환 별칭 (기존 노트북이 참조)
KO_NODES = _NODES["ko"]
KO_EDGES = _EDGES["ko"]
KO_TEAMS = _TEAMS["ko"]

# 01 노트북의 조립 단계: 단계별로 보이는 노드 집합 (누적)
BUILD_STAGES = [
    ["START", "guardrail_check", "intent_router", "respond", "END"],
    ["clarify_gate", "ask_clarify"],
    ["tool_step", "give_up"],
    ["summarize", "risk_check", "kyc_verify"],
    ["escalation_gate"],
]


def draw_graph(visible=None, highlight=None, new=None, width=None, show_teams=False,
               lang=None):
    """workflow 그래프를 그려 IPython Image로 반환 (캡션은 lang/DMWS_LANG 언어).

    visible    : 표시할 노드 집합 (None=전체). edge는 양끝이 모두 보일 때만 그림
    highlight  : 시나리오 경로 강조(빨간 두꺼운 테두리, team_<cat> 포함 가능)
    new        : 이번 단계에 추가된 노드 강조(초록 두꺼운 테두리)
    show_teams : intent_router의 12개 전문 상담팀을 점선 cluster 노드로 표시
    lang       : 'ko' | 'en' (기본: DMWS_LANG)
    """
    import graphviz
    from IPython.display import Image
    lang = lang or LANG
    nodes, edges, teams = _NODES[lang], _EDGES[lang], _TEAMS[lang]
    ui = _UI[lang]
    visible = set(visible) if visible else set(nodes)
    highlight = set(highlight or [])
    new = set(new or [])
    g = graphviz.Digraph(format="png")
    g.attr(rankdir="TB", fontname=FONT, splines="spline",
           nodesep="0.3", ranksep="0.4", dpi="110", compound="true")
    g.attr("node", fontname=FONT, fontsize="11")
    g.attr("edge", fontname=FONT, fontsize="9")
    for n in visible:
        label = nodes[n]
        kw = {}
        if n in ("START", "END"):
            kw = dict(shape="circle" if n == "START" else "doublecircle",
                      style="filled", fillcolor="#333333", fontcolor="white", width="0.45")
            label = n
        elif n in _DECISION_NODES:
            kw = dict(shape="diamond", style="filled", fillcolor="#FFE08A")
        elif n in _LLM_NODES:
            kw = dict(shape="box", style="filled,rounded", fillcolor="#BFD7FF")
        else:
            kw = dict(shape="box", style="filled", fillcolor="#EEEEEE")
        if n in new:
            kw.update(color="#1E8449", penwidth="3")
        elif n in highlight:
            kw.update(color="#C0392B", penwidth="3")
        g.node(n, label, **kw)
    for src, dst, lbl, style in edges:
        if src not in visible or dst not in visible:
            continue
        kw = {"label": lbl}
        if style == "dashed":
            kw["style"] = "dashed"
        elif style == "loop":
            kw.update(color="#CC6677", penwidth="2.2", fontcolor="#CC6677")
        if src in highlight and dst in highlight and style != "dashed":
            kw.setdefault("penwidth", "2.2")
        if show_teams and (src, dst) == ("intent_router", "clarify_gate"):
            continue   # 팀 cluster를 경유하도록 아래에서 대체
        g.edge(src, dst, **kw)
    if show_teams and "intent_router" in visible:
        with g.subgraph(name="cluster_teams") as c:
            c.attr(label=ui["team_cluster"], fontname=FONT,
                   fontsize="10", style="dashed", color="#888888")
            for cat, cap in teams.items():
                kw = dict(shape="box", style="filled", fillcolor="#F6EBD9", fontsize="9")
                if f"team_{cat}" in highlight:
                    kw.update(color="#C0392B", penwidth="3")
                c.node(f"team_{cat}", f"{cat}\n{cap}", **kw)
        g.edge("intent_router", "team_exchange_fx", lhead="cluster_teams",
               label=ui["team_assign"])
        if "clarify_gate" in visible:
            g.edge("team_exchange_fx", "clarify_gate", ltail="cluster_teams")
    return Image(data=g.pipe(), width=width)


def draw_build_stage(stage, width=640, show_teams=True, lang=None):
    """01 노트북용: stage(1부터) 단계까지 누적된 그래프, 이번 단계 노드는 초록 강조.
    show_teams=True면 intent_router 아래 12개 상담팀 cluster를 함께 표시."""
    visible, new = [], []
    for i, nodes in enumerate(BUILD_STAGES[:stage], start=1):
        visible += nodes
        if i == stage:
            new = nodes
    return draw_graph(visible=visible, new=new, width=width, show_teams=show_teams,
                      lang=lang)


# 시나리오별 통과 경로 (전체 그래프에서 빨간 강조; team_* 는 배정되는 상담팀 노드)
# EN은 발화 표현상 시나리오 1이 declined_pay 팀으로 라우팅됨 (KR=cards와 대비 포인트)
def scenario_path(n, lang=None):
    path = list(SCENARIO_PATHS[n])
    if (lang or LANG) == "en" and n == 1:
        path[path.index("team_cards")] = "team_declined_pay"
    return path


SCENARIO_PATHS = {
    1: ["START", "guardrail_check", "intent_router", "team_cards", "clarify_gate",
        "ask_clarify", "tool_step", "summarize", "risk_check", "escalation_gate",
        "respond", "END"],
    2: ["START", "guardrail_check", "intent_router", "team_transfers", "clarify_gate",
        "tool_step", "summarize", "risk_check", "escalation_gate", "respond", "END"],
    3: ["START", "guardrail_check", "intent_router", "team_card_security",
        "clarify_gate", "tool_step", "summarize", "risk_check", "kyc_verify",
        "escalation_gate", "respond", "END"],
}


# ------------------------------------------------------------- notebook 07 --
# 생성 데모용 상담원 persona (notebook 07의 자유 텍스트 생성 호출)
_GEN_SYSTEM = {
    "ko": ("너는 은행 고객센터의 상담 agent다. 고객의 질문에 정확하고 간결하게 "
           "한국어로 답하라. 모르는 내용은 추측하지 말고 확인이 필요하다고 안내하라."),
    "en": ("You are a banking customer-support agent. Answer the customer's "
           "question accurately and concisely in English. If you are unsure, say "
           "the matter needs verification instead of guessing."),
}
GEN_SYSTEM = _GEN_SYSTEM[LANG]

# notebook 07: 난이도 라우터 — /v1/systemone 2-choice decision
# 주의: choices 순서는 ["complex", "simple"] — decision_id "route"는 FALLBACKS에
# 없으므로 라우터 오류 시 choices[0]=complex(→Sonnet)로 안전하게 폴백된다.
ROUTER_CHOICES = ["complex", "simple"]
_ROUTER_QUESTION = {
    "ko": ("다음 은행 고객 문의의 난이도를 분류하라. "
           "simple = 한두 문장으로 답할 수 있는 단일 사실·절차 안내"
           "(운영시간, 메뉴 위치, 단순 신고/변경 방법 등). "
           "complex = 여러 단계의 추론이나 여러 항목의 비교, 규정·세금·수수료 해석, "
           "고객 상황에 따른 판단이 필요한 문의."),
    "en": ("Classify the difficulty of this banking customer inquiry. "
           "simple = a single fact or procedure that can be answered in one or two "
           "sentences (business hours, where a menu is, how to report/change one "
           "thing). complex = requires multi-step reasoning, comparing options, "
           "interpreting policy/tax/fee rules, or judgment about the customer's "
           "specific situation."),
}
ROUTER_QUESTION = _ROUTER_QUESTION[LANG]

# notebook 07: 난이도 혼합 10문항 (gold = 출제 의도 — simple 5 / complex 5)
ROUTING_QUESTIONS = [
    {"q": "고객센터 운영 시간이 어떻게 되나요?", "gold": "simple"},
    {"q": "체크카드 비밀번호는 앱 어디에서 변경할 수 있나요?", "gold": "simple"},
    {"q": "카드를 분실했는데 분실 신고는 어떻게 하나요?", "gold": "simple"},
    {"q": "해외 ATM에서도 출금이 가능한가요?", "gold": "simple"},
    {"q": "계좌 잔액 증명서는 어디서 발급받나요?", "gold": "simple"},
    {"q": ("급여계좌를 A은행에서 B은행으로 옮기면서 자동이체 7건과 카드 결제계좌를 "
           "함께 이전하려고 합니다. 연체나 불이익 없이 진행하는 순서를 알려주세요."),
     "gold": "complex"},
    {"q": ("해외 송금이 중개은행 수수료로 일부 차감되어 도착했습니다. 수수료 부담 "
           "방식(OUR/SHA/BEN)의 차이와 차감분 환불 가능 여부를 설명해 주세요."),
     "gold": "complex"},
    {"q": ("USD 5,000 환전 기준으로 환율 우대 80%와 송금 수수료 면제 중 어느 쪽이 "
           "유리한지 계산 과정을 포함해 비교해 주세요."), "gold": "complex"},
    {"q": ("연 매출 3억원 개인사업자입니다. 사업자 대출과 개인 신용대출을 함께 "
           "이용할 때 DSR 규제가 각각 어떻게 적용되는지 설명해 주세요."),
     "gold": "complex"},
    {"q": ("미성년 자녀 명의 계좌로 매달 50만원씩 이체해 주려고 합니다. 증여세 "
           "신고 기준 금액과 신고 절차를 알려주세요."), "gold": "complex"},
]

ROUTING_QUESTIONS_EN = [
    {"q": "What are your customer-service hours?", "gold": "simple"},
    {"q": "Where in the app can I change my debit-card PIN?", "gold": "simple"},
    {"q": "I lost my card. How do I report it lost?", "gold": "simple"},
    {"q": "Can I withdraw cash from ATMs abroad?", "gold": "simple"},
    {"q": "Where can I get a balance certificate issued?", "gold": "simple"},
    {"q": ("I am moving my salary account from Bank A to Bank B and need to move "
           "7 autopay registrations and my card billing account with it. What "
           "order should I do this in to avoid missed payments or penalties?"),
     "gold": "complex"},
    {"q": ("My overseas transfer arrived short because of intermediary bank fees. "
           "Explain the difference between OUR/SHA/BEN fee options and whether "
           "the deducted amount can be refunded."), "gold": "complex"},
    {"q": ("For exchanging USD 5,000, which is better: an 80% FX-rate discount or "
           "a transfer-fee waiver? Compare them with the calculation."),
     "gold": "complex"},
    {"q": ("I am a sole proprietor with USD 250k annual revenue. How do "
           "debt-service-ratio rules apply when I hold both a business loan and "
           "a personal credit loan?"), "gold": "complex"},
    {"q": ("I want to transfer USD 400 every month into my minor child's account. "
           "What is the gift-tax reporting threshold and the filing procedure?"),
     "gold": "complex"},
]


def routing_questions(lang=None):
    return ROUTING_QUESTIONS_EN if (lang or LANG) == "en" else ROUTING_QUESTIONS


# ------------------------------------------------------------- notebook 08 --
# 미니 RAG 지식 베이스 (합성 KB한빛 FAQ 12건) — 모든 수치·절차는 가상입니다.
# Mini RAG knowledge base (12 synthetic KB-Hanbit FAQ passages) — all fictional.
_RAG_PASSAGES = {
    "ko": [
        {"id": "fx_fee", "title": "해외송금 수수료 부담 방식",
         "text": ("해외송금 수수료 부담 방식은 OUR/SHA/BEN 세 가지입니다. OUR는 "
                  "송금인이 전액 부담(전신료 외 중개은행 수수료 USD 25 선납), SHA는 "
                  "송금인과 수취인이 분담, BEN은 수취인이 전액 부담합니다. 수취인이 "
                  "금액을 온전히 받아야 하면 OUR를 선택하세요.")},
        {"id": "fx_rate", "title": "환전 환율 우대",
         "text": ("모바일 앱 환전은 USD·EUR·JPY 기준 환율 우대 최대 80%가 적용됩니다. "
                  "영업점 창구 환전은 최대 50%입니다. 우대율은 고객 등급과 환전 금액에 "
                  "따라 달라지며, 앱의 환전 신청 화면에서 적용 우대율을 미리 확인할 수 "
                  "있습니다.")},
        {"id": "card_lost", "title": "카드 분실 신고",
         "text": ("카드 분실 시 앱의 '카드 > 분실신고' 메뉴 또는 24시간 콜센터 "
                  "1588-0000으로 신고하세요. 신고 즉시 카드가 정지되며, 재발급 신청 시 "
                  "수수료 2,000원이 부과됩니다. 신고 전 발생한 부정 사용액은 신고일로부터 "
                  "60일 이내 보상 신청할 수 있습니다.")},
        {"id": "card_reissue", "title": "체크카드 재발급 배송",
         "text": ("체크카드 재발급은 앱에서 신청하면 영업일 기준 3～5일 내 등기우편으로 "
                  "배송됩니다. 실물 카드 수령 전에도 모바일카드를 발급받아 온라인 결제와 "
                  "간편결제에 바로 사용할 수 있습니다.")},
        {"id": "atm_abroad", "title": "해외 ATM 출금 한도와 수수료",
         "text": ("해외 ATM 출금은 1회 USD 400, 1일 USD 1,000까지 가능합니다. 건당 "
                  "수수료 USD 3와 국제 네트워크 수수료(출금액의 1%)가 부과됩니다. 일부 "
                  "국가에서는 현지 ATM 운영사가 별도 수수료를 추가로 청구할 수 있습니다.")},
        {"id": "transfer_limit", "title": "이체 한도 변경",
         "text": ("모바일 이체 한도는 기본 1일 5,000만원, 1회 1,000만원입니다. 한도 "
                  "상향은 영업점 방문 또는 앱의 영상통화 본인확인으로 신청할 수 있으며, "
                  "영상통화 상향은 1일 최대 1억원까지 가능합니다.")},
        {"id": "balance_cert", "title": "잔액증명서 발급",
         "text": ("잔액증명서는 앱의 '증명서 발급' 메뉴에서 PDF로 즉시 발급되며 "
                  "수수료는 무료입니다. 영업점 창구 발급은 통당 2,000원입니다. 증명서는 "
                  "발급일 당일 0시 기준 잔액으로 작성되며, 발급 당일에는 해당 계좌의 "
                  "출금이 제한됩니다.")},
        {"id": "autopay_move", "title": "자동이체 일괄 이전",
         "text": ("계좌이동서비스를 이용하면 타행 자동이체를 새 계좌로 일괄 이전할 수 "
                  "있습니다. 신청 후 5영업일 이내에 완료되며, 이전 처리 중 출금이 "
                  "실패한 건은 다음 영업일에 1회 자동 재출금됩니다.")},
        {"id": "dormant", "title": "휴면계좌(장기 미사용) 해제",
         "text": ("1년 이상 입출금 거래가 없는 계좌는 전자금융 거래가 제한됩니다. "
                  "앱에서 본인확인(신분증 촬영 + 안면인증)을 완료하면 즉시 해제되며, "
                  "영업점 방문 시에는 신분증만 지참하면 됩니다.")},
        {"id": "otp", "title": "OTP·보안매체",
         "text": ("모바일OTP는 앱에서 무료로 발급됩니다. 카드형 OTP는 발급 수수료 "
                  "5,000원이며 영업점에서 수령합니다. 보안매체를 분실하면 기존 매체를 "
                  "폐기 등록한 뒤 새 매체를 등록해야 이체 거래가 가능합니다.")},
        {"id": "deposit_cancel", "title": "적금 중도해지",
         "text": ("적금을 만기 전에 해지하면 약정금리 대신 중도해지금리(가입 기간에 "
                  "따라 연 0.1～1.0%)가 적용됩니다. 중도해지는 앱에서 신청할 수 있으며 "
                  "해지금은 신청 즉시 입출금 계좌로 입금됩니다.")},
        {"id": "dispute", "title": "해외 결제 이의제기",
         "text": ("해외 가맹점 결제에 대한 이의제기는 승인일로부터 120일 이내에 접수해야 "
                  "합니다. 앱의 '카드 > 이용내역 > 이의제기' 메뉴에서 신청하며, 국제 "
                  "브랜드사 조사를 거쳐 접수 후 30～45일 내에 결과가 안내됩니다.")},
    ],
    "en": [
        {"id": "fx_fee", "title": "Overseas remittance fee options",
         "text": ("Overseas remittance offers three fee options: OUR, SHA and BEN. "
                  "With OUR the sender pays everything (wire fee plus a prepaid "
                  "USD 25 intermediary-bank fee), with SHA sender and recipient "
                  "share, and with BEN the recipient pays all fees. Choose OUR if "
                  "the recipient must receive the full amount.")},
        {"id": "fx_rate", "title": "FX-rate discount",
         "text": ("Mobile-app currency exchange applies an FX-rate discount of up "
                  "to 80% for USD, EUR and JPY. Branch-counter exchange offers up "
                  "to 50%. The discount depends on customer tier and amount, and "
                  "the app shows the applied rate before you confirm.")},
        {"id": "card_lost", "title": "Reporting a lost card",
         "text": ("If your card is lost, report it in the app under 'Card > Report "
                  "loss' or call the 24-hour center at 1588-0000. The card is "
                  "blocked immediately, and reissuing costs a 2,000 KRW fee. "
                  "Fraudulent charges made before the report can be claimed within "
                  "60 days of the report date.")},
        {"id": "card_reissue", "title": "Debit-card reissue delivery",
         "text": ("A reissued debit card requested in the app is delivered by "
                  "registered mail within 3-5 business days. Before the physical "
                  "card arrives you can issue a mobile card and use it right away "
                  "for online and contactless payments.")},
        {"id": "atm_abroad", "title": "Overseas ATM limits and fees",
         "text": ("Overseas ATM withdrawals are limited to USD 400 per withdrawal "
                  "and USD 1,000 per day. Each withdrawal incurs a USD 3 fee plus "
                  "a 1% international network fee. Some local ATM operators may "
                  "charge an additional surcharge of their own.")},
        {"id": "transfer_limit", "title": "Changing transfer limits",
         "text": ("The default mobile transfer limit is KRW 50 million per day and "
                  "KRW 10 million per transaction. You can raise it at a branch or "
                  "through video-call identity verification in the app; the "
                  "video-call route allows up to KRW 100 million per day.")},
        {"id": "balance_cert", "title": "Balance certificate issuance",
         "text": ("A balance certificate can be issued instantly as a PDF from the "
                  "app's 'Certificates' menu free of charge. Branch-counter "
                  "issuance costs 2,000 KRW per copy. The certificate reflects the "
                  "balance as of midnight on the issue date, and withdrawals from "
                  "that account are restricted for the rest of that day.")},
        {"id": "autopay_move", "title": "Bulk autopay transfer",
         "text": ("The account-switching service moves autopay registrations from "
                  "another bank to your new account in one request. It completes "
                  "within 5 business days, and any debit that fails during the "
                  "switch is retried automatically once on the next business day.")},
        {"id": "dormant", "title": "Reactivating a dormant account",
         "text": ("Accounts with no deposits or withdrawals for over a year are "
                  "restricted from electronic banking. Completing identity "
                  "verification in the app (ID photo plus face recognition) lifts "
                  "the restriction immediately; at a branch you only need your ID.")},
        {"id": "otp", "title": "OTP and security devices",
         "text": ("The mobile OTP is issued free in the app. A card-type OTP costs "
                  "a 5,000 KRW issuance fee and is picked up at a branch. If you "
                  "lose a security device, you must deregister the old one and "
                  "register a new one before transfers are allowed again.")},
        {"id": "deposit_cancel", "title": "Early savings-account closure",
         "text": ("Closing an installment savings account before maturity applies "
                  "the early-closure rate (0.1-1.0% per year depending on how long "
                  "you held it) instead of the contracted rate. You can close it "
                  "in the app, and the balance is paid to your checking account "
                  "immediately.")},
        {"id": "dispute", "title": "Disputing an overseas charge",
         "text": ("A dispute against an overseas merchant charge must be filed "
                  "within 120 days of the approval date. File it in the app under "
                  "'Card > Transactions > Dispute'; after the international card "
                  "network's investigation, the result is announced within 30-45 "
                  "days of filing.")},
    ],
}

# 게이트 1 평가용 쿼리 10건 — gate_gold: retrieve(문서 필요) / direct(바로 응대),
# gold_id: 답 근거 문서 (e2e에서 filter의 정답 보존 확인에 사용)
_RAG_QUERIES = {
    "ko": [
        {"q": "해외송금 수수료를 제가 전부 부담하려면 어떤 방식을 선택해야 하나요?",
         "gate_gold": "retrieve", "gold_id": "fx_fee"},
        {"q": "모바일 앱에서 환전하면 환율 우대를 얼마나 받을 수 있나요?",
         "gate_gold": "retrieve", "gold_id": "fx_rate"},
        {"q": "카드를 잃어버렸는데 분실 신고는 어떻게 하나요?",
         "gate_gold": "retrieve", "gold_id": "card_lost"},
        {"q": "해외 ATM에서 하루에 얼마까지 출금할 수 있나요?",
         "gate_gold": "retrieve", "gold_id": "atm_abroad"},
        {"q": "잔액증명서는 어디서 발급받을 수 있고 수수료가 있나요?",
         "gate_gold": "retrieve", "gold_id": "balance_cert"},
        {"q": "적금을 만기 전에 해지하면 이자는 어떻게 되나요?",
         "gate_gold": "retrieve", "gold_id": "deposit_cancel"},
        {"q": "해외 결제 건에 이의제기를 하려면 언제까지 신청해야 하나요?",
         "gate_gold": "retrieve", "gold_id": "dispute"},
        {"q": "안녕하세요, 지금 상담 가능한가요?", "gate_gold": "direct", "gold_id": None},
        {"q": "빠른 답변 감사합니다. 좋은 하루 되세요!", "gate_gold": "direct", "gold_id": None},
        {"q": "네, 알겠습니다. 말씀하신 대로 진행해 주세요.", "gate_gold": "direct", "gold_id": None},
    ],
    "en": [
        {"q": "Which fee option should I choose so I pay all overseas remittance fees myself?",
         "gate_gold": "retrieve", "gold_id": "fx_fee"},
        {"q": "How much of an FX-rate discount do I get when exchanging in the mobile app?",
         "gate_gold": "retrieve", "gold_id": "fx_rate"},
        {"q": "I lost my card. How do I report it lost?",
         "gate_gold": "retrieve", "gold_id": "card_lost"},
        {"q": "How much cash can I withdraw per day from ATMs abroad?",
         "gate_gold": "retrieve", "gold_id": "atm_abroad"},
        {"q": "Where can I get a balance certificate issued, and is there a fee?",
         "gate_gold": "retrieve", "gold_id": "balance_cert"},
        {"q": "What happens to the interest if I close my savings account before maturity?",
         "gate_gold": "retrieve", "gold_id": "deposit_cancel"},
        {"q": "Until when can I file a dispute against an overseas charge?",
         "gate_gold": "retrieve", "gold_id": "dispute"},
        {"q": "Hello, is anyone available to chat right now?", "gate_gold": "direct", "gold_id": None},
        {"q": "Thanks for the quick answer. Have a great day!", "gate_gold": "direct", "gold_id": None},
        {"q": "Okay, understood. Please go ahead as you described.", "gate_gold": "direct", "gold_id": None},
    ],
}

# RAG 판단 3종의 rubric — 선택지 순서가 폴백 방향(choices[0])을 정합니다:
#   gate   -> 오류 시 retrieve(검색하는 쪽 = 품질 안전)
#   filter -> 오류 시 yes(문서 유지 = 정보 손실 방지)
#   ground -> 오류 시 no(응답 차단 = hallucination 안전)
RAG_GATE_CHOICES = ["retrieve", "direct"]
RAG_FILTER_CHOICES = ["yes", "no"]
RAG_GROUND_CHOICES = ["no", "yes"]

_RAG_GATE_QUESTION = {
    "ko": ("다음 고객 발화에 답하려면 은행 내부 지식 문서 검색이 필요한지 판정하라. "
           "retrieve = 상품·수수료·한도·절차·규정 같은 문서의 사실 정보가 있어야 "
           "답할 수 있는 질문. direct = 인사말, 감사 인사, 단순 동의/확인처럼 문서 "
           "없이 바로 응대할 수 있는 발화."),
    "en": ("Decide whether answering this customer utterance requires searching "
           "the bank's internal knowledge documents. retrieve = the answer needs "
           "factual document content such as products, fees, limits, procedures "
           "or policies. direct = a greeting, thanks, or simple acknowledgement "
           "that can be answered without any document."),
}
_RAG_FILTER_QUESTION = {
    "ko": ("질문: {q}\n위 질문에 답하는 근거로 아래 문서가 쓸모 있는지 판정하라. "
           "yes = 문서가 질문의 답을 직접 담고 있다. no = 질문과 다른 주제의 "
           "문서다."),
    "en": ("Question: {q}\nDecide whether the document below is useful evidence "
           "for answering the question above. yes = the document directly "
           "contains the answer. no = the document is about a different topic."),
}
_RAG_GROUND_QUESTION = {
    "ko": ("답변의 사실 주장(수치·기한·수수료·절차)이 전부 문서 발췌에서 확인되는지 "
           "판정하라. yes = 모든 주장이 문서에 근거한다. no = 문서에 없거나 문서와 "
           "모순되는 주장이 있다."),
    "en": ("Decide whether every factual claim in the answer (numbers, deadlines, "
           "fees, procedures) is supported by the document excerpt. yes = all "
           "claims are grounded in the document. no = the answer contains a claim "
           "that is absent from or contradicts the document."),
}

# 게이트 3 평가용 (답변, 근거 문서) 쌍 — 2건은 근거함(yes), 2건은 조작된
# 수치/규정 포함(no). 답변은 재현성을 위해 고정 문자열입니다.
_RAG_GROUND_PROBES = {
    "ko": [
        {"doc_id": "fx_fee", "gold": "yes",
         "answer": ("수수료를 전부 부담하시려면 OUR 방식을 선택하시면 됩니다. 전신료 "
                    "외에 중개은행 수수료 USD 25를 선납하시며, 수취인은 금액을 온전히 "
                    "받게 됩니다.")},
        {"doc_id": "atm_abroad", "gold": "yes",
         "answer": ("해외 ATM에서는 1회 USD 400, 하루 USD 1,000까지 출금하실 수 "
                    "있습니다. 건당 수수료 USD 3와 출금액의 1%인 네트워크 수수료가 "
                    "부과됩니다.")},
        {"doc_id": "balance_cert", "gold": "no",
         "answer": ("잔액증명서는 영업점 창구에서 무료로 발급되며, 발급 후에도 해당 "
                    "계좌의 출금에는 아무 제한이 없습니다.")},
        {"doc_id": "deposit_cancel", "gold": "no",
         "answer": ("적금을 중도해지하셔도 약정금리가 전액 지급되며, 다만 중도해지 "
                    "위약금 30,000원이 부과됩니다.")},
    ],
    "en": [
        {"doc_id": "fx_fee", "gold": "yes",
         "answer": ("To pay all fees yourself, choose the OUR option. You prepay a "
                    "USD 25 intermediary-bank fee on top of the wire fee, and the "
                    "recipient receives the full amount.")},
        {"doc_id": "atm_abroad", "gold": "yes",
         "answer": ("At overseas ATMs you can withdraw up to USD 400 per "
                    "withdrawal and USD 1,000 per day. Each withdrawal costs USD 3 "
                    "plus a 1% network fee on the amount.")},
        {"doc_id": "balance_cert", "gold": "no",
         "answer": ("Balance certificates are issued free of charge at branch "
                    "counters, and there is no restriction at all on withdrawals "
                    "after issuance.")},
        {"doc_id": "deposit_cancel", "gold": "no",
         "answer": ("Even if you close the savings account early, the full "
                    "contracted interest rate is paid; only a 30,000 KRW "
                    "early-closure penalty applies.")},
    ],
}

# RAG 응답 생성 프롬프트 (생성은 양 arm 모두 Sonnet — 판단 backend만 교체)
_RAG_GEN_SYSTEM = {
    "ko": ("너는 은행 고객센터의 상담 agent다. 제공된 문서 발췌에 있는 내용만으로 "
           "고객 질문에 한국어로 간결하게 답하라. 문서가 제공되지 않은 경우에는 "
           "인사/확인 발화에 한두 문장으로 자연스럽게 응대하라."),
    "en": ("You are a banking customer-support agent. Answer the customer's "
           "question concisely in English using ONLY the provided document "
           "excerpts. If no documents are provided, respond naturally in one or "
           "two sentences to the greeting or acknowledgement."),
}


def rag_passages(lang=None):
    return _RAG_PASSAGES[lang or LANG]


def rag_queries(lang=None):
    return _RAG_QUERIES[lang or LANG]


def rag_ground_probes(lang=None):
    return _RAG_GROUND_PROBES[lang or LANG]


RAG_GATE_QUESTION = _RAG_GATE_QUESTION[LANG]
RAG_FILTER_QUESTION = _RAG_FILTER_QUESTION[LANG]
RAG_GROUND_QUESTION = _RAG_GROUND_QUESTION[LANG]
RAG_GEN_SYSTEM = _RAG_GEN_SYSTEM[LANG]


def _bigrams(text):
    t = re.sub(r"\s+", " ", text.lower())
    return {t[i:i + 2] for i in range(len(t) - 1)}


def rag_retrieve(query, k=4, lang=None):
    """문자 bigram 겹침 기반의 단순 lexical 검색기 (의존성 없음, 결정적).

    실무에서는 벡터 검색 + cross-encoder reranker를 쓰지만, 이 실습의 주제는
    검색이 아니라 '검색 전후의 판단'이므로 검색기는 가장 단순한 것으로
    고정합니다. 반환: [(passage dict, score)] 상위 k건."""
    qb = _bigrams(query)
    scored = []
    for p in rag_passages(lang):
        pb = _bigrams(p["title"] + " " + p["text"])
        scored.append((p, len(qb & pb) / (len(pb) ** 0.5)))
    return sorted(scored, key=lambda x: -x[1])[:k]


def p_yes(decision):
    """binary(yes/no) 판정 결과에서 p(yes) 확률을 읽는다 — label이 no면 1-conf."""
    if decision.confidence is None:
        return None
    return decision.confidence if decision.label == "yes" else 1.0 - decision.confidence


def rag_report(res_a, res_b):
    """notebook 08: 두 arm의 end-to-end 실행 결과 비교표 (res_a=Haiku 권장).

    res_*: {"arm", "n_queries", "n_judgments", "judge_ms", "gen_ms", "e2e_ms",
            "gate_correct", "ground_correct"} dict."""
    from IPython.display import Markdown, display
    a, b = res_a, res_b
    ratio = a["judge_ms"] / b["judge_ms"] if b["judge_ms"] else float("nan")
    per_a = a["judge_ms"] / a["n_queries"] / 1000.0
    per_b = b["judge_ms"] / b["n_queries"] / 1000.0
    def x(va, vb):
        return f"**{va / vb:,.1f}x**" if vb else "-"
    if LANG == "en":
        md = [
            f"| metric | {a['arm']} | {b['arm']} | ratio({a['arm']}/{b['arm']}) |",
            "|---|---:|---:|---:|",
            f"| queries | {a['n_queries']} | {b['n_queries']} | |",
            f"| judgments (gate+filter+ground) | {a['n_judgments']} | {b['n_judgments']} | |",
            f"| **judgment latency total** | **{a['judge_ms']:,.0f}ms** | **{b['judge_ms']:,.0f}ms** "
            f"| {x(a['judge_ms'], b['judge_ms'])} |",
            f"| generation (Sonnet) latency | {a['gen_ms']:,.0f}ms | {b['gen_ms']:,.0f}ms "
            f"| {x(a['gen_ms'], b['gen_ms'])} |",
            f"| e2e | {a['e2e_ms']:,.0f}ms | {b['e2e_ms']:,.0f}ms | {x(a['e2e_ms'], b['e2e_ms'])} |",
            "",
            f"**Judgment overhead per query: {per_a:,.2f}s -> {per_b:,.2f}s "
            f"({ratio:,.1f}x)** — generation stays on Sonnet in both arms, so the "
            f"e2e gap is the judgment share.",
        ]
    else:
        md = [
            f"| 항목 | {a['arm']} | {b['arm']} | 배수({a['arm']}/{b['arm']}) |",
            "|---|---:|---:|---:|",
            f"| 질의 수 | {a['n_queries']} | {b['n_queries']} | |",
            f"| 판단 횟수 (gate+filter+ground) | {a['n_judgments']} | {b['n_judgments']} | |",
            f"| **판단 latency 합계** | **{a['judge_ms']:,.0f}ms** | **{b['judge_ms']:,.0f}ms** "
            f"| {x(a['judge_ms'], b['judge_ms'])} |",
            f"| 생성(Sonnet) latency | {a['gen_ms']:,.0f}ms | {b['gen_ms']:,.0f}ms "
            f"| {x(a['gen_ms'], b['gen_ms'])} |",
            f"| e2e | {a['e2e_ms']:,.0f}ms | {b['e2e_ms']:,.0f}ms | {x(a['e2e_ms'], b['e2e_ms'])} |",
            "",
            f"**query당 판단 오버헤드: {per_a:,.2f}초 -> {per_b:,.2f}초 "
            f"({ratio:,.1f}x)** — 생성은 양 arm 모두 Sonnet이므로 e2e 차이가 곧 "
            f"판단 몫입니다.",
        ]
    display(Markdown("\n".join(md)))
    return {"judge_ratio": ratio, "per_query_a_s": per_a, "per_query_b_s": per_b,
            "e2e_saved_ms": a["e2e_ms"] - b["e2e_ms"]}


def routing_cost_report(baseline, routed, req_per_day=10_000):
    """notebook 07: 전부-Sonnet baseline vs 난이도 라우팅(simple->Haiku,
    complex->Sonnet) arm의 비용 비교표 출력.

    baseline/routed: {"backend", "cost_usd", "input_tokens", "output_tokens"} dict
    목록. backend는 "sonnet" 또는 "haiku" — 생성은 양 arm 모두 Bedrock이므로 모든
    행이 실제 토큰 비용을 가진다. 라우터(strands-decider E4B)는 decision layer가
    이미 사용 중인 GPU에서 돌므로 라우팅 판단 자체의 Bedrock 비용은 0이다.
    월 환산(req_per_day x 30일)을 포함해 표시하고 계산 결과 dict를 반환."""
    from IPython.display import Markdown, display
    gpu_hourly = PRICING.get("sagemaker", {}).get("ml.g5.2xlarge", 1.515)
    gpu_month = gpu_hourly * 24 * 30
    n = len(baseline)
    def tot(rows, key):
        return sum(r[key] for r in rows)
    b_cost, r_cost = tot(baseline, "cost_usd"), tot(routed, "cost_usd")
    r_sonnet = sum(1 for r in routed if r["backend"] == "sonnet")
    r_haiku = sum(1 for r in routed if r["backend"] == "haiku")
    save_pct = (1 - r_cost / b_cost) * 100 if b_cost else 0.0
    b_month = b_cost / n * req_per_day * 30
    r_month = r_cost / n * req_per_day * 30
    if LANG == "en":
        md = [
            "| metric | all-Sonnet baseline | difficulty-routed |",
            "|---|---:|---:|",
            f"| questions | {n} | {len(routed)} |",
            f"| Sonnet / Haiku calls | {n} / 0 | {r_sonnet} / {r_haiku} |",
            f"| Bedrock tokens (in/out) | {tot(baseline,'input_tokens'):,}/"
            f"{tot(baseline,'output_tokens'):,} | {tot(routed,'input_tokens'):,}/"
            f"{tot(routed,'output_tokens'):,} |",
            f"| Bedrock token cost ({n} questions) | ${b_cost:.4f} | ${r_cost:.4f} |",
            f"| monthly token cost ({req_per_day:,} req/day) "
            f"| ${b_month:,.0f} | ${r_month:,.0f} |",
            "",
            f"**Token-cost savings: {save_pct:,.1f}%** "
            f"(${b_month - r_month:,.0f}/month at {req_per_day:,} req/day)",
            "",
            f"Honest accounting: the routing decisions run on the decision-layer "
            f"GPU (ml.g5.2xlarge, ~${gpu_hourly}/h, ~${gpu_month:,.0f}/month) that "
            f"this architecture already provisions for the agent's D1-D8 "
            f"decisions, so the router itself adds no per-call Bedrock cost. If "
            f"you provisioned the GPU for routing alone, weigh that fixed cost "
            f"against the monthly savings above.",
        ]
    else:
        md = [
            "| 항목 | 전부 Sonnet (baseline) | 난이도 라우팅 |",
            "|---|---:|---:|",
            f"| 질문 수 | {n} | {len(routed)} |",
            f"| Sonnet / Haiku 호출 수 | {n} / 0 | {r_sonnet} / {r_haiku} |",
            f"| Bedrock 토큰 (입력/출력) | {tot(baseline,'input_tokens'):,}/"
            f"{tot(baseline,'output_tokens'):,} | {tot(routed,'input_tokens'):,}/"
            f"{tot(routed,'output_tokens'):,} |",
            f"| Bedrock 토큰 비용 ({n}문항) | ${b_cost:.4f} | ${r_cost:.4f} |",
            f"| 월 환산 토큰 비용 (하루 {req_per_day:,}건 기준) "
            f"| ${b_month:,.0f} | ${r_month:,.0f} |",
            "",
            f"**토큰 비용 절감률: {save_pct:,.1f}%** "
            f"(하루 {req_per_day:,}건 기준 월 ${b_month - r_month:,.0f} 절감)",
            "",
            f"정직한 비용 계산: 라우팅 판단은 이 아키텍처가 agent의 D1-D8 결정을 "
            f"위해 이미 운용 중인 decision layer GPU(ml.g5.2xlarge, 약 "
            f"${gpu_hourly}/시간, 월 약 ${gpu_month:,.0f})에서 수행되므로, 라우터 "
            f"자체의 Bedrock 호출 비용은 0입니다. 라우팅만을 위해 GPU를 새로 "
            f"띄운다면 위 월 절감액과 GPU 고정비를 비교해 판단해야 합니다.",
        ]
    display(Markdown("\n".join(md)))
    return {"baseline_cost": b_cost, "routed_cost": r_cost, "save_pct": save_pct,
            "baseline_month": b_month, "routed_month": r_month,
            "gpu_month": gpu_month}
