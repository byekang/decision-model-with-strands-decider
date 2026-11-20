# Workshop: Decision Model with Strands Decider / 금융 상담 Agent의 Decision Layer 실습

**EN** — Build a Korean banking-support agent with LangGraph, then replace its internal decision layer with **strands-decider E4B** — a dedicated, calibrated decision model — and measure speed, cost and quality against LLM-based decisions (Bedrock Claude Haiku). Notebooks come in two languages with a filename prefix: **`[EN]_`** (English) and **`[KR]_`** (Korean). Each notebook also ships an executed reference copy `*_solution.ipynb`. The strands-decider server (:8004, `StrandsAgents/strands-decider-E4B-gemma4-v1-2610`) must be running; in the hosted workshop it runs on the notebook instance's own GPU (ml.g5.2xlarge) — no separate server, see `setup/`. `[EN]_` notebooks use English customer utterances end-to-end; `[KR]_` notebooks use Korean.

**KR** — 한국어 금융 상담 agent를 LangGraph로 직접 조립하고, 판단(decision) 계층을 LLM 판단(Bedrock Claude Haiku)에서 전용 decision model인 **strands-decider E4B**로 교체해 속도·비용·품질을 실측하는 실습입니다. 노트북은 파일명 prefix로 두 언어를 제공합니다: **`[KR]_`**(한국어), **`[EN]_`**(영어). 각 노트북에는 실행 완료본 `*_solution.ipynb`가 함께 있습니다. strands-decider 서버(:8004, `StrandsAgents/strands-decider-E4B-gemma4-v1-2610`)가 떠 있어야 하며, 호스팅 workshop에서는 노트북 인스턴스(ml.g5.2xlarge)의 GPU에서 직접 돕니다 — 별도 서버 없음 (`setup/` 참조).

## Contents / 대목차

| Folder | Notebooks (EN / KR) | Topic |
|---|---|---|
| `10_agent_foundation/` | `[EN]_01_build_agent.ipynb` / `[KR]_01_build_agent.ipynb` | Build the agent step by step with LangGraph; staged graph visualization; one standalone decision on Haiku vs strands-decider, calibrated confidence, the `/v1/systemone` wire format / LangGraph로 agent를 한 조각씩 조립, 단계별 그래프 시각화, calibrated confidence와 wire format |
| `20_text_scenarios/` | `[EN]_02_scenario1_clarify` / `[KR]_02...` | Vague inquiry -> clarification loop / 모호한 문의 -> 되묻기 loop |
| | `[EN]_03_scenario2_retry` / `[KR]_03...` | Tool system error -> retry loop / tool 오류 -> 대체 tool 재시도 loop |
| | `[EN]_04_scenario3_risk_kyc` / `[KR]_04...` | Fraud flag -> KYC sub-flow + guardrail probing on both backends / 이상거래 flag -> KYC + guardrail 경계 체험 |
| `30_other_deciders/` | `[EN]_05_bedrock_intelligent_routing` / `[KR]_05...` | Bedrock Intelligent Prompt Routing (Nova live lab, custom router, $1/1,000 req) |
| | `[EN]_06_laya` / `[KR]_06...` | Laya — open-source multilingual System 1 engine (322M), native `Router` + 20-line adapter |
| `40_cost_routing/` | `[EN]_07_cost_routing` / `[KR]_07...` | Difficulty routing: the decision model routes simple questions to Haiku, complex to Sonnet — measured Bedrock token-cost savings / 난이도 라우팅(simple->Haiku, complex->Sonnet)으로 토큰 비용 절감 실측 |
| `50_rag_pipeline/` | `[EN]_08_rag_gates` / `[KR]_08...` | The three hidden judgments in RAG (retrieval gate / chunk filter / groundedness, cf. Adaptive-RAG / CRAG / Self-RAG) on Haiku vs the decision model — measured per-query judgment overhead drop / RAG의 판단 3곳(검색 필요성·문서 필터·근거 검증)을 교체해 query당 판단 오버헤드 단축 실측 |

Scenario notebooks follow the same shape: show the scenario's path on the full graph -> run with Haiku (STEP-by-STEP output) -> run with StrandsDecider -> markdown comparison table. / 시나리오 노트북 공통 구성: 전체 그래프에서 경로 표시 -> Haiku 실행 -> StrandsDecider 실행 -> 비교표.

> Perspective: the **decision layer** (strands-decider in these labs) *removes* the decision LLM call; **Intelligent Prompt Routing** picks *which model* a generation call goes to — complements, not substitutes. / **decision layer**는 판단용 LLM 호출 자체를 제거하고, **IPR**은 생성 호출의 모델을 골라줍니다 — 대체재가 아니라 보완재. Notebook 07 shows the decision layer can also *route* generation traffic between frontier models of different cost tiers. / notebook 07은 decision layer가 비용 등급이 다른 frontier 모델 간 생성 트래픽 라우팅까지 담당할 수 있음을 보여줍니다.

## Measured results / 핵심 실측

**Korean decision quality** (160 conversations, D1 guardrail / D2 12-way routing, companion decider-quality bench):
strands-decider **E4B 0.938 / 0.729** (D1 / D2 lenient) — above a general-purpose 8B VLM (0.913 / 0.707, reference) at roughly half the effective size, and far above its own 2B sibling (0.613 / 0.593) and Laya multilingual 322M (0.419 / 0.379). A dedicated, *calibrated* decision model closes the quality gap that small zero-shot models usually show. / 전용 decision model인 E4B가 일반 8B VLM보다 높은 한국어 판단 품질을 보이며, 소형 모델(2B/322M)과의 격차를 뒤집습니다. confidence가 보정(calibrated)되어 있어 임계값 운용이 가능한 것이 특징입니다.

## Kernel dependencies / 커널 의존성

Install with `pip install -r requirements.txt`. **anthropic is version-pinned** — newer SDKs send an API version header that Bedrock rejects (400 "Invalid API version"). / `requirements.txt`의 고정 버전을 사용하세요. 최신 anthropic SDK는 Bedrock이 거부하는 API version 헤더를 보냅니다.

## Shared code / 공용 코드

- `ws_common.py` : graph building/drawing (`draw_graph`, `draw_build_stage`), streamed STEP output (`run_verbose`), comparison tables (`compare`), routing/RAG helpers. Language of helper output and graph captions follows **`DMWS_LANG`** (`ko` default, `en`) — the `[EN]_` notebooks set it in their first cell. / helper 출력·그래프 캡션 언어는 `DMWS_LANG` 환경변수(`ko` 기본, `en`)를 따르며 `[EN]_` 노트북 첫 셀에서 설정됩니다.
- `agent/` : the LangGraph agent (state, nodes, tools, prompts) and the two decision backends (`HaikuDecision`, `SystemOneDecision`).
- `setup/` : `bootstrap.sh` (one-time venv + ~16.5GB model download) and `onstart.sh` (kernel registration + strands-decider server launch on :8004), used by the CloudFormation lifecycle configs.
