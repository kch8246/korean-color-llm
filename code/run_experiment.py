#!/usr/bin/env python3
"""
다중 LLM 실험 실행기 (단일 파일, 추가 패키지 설치 불필요 / Python 3.9+)

문항 풀(item_pool_v0.3.json: 본 실험 436문항 + S3a 확장 32문항 = 468)을 모델 x 문항 x 조건 x 반복으로 호출해 JSONL로 저장합니다.
- 본 실험 436문항만: --all --core / 확장 32문항만: --all --ext (또는 --items S3-051,...). 문항 풀은 v0.3 → v0.2 → v0.1 순으로 자동 탐색합니다.
- 중단 후 이어하기: 같은 명령을 다시 실행하면 성공한 호출은 건너뛰고 실패한 호출만 다시 시도합니다.
- 사전 점검: 실행 전에 각 모델에 짧은 호출을 보내, 호출이 안 되는 모델은 자동으로 제외합니다.
- 선택지 순서: (시드, 문항, 반복 번호)로 결정되므로 모델 간에 같은 순서가 제시되고 재현할 수 있습니다.
- 시스템 프롬프트는 쓰지 않습니다(제로샷).

사용법
  1) 아래 [설정]의 API_KEYS에 키를 붙여넣기 (또는 환경변수)
  2) python run_experiment.py --check                       # 모델 호출 가능 여부만 점검
     python run_experiment.py --pilot --dry-run             # 호출 수 계산만 (비용 없음)
     python run_experiment.py --pilot --reps 5              # 파일럿(57문항) 실행
     python run_experiment.py --all --reps 10 --temps default,0   # 전체 문항, 기본/온도0 두 조건
     python run_experiment.py --sets S1,S3 --models gpt-4o,claude-haiku-5-5
     python run_experiment.py --mock --pilot --reps 2       # 키 없이 파이프라인 시험
  모델 범위: 기본은 --tier core(필수 12개). 규모/세대 대조까지 보려면 --tier core,axis, 전체는 --tier all,
            특정 모델만 보려면 --models 라벨1,라벨2 (tier와 무관하게 전체 목록에서 선택)
  주피터/Colab에서는:  main(["--pilot", "--reps", "5"])
"""

import argparse
import collections
import concurrent.futures as cf
import http.client
import json
import os
import random
import re
import socket
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid

# =====================================================================
# [설정] 여기만 수정하세요
# =====================================================================
API_KEYS = {
    "openai": "",
    "claude": "",
    "gemini": "",      # Google AI Studio
    "deepseek": "",
    "groq": "",
    "perplexity": "",
    "clova": "",       # CLOVA Studio
}
# 위 칸이 비어 있으면 아래 환경변수에서 읽습니다.
ENV_NAMES = {
    "openai": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY", "gemini": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY", "groq": "GROQ_API_KEY", "perplexity": "PERPLEXITY_API_KEY",
    "clova": "CLOVA_API_KEY",
}

# CLOVA Studio 주소. 테스트 앱 키(nv-로 시작)면 "/v3/" 앞에 "/testapp"을 붙이세요.
CLOVA_URL = "https://clovastudio.stream.ntruss.com/v3/chat-completions/{model}"

# 문항 풀 파일 경로. 비워 두면 현재 폴더, 스크립트 폴더, 다운로드/바탕화면/문서 폴더에서 자동으로 찾습니다.
# 예: POOL_PATH = r"C:\item_pool_v0.3.json"
POOL_PATH = ""

SEED = 20261018
TIMEOUT = 90  # 초
MAX_ATTEMPTS = 5
# 서비스별 동시 호출 수 (호출 한도에 걸리면 줄이세요)
PROVIDER_CONCURRENCY = {"openai": 4, "claude": 4, "gemini": 4, "deepseek": 4, "groq": 1,
                        "perplexity": 2, "clova": 2, "mock": 8}


def M(label, provider, model_id, **kw):
    return dict(label=label, provider=provider, id=model_id, **kw)


# 모델 목록. 라벨은 결과 파일에 그대로 기록되니 한 번 정하면 바꾸지 마세요.
# tier(기본 실행 범위):
#   core  : 필수 12개 (기본값). 기준선·최신 주 비교·추론 모드·검색 효과를 보는 최소 구성
#   axis  : 규모/세대 대조용. 가설이 있는 축만 골라 추가 (--tier core,axis 또는 --models)
#   extra : 목록에만 보관. 리뷰 대응이나 부록용 (--tier all 또는 --models)
# 옵션:
#   optional=True : 사전 점검에서 호출이 안 되면 건너뜀 (점검 표에는 skip으로 표시됨)
#   no_temp=True  : temperature를 보내지 않음 (추론형 모델 등)
#   search=True   : 검색 도구를 켠 조건. 검색 형식은 서비스 변경이 잦아 --check로 확인
#   sets=[...]    : 이 모델은 해당 세트만 실행 (검색 조건은 비용 때문에 일부 세트만)
#   min_tokens    : 최소 출력 토큰 (추론/생각 토큰이 큰 모델은 넉넉히. 모자라면 응답이 비어 나옴)
# 모델 ID는 --list-models 결과(2026-10-18)에서 골랐고, --check로 호출 가능 여부를 확인했습니다.
MODELS = [
    # ================= core: 필수 12개 =================
    # 기준선 (1차 논문과 같은 계열)
    M("gpt-4o", "openai", "gpt-4o", tier="core"),
    M("perplexity-sonar", "perplexity", "sonar", preset="fast", no_temp=True, tier="core"),
    M("HCX-005", "clova", "HCX-005", tier="core"),   # CLOVA Studio API는 기본적으로 검색을 쓰지 않음(서비스 'CLOVA X'와 다름)
    # 최신 주 비교
    M("claude-sonnet-5-5", "claude", "claude-sonnet-5-5", tier="core"),
    M("gpt-5.4", "openai", "gpt-5.4", no_temp=True, min_tokens=4096, tier="core"),
    M("gemini-3.8-flash", "gemini", "gemini-3.8-flash", min_tokens=4096, tier="core"),
    M("deepseek-v4-pro", "deepseek", "deepseek-v4-pro", min_tokens=8000, timeout=240, tier="core"),  # deepseek-chat/reasoner와 별칭 관계인지 model_returned로 확인
    M("qwen3.8-27b", "groq", "qwen/qwen3.8-27b", min_tokens=4096, timeout=180, tier="extra"),        # Groq 무료 한도(분당 출력 1000토큰)에 걸려 제외
    # 규모(소형)·추론 모드
    M("claude-haiku-5-5", "claude", "claude-haiku-5-5", tier="core"),
    M("HCX-007", "clova", "HCX-007", reasoning=True, no_temp=True, timeout=240, tier="core"),
    # 검색 on 조건 (같은 모델의 검색 유무 비교용, 일부 세트만)
    M("gpt-4o+websearch", "openai", "gpt-4o", search=True, no_temp=True, sets=["S0", "S3", "S4", "S8"], tier="core"),
    M("claude-sonnet-5-5+websearch", "claude", "claude-sonnet-5-5", search=True, sets=["S0", "S3", "S4", "S8"], tier="core"),

    # ================= axis: 규모/세대 대조 =================
    M("gpt-4o-mini", "openai", "gpt-4o-mini", tier="axis"),
    M("gpt-4-turbo", "openai", "gpt-4-turbo", tier="axis"),                                  # 구세대 대조
    M("gpt-5.4-mini", "openai", "gpt-5.4-mini", no_temp=True, min_tokens=4096, tier="axis"),
    M("claude-haiku-4-5", "claude", "claude-haiku-4-5-20251001", tier="axis"),               # 소형 세대 대조
    M("claude-sonnet-4-6", "claude", "claude-sonnet-4-6", tier="axis"),                      # 중형 세대 대조
    M("gemini-3.5-flash", "gemini", "gemini-3.5-flash", min_tokens=4096, tier="axis"),
    M("gemini-3.5-flash-lite", "gemini", "gemini-3.5-flash-lite", min_tokens=4096, tier="axis"),
    M("gpt-oss-120b", "groq", "openai/gpt-oss-120b", min_tokens=4096, timeout=180, tier="axis"),
    M("gpt-oss-20b", "groq", "openai/gpt-oss-20b", min_tokens=4096, timeout=180, tier="axis"),
    M("HCX-DASH-002", "clova", "HCX-DASH-002", tier="axis"),
    M("perplexity-sonar-pro", "perplexity", "sonar-pro", preset="low", no_temp=True, tier="axis"),

    # ================= extra: 보관용 =================
    # 목록상 가장 최신으로 보이는 모델. sol/luna/terra/astra 등급의 의미는 OpenAI 문서에서 확인 후 사용
    M("gpt-6.1-sol", "openai", "gpt-6.1-sol", no_temp=True, min_tokens=4096, tier="extra"),
    M("claude-opus-5-5", "claude", "claude-opus-5-5", tier="extra"),
    M("claude-fable-5-1", "claude", "claude-fable-5-1", tier="extra"),
    M("gemma-4-31b-it", "gemini", "gemma-4-31b-it", min_tokens=4096, tier="core"),           # 오픈 웨이트 대표(Gemini 키로 호출). 응답이 느림(약 15초)
    M("deepseek-flash", "deepseek", "deepseek-flash", min_tokens=4096, tier="extra"),
    M("deepseek-chat", "deepseek", "deepseek-chat", tier="extra"),                           # 별칭일 수 있음
    M("deepseek-reasoner", "deepseek", "deepseek-reasoner", no_temp=True, min_tokens=8000, timeout=240, tier="extra"),
    # 현재 요금제에서 호출 불가 (결제 설정 후 사용 가능할 수 있음)
    M("gemini-3.1-pro-preview", "gemini", "gemini-3.1-pro-preview", min_tokens=4096, optional=True, tier="extra"),       # 429
    M("gemini-3.8-flash+search", "gemini", "gemini-3.8-flash", search=True, min_tokens=4096, optional=True,
      sets=["S0", "S3", "S4"], tier="extra"),                                                # 429
    M("gemini-2.5-flash", "gemini", "gemini-2.5-flash", min_tokens=4096, optional=True, tier="extra"),                   # 404(신규 사용자 종료)
]
MOCK_MODELS = [M("mock-model", "mock", "mock-1")]
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
FREE_SETS = {"S0", "S9"}  # 자유 서술 그대로 수집(1차 논문 비교용 / 설명 충실성)


# ---------------------------------------------------------------- HTTP
class ApiError(Exception):
    def __init__(self, msg, status=None, retryable=False):
        super().__init__(msg)
        self.status, self.retryable = status, retryable


def short_err(text):
    try:
        d = json.loads(text)
        e = d.get("error")
        if isinstance(e, dict) and e.get("message"):
            return str(e["message"])[:300]
        if isinstance(e, str):
            return e[:300]
        st = d.get("status")
        if isinstance(st, dict) and st.get("message"):
            return str(st["message"])[:300]
    except Exception:  # noqa: BLE001
        pass
    return " ".join(text.split())[:300]


def http_json(url, headers, body=None, timeout=TIMEOUT):
    h = {"Content-Type": "application/json", "User-Agent": "llm-korean-color-runner/1.0", **headers}
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=h, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", errors="replace")
        raise ApiError(f"HTTP {e.code}: {short_err(txt)}", e.code, e.code in (408, 409, 425, 429) or e.code >= 500)
    except urllib.error.URLError as e:
        raise ApiError(f"네트워크 오류: {e.reason}", None, True)
    except (TimeoutError, socket.timeout):
        raise ApiError("타임아웃", None, True)
    except json.JSONDecodeError:
        raise ApiError("응답 JSON 파싱 실패", None, True)
    except (ConnectionError, OSError, http.client.HTTPException) as e:  # 연결 끊김(WinError 10054 등), SSL 오류 등
        raise ApiError(f"연결 오류: {type(e).__name__} {str(e)[:100]}", None, True)


# ---------------------------------------------------------------- 제공자별 호출
# 반환: dict(text, reasoning, tokens_in, tokens_out, citations, search_calls, finish, model_returned)
def make_openai_compat(base_url, token_param):
    def call(m, key, prompt, max_tokens, temp):
        body = {"model": m["id"], "messages": [{"role": "user", "content": prompt}], token_param: max_tokens}
        if temp is not None and not m.get("no_temp"):
            body["temperature"] = temp
        d = http_json(base_url + "/chat/completions", {"Authorization": f"Bearer {key}"}, body, m.get("timeout", TIMEOUT))
        ch = d["choices"][0]
        msg = ch["message"]
        u = d.get("usage") or {}
        return dict(text=(msg.get("content") or "").strip(),
                    reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
                    tokens_in=u.get("prompt_tokens"), tokens_out=u.get("completion_tokens"), citations=[],
                    search_calls=0, finish=ch.get("finish_reason"), model_returned=d.get("model"))
    return call


def call_openai_search(m, key, prompt, max_tokens, temp):
    body = {"model": m["id"], "input": prompt, "tools": [{"type": "web_search"}], "tool_choice": "required", "max_output_tokens": max(max_tokens, 64)}
    d = http_json("https://api.openai.com/v1/responses", {"Authorization": f"Bearer {key}"}, body, m.get("timeout", TIMEOUT))
    texts, cites, n_search = [], [], 0
    for it in d.get("output", []):
        if it.get("type") == "web_search_call":
            n_search += 1
        if it.get("type") == "message":
            for c in it.get("content", []):
                if c.get("type") == "output_text":
                    texts.append(c.get("text", ""))
                    cites += [a.get("url") for a in c.get("annotations", []) if a.get("type") == "url_citation"]
    u = d.get("usage") or {}
    return dict(text="".join(texts).strip(), reasoning=None, tokens_in=u.get("input_tokens"),
                tokens_out=u.get("output_tokens"), citations=cites, search_calls=n_search,
                finish=d.get("status"), model_returned=d.get("model"))


def call_claude(m, key, prompt, max_tokens, temp):
    body = {"model": m["id"], "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if temp is not None and not m.get("no_temp"):
        body["temperature"] = temp
    if m.get("search"):
        body["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
        # 검색을 유도: tool_choice 강제는 웹 검색 서버 도구와 맞지 않아 오류가 날 수 있어 시스템 지시문으로 대신함
        body["system"] = "답하기 전에 반드시 web_search 도구로 한 번 이상 검색해 확인하세요. 최종 답은 사용자가 요구한 형식 그대로 출력하세요."
    d = http_json("https://api.anthropic.com/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"},
                  body, m.get("timeout", TIMEOUT))
    texts, cites, n_search = [], [], 0
    for b in d.get("content", []):
        if b.get("type") == "text":
            texts.append(b.get("text", ""))
            cites += [c.get("url") for c in (b.get("citations") or []) if c.get("url")]
        elif b.get("type") == "server_tool_use":
            n_search += 1
    u = d.get("usage") or {}
    return dict(text="".join(texts).strip(), reasoning=None, tokens_in=u.get("input_tokens"),
                tokens_out=u.get("output_tokens"), citations=cites, search_calls=n_search,
                finish=d.get("stop_reason"), model_returned=d.get("model"))


def call_gemini(m, key, prompt, max_tokens, temp):
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens}}
    if temp is not None and not m.get("no_temp"):
        body["generationConfig"]["temperature"] = temp
    if m.get("search"):
        body["tools"] = [{"google_search": {}}]
    d = http_json(f"https://generativelanguage.googleapis.com/v1beta/models/{m['id']}:generateContent",
                  {"x-goog-api-key": key}, body, m.get("timeout", TIMEOUT))
    if not d.get("candidates"):
        raise ApiError(f"candidates 없음: {json.dumps(d.get('promptFeedback', d), ensure_ascii=False)[:200]}", None, False)
    cand = d["candidates"][0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought")).strip()
    reasoning = "".join(p.get("text", "") for p in parts if p.get("text") and p.get("thought")) or None
    gm = cand.get("groundingMetadata") or {}
    cites = [c["web"]["uri"] for c in gm.get("groundingChunks", []) if c.get("web", {}).get("uri")]
    u = d.get("usageMetadata") or {}
    return dict(text=text, reasoning=reasoning, tokens_in=u.get("promptTokenCount"),
                tokens_out=u.get("candidatesTokenCount"), citations=cites,
                search_calls=len(gm.get("webSearchQueries", [])), finish=cand.get("finishReason"),
                model_returned=d.get("modelVersion"))


def call_perplexity(m, key, prompt, max_tokens, temp):
    body = {"input": prompt, "max_output_tokens": max_tokens}
    if m.get("preset"):
        body["preset"] = m["preset"]
    else:
        body["model"] = m["id"]
    d = http_json("https://api.perplexity.ai/v1/agent", {"Authorization": f"Bearer {key}"}, body, m.get("timeout", TIMEOUT))
    texts, cites = [], []
    for it in d.get("output", []):
        if it.get("type") == "message":
            c = it.get("content", [])
            texts.append(c if isinstance(c, str) else "".join(x.get("text", "") for x in c if isinstance(x, dict)))
        if it.get("type") == "search_results":
            cites += [r.get("url") for r in it.get("results", []) if r.get("url")]
    text = d["output_text"] if isinstance(d.get("output_text"), str) else "".join(texts)
    u = d.get("usage") or {}
    return dict(text=text.strip(), reasoning=None, tokens_in=u.get("input_tokens"), tokens_out=u.get("output_tokens"),
                citations=cites, search_calls=sum(1 for it in d.get("output", []) if it.get("type") == "search_results"),
                finish=d.get("status"), model_returned=d.get("model"))


def call_clova(m, key, prompt, max_tokens, temp):
    body = {"messages": [{"role": "user", "content": prompt}]}
    if not m.get("reasoning"):  # 추론 모델(HCX-007)은 토큰/샘플링 옵션 없이 기본값으로 호출
        body["maxTokens"] = max_tokens
        if temp is not None and not m.get("no_temp"):
            body["temperature"] = temp
    d = http_json(CLOVA_URL.format(model=m["id"]),
                  {"Authorization": f"Bearer {key}", "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex},
                  body, m.get("timeout", TIMEOUT))
    st = d.get("status") or {}
    code = str(st.get("code", "20000"))
    if code != "20000":
        raise ApiError(f"CLOVA status {code}: {st.get('message')}", None, code.startswith(("429", "5")))
    r = d["result"]
    msg = r["message"]
    u = r.get("usage") or {}
    return dict(text=(msg.get("content") or "").strip(), reasoning=msg.get("thinkingContent"),
                tokens_in=u.get("promptTokens"), tokens_out=u.get("completionTokens"), citations=[], search_calls=0,
                finish=r.get("finishReason"), model_returned=m["id"])


def call_mock(m, key, prompt, max_tokens, temp):
    time.sleep(0.01)
    bullets = re.findall(r"^- (.+)$", prompt, flags=re.M)
    if '"answer"' in prompt and bullets:
        text = json.dumps({"answer": bullets[0], "reason": "mock"}, ensure_ascii=False)
    elif '"ranking"' in prompt and bullets:
        text = json.dumps({"ranking": bullets, "reason": "mock"}, ensure_ascii=False)
    elif '"score"' in prompt:
        text = json.dumps({"score": 4, "reason": "mock"})
    else:
        text = "mock 자유 서술 응답"
    return dict(text=text, reasoning=None, tokens_in=len(prompt), tokens_out=len(text), citations=[], search_calls=0,
                finish="stop", model_returned="mock-1")


OPENAI_CHAT = make_openai_compat("https://api.openai.com/v1", "max_completion_tokens")


def get_caller(m):
    p = m["provider"]
    if p == "openai":
        return call_openai_search if m.get("search") else OPENAI_CHAT
    return {"claude": call_claude, "gemini": call_gemini, "perplexity": call_perplexity, "clova": call_clova,
            "deepseek": make_openai_compat("https://api.deepseek.com", "max_tokens"),
            "groq": make_openai_compat("https://api.groq.com/openai/v1", "max_tokens"),
            "mock": call_mock}[p]


def get_key(provider):
    if provider == "mock":
        return "mock"
    return (API_KEYS.get(provider) or os.getenv(ENV_NAMES.get(provider, ""), "")).strip()


# ---------------------------------------------------------------- 프롬프트 구성
def option_order(item, rep):
    opts = list(item["options"])
    mode = item.get("option_order", "fixed")
    rng = random.Random(f"{SEED}|{item['item_id']}|{rep}")
    if mode == "shuffle":
        rng.shuffle(opts)
    elif mode == "shuffle_except_last" and len(opts) > 1:
        head = opts[:-1]
        rng.shuffle(head)
        opts = head + [opts[-1]]
    return opts


def build_prompt(item, rep):
    """(전송할 프롬프트, 제시한 선택지 순서) 반환."""
    if item["set_id"] in FREE_SETS or item["task_type"] == "open":
        return item["prompt"], list(item["options"]) if item["options"] else []
    opts = option_order(item, rep)
    en = item["lang"] == "en"
    p = item["prompt"]
    if item["task_type"] in ("choice", "rank"):
        listing = "\n".join(f"- {o}" for o in opts)
        if en:
            head = "\n\nOptions (order is irrelevant):\n" if item["task_type"] == "choice" else "\n\nItems to sort (shown in random order):\n"
            p += head + listing
        else:
            head = "\n\n선택지(순서는 무관합니다):\n" if item["task_type"] == "choice" else "\n\n정렬할 항목(무작위 순서로 제시):\n"
            p += head + listing
    if item["task_type"] == "choice":
        p += ('\nRespond with only one line of JSON and nothing else: {"answer": "<exactly one of the options>", "reason": "<one sentence>"}'
              if en else '\n다른 텍스트 없이 다음 JSON 한 줄로만 답하세요: {"answer": "<선택지 중 하나를 그대로>", "reason": "<한 문장 근거>"}')
    elif item["task_type"] == "rank":
        p += ('\nRespond with only one line of JSON and nothing else, listing every item from strongest to weakest: {"ranking": ["...", "..."], "reason": "<one sentence>"}'
              if en else '\n다른 텍스트 없이 다음 JSON 한 줄로만 답하세요(모든 항목을 강한 것부터 약한 것 순으로): {"ranking": ["...", "..."], "reason": "<한 문장 근거>"}')
    elif item["task_type"] == "rating":
        p += ('\nRespond with only one line of JSON and nothing else: {"score": <integer 1-7>, "reason": "<one sentence>"}'
              if en else '\n다른 텍스트 없이 다음 JSON 한 줄로만 답하세요: {"score": <1~7 정수>, "reason": "<한 문장 근거>"}')
    return p, opts


# ---------------------------------------------------------------- 응답 파싱
def norm(s):
    s = unicodedata.normalize("NFC", str(s)).strip().strip("\"'`“”‘’[]()")
    return re.sub(r"\s+", " ", s)


def find_json(text, key):
    dec = json.JSONDecoder()
    found = []
    t = text[:20000]
    for i, ch in enumerate(t):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(t[i:])
            except ValueError:
                continue
            if isinstance(obj, dict) and key in obj:
                found.append(obj)
    return found[-1] if found else None


def match_option(ans, options):
    a = norm(ans)
    for o in options:
        if norm(o) == a:
            return o
    cands = [o for o in options if (norm(o) in a) or (len(a) >= 2 and a in norm(o))]
    return cands[0] if len(cands) == 1 else None


def parse_response(item, text, shown_options):
    tt = item["task_type"]
    if item["set_id"] in FREE_SETS or tt == "open":
        return dict(parsed={}, parse_ok=None, parse_method="free")
    opts = item["options"]
    if tt == "choice":
        obj = find_json(text, "answer")
        if obj is not None:
            o = match_option(obj["answer"], opts)
            if o:
                return dict(parsed={"answer": o}, parse_ok=True, parse_method="json")
        hits = [o for o in opts if norm(o) in text]
        if len(hits) == 1:
            return dict(parsed={"answer": hits[0]}, parse_ok=True, parse_method="text_contains")
        return dict(parsed={}, parse_ok=False, parse_method="fail")
    if tt == "rank":
        obj = find_json(text, "ranking")
        if obj is not None and isinstance(obj["ranking"], list):
            mapped = [match_option(x, opts) for x in obj["ranking"]]
            if None not in mapped and sorted(mapped) == sorted(opts):
                return dict(parsed={"ranking": mapped}, parse_ok=True, parse_method="json")
        pos = [(text.find(o), o) for o in opts]
        if all(p >= 0 for p, _ in pos):
            return dict(parsed={"ranking": [o for _, o in sorted(pos)]}, parse_ok=True, parse_method="text_order")
        return dict(parsed={}, parse_ok=False, parse_method="fail")
    if tt == "rating":
        obj = find_json(text, "score")
        if obj is not None:
            try:
                s = int(round(float(obj["score"])))
                if 1 <= s <= 7:
                    return dict(parsed={"score": s}, parse_ok=True, parse_method="json")
            except (TypeError, ValueError):
                pass
        mt = re.findall(r"\b([1-7])\b", text) if len(text) < 60 else []
        if len(mt) == 1:
            return dict(parsed={"score": int(mt[0])}, parse_ok=True, parse_method="regex")
        return dict(parsed={}, parse_ok=False, parse_method="fail")
    return dict(parsed={}, parse_ok=None, parse_method="free")


# ---------------------------------------------------------------- 호출 + 재시도
def call_with_retry(m, prompt, max_tokens, temp):
    key = get_key(m["provider"])
    fn = get_caller(m)
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            t0 = time.time()
            res = fn(m, key, prompt, max_tokens, temp)
            res["latency_s"] = round(time.time() - t0, 2)
            return res, attempt, None
        except ApiError as e:
            last = e
            if not e.retryable or attempt == MAX_ATTEMPTS:
                break
            time.sleep(min(60, 2 ** attempt) + random.random())
        except (KeyError, IndexError, TypeError) as e:
            last = ApiError(f"응답 형식 오류: {type(e).__name__} {e}", None, False)
            break
        except Exception as e:  # noqa: BLE001  예기치 못한 오류도 재시도 후 기록(실행 전체가 중단되지 않게)
            last = ApiError(f"예기치 못한 오류: {type(e).__name__} {str(e)[:100]}", None, True)
            if attempt == MAX_ATTEMPTS:
                break
            time.sleep(min(60, 2 ** attempt) + random.random())
    return None, attempt, last


def preflight(models, mock=False):
    """각 모델에 pong 호출. 사용 가능 모델 리스트와 점검 결과 반환."""
    print("\n[사전 점검] 모델별 짧은 호출 테스트")
    results = []
    lock = threading.Lock()

    def one(m):
        if not get_key(m["provider"]):
            r = dict(label=m["label"], ok=False, detail="키 없음", optional=bool(m.get("optional")))
        else:
            res, _, err = call_with_retry(m, "Reply with exactly one word: pong", max(64, m.get("min_tokens", 0)), None)
            r = dict(label=m["label"], ok=err is None, optional=bool(m.get("optional")),
                     detail=(f"응답: {res['text'][:25]!r} ({res['latency_s']}s)" if err is None else str(err)[:200]))
        with lock:
            results.append(r)

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, models))
    order = {m["label"]: i for i, m in enumerate(models)}
    results.sort(key=lambda r: order[r["label"]])
    for r in results:
        tag = "OK  " if r["ok"] else ("skip" if r["optional"] else "FAIL")
        print(f"  {tag} {r['label']:<30}{r['detail']}")
    ok = {r["label"] for r in results if r["ok"]}
    bad_required = [r["label"] for r in results if not r["ok"] and not r["optional"]]
    if bad_required:
        print(f"  ※ 필수 모델 중 호출 불가: {', '.join(bad_required)} → 제외하고 진행합니다 (키/모델 ID 확인).")
    return [m for m in models if m["label"] in ok], results


# ---------------------------------------------------------------- 실행
def list_available_models(out_dir):
    """키가 있는 서비스별로 현재 사용 가능한 모델 ID 목록을 조회해 출력·저장."""
    found = {}
    for prov in ["openai", "claude", "gemini", "deepseek", "groq"]:
        key = get_key(prov)
        if not key:
            print(f"\n[{prov}] 키 없음 → 건너뜀")
            continue
        try:
            if prov == "openai":
                d = http_json("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"})
                names = sorted(x["id"] for x in d["data"] if x["id"].startswith(("gpt", "o1", "o3", "o4", "o5", "chatgpt")))
            elif prov == "claude":
                d = http_json("https://api.anthropic.com/v1/models?limit=100",
                              {"x-api-key": key, "anthropic-version": "2023-06-01"})
                names = [x["id"] for x in d["data"]]
            elif prov == "gemini":
                d = http_json("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", {"x-goog-api-key": key})
                names = sorted(x["name"].replace("models/", "") for x in d.get("models", [])
                               if "generateContent" in x.get("supportedGenerationMethods", []))
            elif prov == "deepseek":
                d = http_json("https://api.deepseek.com/models", {"Authorization": f"Bearer {key}"})
                names = [x["id"] for x in d["data"]]
            else:  # groq
                d = http_json("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"})
                skip = ("whisper", "tts", "guard", "orpheus", "embed")
                names = sorted(x["id"] for x in d["data"] if not any(s in x["id"].lower() for s in skip))
        except ApiError as e:
            print(f"\n[{prov}] 목록 조회 실패: {str(e)[:200]}")
            continue
        found[prov] = names
        print(f"\n[{prov}] 사용 가능한 모델 {len(names)}개:")
        print("  " + ", ".join(names))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "available_models.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": found}, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {path}")
    print("※ CLOVA Studio와 Perplexity는 목록 API가 없어 조회하지 않습니다.")


def find_pool(explicit):
    dirs = [os.getcwd(), os.path.join(os.getcwd(), "items"), BASE_DIR, os.path.join(BASE_DIR, "..", "items"), os.path.dirname(POOL_PATH) if POOL_PATH else ""]
    name = next((n for n in ("item_pool_v0.3.json", "item_pool_v0.2.json", "item_pool_v0.1.json")
                 if any(d and os.path.isfile(os.path.join(d, n)) for d in dirs)), "item_pool_v0.3.json")
    home = os.path.expanduser("~")
    cands = [explicit, POOL_PATH] + [os.path.join(d, name) for d in dirs if d]
    cands += [os.path.join(home, d, name) for d in ("Downloads", "Desktop", "Documents")]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise SystemExit(
        f"문항 풀 파일({name})을 찾지 못했습니다.\n"
        f"  - 현재 작업 폴더: {os.getcwd()}\n"
        f"  - 해결: 파일을 위 폴더로 옮기거나, 파일 상단 POOL_PATH에 전체 경로를 적거나, --pool 경로를 지정하세요.")


def load_pool(path):
    with open(find_pool(path), encoding="utf-8") as f:
        return json.load(f)["items"]


def select_items(items, args):
    if args.pilot:
        items = [i for i in items if i["pilot"]]
    if getattr(args, "new", False):
        items = [i for i in items if i.get("version_added") == "0.2"]
        if not items:
            raise SystemExit("--new는 v0.2 문항 풀이 필요합니다. 불러온 파일이 item_pool_v0.1.json이 아닌지 확인하고, "
                             "POOL_PATH 또는 --pool을 item_pool_v0.2.json으로 지정하세요.")
    if getattr(args, "core", False):
        items = [i for i in items if i.get("version_added") != "0.3"]   # 본 실험 436문항
    if getattr(args, "ext", False):
        items = [i for i in items if i.get("version_added") == "0.3"]   # S3a 확장 32문항
    if args.sets:
        want = {s.strip() for s in args.sets.split(",")}
        items = [i for i in items if i["set_id"] in want]
    if args.items:
        want = {s.strip() for s in args.items.split(",")}
        items = [i for i in items if i["item_id"] in want]
    return items


def parse_temps(s):
    out = []
    for t in s.split(","):
        t = t.strip().lower()
        out.append(None if t in ("default", "none", "") else float(t))
    return out


def temp_label(t):
    return "default" if t is None else f"t{t:g}"


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False, description="다중 LLM 실험 실행기")
    ap.add_argument("--pool", default="", help="문항 풀 JSON 경로(비우면 자동 탐색)")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "results", "trials.jsonl"))
    ap.add_argument("--all", action="store_true", help="전체 문항 실행 (명시하지 않으면 전체 실험은 시작되지 않음)")
    ap.add_argument("--pilot", action="store_true", help="파일럿 문항만")
    ap.add_argument("--new", action="store_true", help="v0.2에서 추가된 문항만")
    ap.add_argument("--core", action="store_true", help="본 실험 436문항만(v0.3 확장 문항 제외)")
    ap.add_argument("--ext", action="store_true", help="S3a 확장 32문항만(v0.3에서 추가)")
    ap.add_argument("--sets", default="", help="예: S1,S3")
    ap.add_argument("--items", default="", help="문항 ID 쉼표 구분")
    ap.add_argument("--models", default="", help="모델 라벨 쉼표 구분")
    ap.add_argument("--tier", default="core", help="모델 범위: core(기본, 필수 12개) / axis / extra / all (쉼표로 조합, 예: core,axis)")
    ap.add_argument("--reps", type=int, default=0, help="반복 횟수(기본: 파일럿 5, 전체 10)")
    ap.add_argument("--temps", default="default", help="예: default,0")
    ap.add_argument("--tag", default="", help="실행 태그(결과에 기록)")
    ap.add_argument("--rerun-truncated", action="store_true", help="출력 한도에서 잘렸거나 응답이 빈 호출을 다시 실행(이전 행은 남기고 새 행을 추가)")
    ap.add_argument("--token-mult", type=float, default=1.0, help="출력 토큰 한도 배수 (예: 2). 잘린 호출 재실행 시 사용")
    ap.add_argument("--check", action="store_true", help="사전 점검만 하고 종료")
    ap.add_argument("--list-models", action="store_true", help="서비스별 현재 사용 가능한 모델 목록 조회 후 종료")
    ap.add_argument("--dry-run", action="store_true", help="호출 수 계산만")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--mock", action="store_true", help="가짜 모델로 파이프라인 시험")
    args, _unknown = ap.parse_known_args(argv)  # 주피터가 넘기는 -f ...json 같은 인자는 무시

    # 안전장치: 실행할 범위를 명시하지 않으면 아무것도 하지 않음 (실수로 전체 실험이 시작되는 것 방지)
    if args.list_models:
        list_available_models(os.path.dirname(args.out))
        return 0
    if not (args.check or args.dry_run or args.pilot or args.new or args.core or args.ext or args.all or args.sets or args.items or args.mock):
        print("실행 범위가 지정되지 않아 아무 작업도 하지 않았습니다. 예시:")
        print('  main(["--list-models"])                # 서비스별 사용 가능한 모델 목록')
        print('  main(["--check"])                      # 모델 호출 가능 여부 점검')
        print('  main(["--pilot", "--dry-run"])         # 파일럿 호출 수 계산(비용 없음)')
        print('  main(["--pilot", "--reps", "5"])       # 파일럿 실행')
        print('  main(["--all", "--reps", "10"])        # 전체 문항 실행')
        return 0

    if args.mock:
        models = [dict(m) for m in MOCK_MODELS]
    elif args.models:  # 라벨로 직접 지정하면 tier와 무관하게 전체 목록에서 선택
        by_label = {m["label"]: m for m in MODELS}
        want = [x.strip() for x in args.models.split(",") if x.strip()]
        unknown = [w for w in want if w not in by_label]
        if unknown:
            print(f"알 수 없는 모델 라벨: {unknown}\n가능: {', '.join(by_label)}")
            return 2
        models = [dict(by_label[w]) for w in want]
    else:
        tiers = {"core", "axis", "extra"} if args.tier.strip() == "all" else {t.strip() for t in args.tier.split(",")}
        bad = tiers - {"core", "axis", "extra"}
        if bad:
            print(f"알 수 없는 tier: {sorted(bad)} (가능: core, axis, extra, all)")
            return 2
        models = [dict(m) for m in MODELS if m.get("tier") in tiers]
    print(f"모델 구성: {len(models)}개 ({'mock' if args.mock else ('직접 지정' if args.models else 'tier=' + args.tier)})")

    if args.check:
        _, res = preflight(models)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        path = os.path.join(os.path.dirname(args.out), "models_check.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": res}, f, ensure_ascii=False, indent=1)
        print(f"\n점검 결과 저장: {path}")
        return 0

    items = select_items(load_pool(args.pool), args)
    if not items:
        print("선택된 문항이 없습니다.")
        return 2
    reps = args.reps or (5 if args.pilot else 10)
    temps = parse_temps(args.temps)

    # 조건 구성: (모델, 온도, 문항, 반복)
    def conditions(ms):
        for m in ms:
            allowed = m.get("sets")
            for t in temps:
                if t is not None and m.get("no_temp"):
                    continue
                for it in items:
                    if allowed and it["set_id"] not in allowed:
                        continue
                    for r in range(1, reps + 1):
                        yield m, t, it, r

    all_conds = list(conditions(models))
    per_model = collections.Counter(m["label"] for m, *_ in all_conds)
    print(f"문항 {len(items)}개 x 반복 {reps} / 온도 조건 {[temp_label(t) for t in temps]}")
    print(f"총 호출 수(사전 점검 전): {len(all_conds):,}")
    for m in models:
        print(f"  {m['label']:<30}{per_model[m['label']]:>7,}회")
    if args.dry_run:
        by_set = collections.Counter(it["set_id"] for it in items)
        print(f"\n문항 세트별: {dict(by_set)}")
        seen = set()
        for it in items:
            if it["set_id"] in seen:
                continue
            seen.add(it["set_id"])
            p, _ = build_prompt(it, 1)
            print(f"\n--- 샘플 프롬프트 [{it['item_id']}] ---\n{p}")
        return 0

    if not args.no_preflight:
        models, _ = preflight(models)
        if not models:
            print("호출 가능한 모델이 없습니다. 키를 확인하세요.")
            return 1
        alive = {m["label"] for m in models}
        all_conds = [c for c in all_conds if c[0]["label"] in alive]

    # 이어하기: 성공한 호출은 건너뜀
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("error") is None:
                    bad = row.get("finish_reason") in ("length", "max_tokens") or not (row.get("response_text") or "").strip()
                    if args.rerun_truncated:
                        # 같은 trial_id의 뒤쪽 행이 정상이면 완료로 본다(재실행 결과 반영)
                        if bad:
                            done.discard(row["trial_id"])
                            continue
                    done.add(row["trial_id"])

    def trial_id(m, t, it, r):
        return f"{m['label']}|{temp_label(t)}|{it['item_id']}|{r}"

    pending = [c for c in all_conds if trial_id(c[0], c[1], c[2], c[3]) not in done]
    random.Random(SEED).shuffle(pending)  # 부분 결과도 모델·세트에 고르게 분포
    print(f"\n이미 완료 {len(all_conds) - len(pending):,}회 / 남은 호출 {len(pending):,}회 → {args.out}")
    if not pending:
        print("할 일이 없습니다.")
        return 0

    sems = {p: threading.Semaphore(PROVIDER_CONCURRENCY.get(p, 2)) for p in {m["provider"] for m in models}}
    wlock = threading.Lock()
    fails = collections.Counter()  # 모델별 연속 비재시도 오류
    dead = set()
    stats = collections.defaultdict(lambda: collections.Counter())
    counter = {"n": 0}
    t_start = time.time()
    run_tag = args.tag or time.strftime("run_%Y%m%d_%H%M")
    fout = open(args.out, "a", encoding="utf-8")

    def work(c):
        m, t, it, r = c
        label = m["label"]
        if label in dead:
            return
        prompt, shown = build_prompt(it, r)
        base = 3000 if (it["set_id"] in FREE_SETS or it["task_type"] == "open") else 800
        max_tokens = int(max(base, m.get("min_tokens", 0)) * args.token_mult)
        with sems[m["provider"]]:
            res, attempts, err = call_with_retry(m, prompt, max_tokens, t)
        rec = dict(trial_id=trial_id(m, t, it, r), run_tag=run_tag, ts=time.strftime("%Y-%m-%d %H:%M:%S"),
                   model_label=label, provider=m["provider"], model_id=m["id"], search=bool(m.get("search")),
                   temperature=temp_label(t), item_id=it["item_id"], set_id=it["set_id"], subset=it["subset"],
                   group_id=it["group_id"], variant=it["variant"], rep=r, lang=it["lang"], task_type=it["task_type"],
                   prompt_sent=prompt, option_order_used=shown, attempts=attempts)
        if err is None:
            rec.update(response_text=res["text"], reasoning_text=res["reasoning"], tokens_in=res["tokens_in"],
                       tokens_out=res["tokens_out"], latency_s=res["latency_s"], finish_reason=res["finish"],
                       model_returned=res["model_returned"], citations=res["citations"], search_calls=res["search_calls"],
                       error=None, **parse_response(it, res["text"], shown))
        else:
            rec.update(response_text=None, error=str(err)[:500], parse_ok=None, parse_method=None, parsed={})
        with wlock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            counter["n"] += 1
            s = stats[label]
            s["total"] += 1
            if err is None:
                s["ok"] += 1
                fails[label] = 0
                if rec["parse_ok"] is True:
                    s["parsed"] += 1
                if rec["parse_ok"] is False:
                    s["parse_fail"] += 1
            else:
                s["err"] += 1
                if not err.retryable:
                    fails[label] += 1
                    if fails[label] >= 8 and label not in dead:
                        dead.add(label)
                        print(f"\n  ※ {label}: 연속 오류 8회 → 이번 실행에서 제외 (마지막 오류: {str(err)[:120]})")
            if counter["n"] % 25 == 0 or counter["n"] == len(pending):
                el = time.time() - t_start
                eta = el / counter["n"] * (len(pending) - counter["n"])
                print(f"  진행 {counter['n']:,}/{len(pending):,} ({counter['n'] / len(pending):.0%}) "
                      f"경과 {el / 60:.1f}분 남은 예상 {eta / 60:.1f}분", flush=True)

    interrupted = False
    try:
        with cf.ThreadPoolExecutor(max_workers=sum(PROVIDER_CONCURRENCY.get(p, 2) for p in sems)) as ex:
            futs = [ex.submit(work, c) for c in pending]
            for f in cf.as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001  한 호출의 예외로 전체 실행이 멈추지 않게
                    print(f"\n  ※ 작업 예외(건너뜀): {type(e).__name__} {str(e)[:120]}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n중단 요청: 진행 중 호출만 마무리합니다. 같은 명령을 다시 실행하면 이어서 진행됩니다.")
        for f in futs:
            f.cancel()
    finally:
        fout.close()

    print("\n" + "=" * 78)
    print(f"{'모델':<30}{'성공':>6}{'오류':>6}{'파싱성공':>9}{'파싱실패':>9}")
    for label, s in sorted(stats.items()):
        print(f"{label:<30}{s['ok']:>6}{s['err']:>6}{s['parsed']:>9}{s['parse_fail']:>9}")
    print("=" * 78)
    print(f"저장: {args.out}" + ("  (중단됨, 재실행하면 이어서 진행)" if interrupted else ""))
    if any(s["err"] for s in stats.values()):
        print("오류가 있는 호출은 같은 명령을 다시 실행하면 재시도됩니다.")
    return 0


if __name__ == "__main__":
    code = main()
    if "ipykernel" not in sys.modules:
        sys.exit(code)
