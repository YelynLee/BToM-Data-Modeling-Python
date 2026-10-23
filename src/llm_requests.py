"""
제공사별 요청 생성 / 응답 해석 (단건 호출과 Batch가 같은 요청 내용을 쓰도록 한 곳에서 정의).

build_request(...)  -> RequestSpec: 어떤 엔드포인트에 어떤 body를 보낼지 (+ 기록용 메타)
parse_response_body(provider, body) -> {"thinking": str, "text": str, "meta": {...토큰 수...}}

provider 종류
  claude            : Anthropic Messages API (thinking / effort / temperature는 api_client.build_claude_params)
  openai_responses  : OpenAI Responses API (gpt-5.4, o4-mini 등 추론형)
  openai_chat       : OpenAI Chat Completions (gpt-4o)
  gemini            : Gemini 네이티브 generateContent (gemini-2.5-*, gemini-3.5-*)
                      ※ 단건 호출에서 gemini-2.5는 기존처럼 OpenAI 호환 엔드포인트를 쓰고(논문 실험과 동일),
                        Batch에서는 네이티브 형식으로 같은 설정(system instruction, temperature, JSON 출력)을 보냄
  gemini_compat     : Gemini OpenAI 호환 Chat Completions (gemini-2.5 단건 호출)
  deepseek          : DeepSeek Chat Completions (Batch 미지원 -> --workers 병렬 호출)

기록하는 토큰 (meta)
  input_tokens, output_tokens, reasoning_tokens (제공사가 알려주는 경우), thinking_chars
  ※ 판단 하나당 토큰 비교: Every-step은 output_tokens / n_judgements, Prefix-step은 output_tokens 그대로
"""
from dataclasses import dataclass, field

TEMPERATURE_DEFAULT = 0.7


@dataclass
class RequestSpec:
    provider: str
    body: dict
    meta: dict = field(default_factory=dict)
    endpoint: str = ""   # OpenAI batch용 ('/v1/responses' 또는 '/v1/chat/completions')


def provider_of(model_name):
    m = model_name.lower()
    if "claude" in m:
        return "claude"
    if "gpt-5" in m or "o4" in m or "o3" in m:
        return "openai_responses"
    if "gpt" in m:
        return "openai_chat"
    if "gemini" in m:
        return "gemini"
    if "deepseek" in m:
        return "deepseek"
    raise ValueError(f"Unsupported model: {model_name}")


BATCH_PROVIDERS = ("claude", "openai_responses", "openai_chat", "gemini")


def build_request(model_name, system_prompt, user_prompt, effort=None, thinking="default", temperature=None,
                  for_batch=False):
    """기존 api_client의 단건 호출과 같은 설정으로 요청 body를 만듦."""
    p = provider_of(model_name)
    temp = temperature if temperature is not None else TEMPERATURE_DEFAULT

    if p == "claude":
        try:
            from src.api_client import build_claude_params
        except ImportError:
            from api_client import build_claude_params
        params, meta = build_claude_params(model_name, system_prompt, user_prompt, effort, thinking, temperature)
        return RequestSpec("claude", params, meta)

    if p == "openai_responses":
        body = {"model": model_name,
                "input": [{"role": "developer", "content": system_prompt},
                          {"role": "user", "content": user_prompt}]}
        if effort:
            eff = "low" if ("o4-mini" in model_name and effort == "none") else effort
            body["reasoning"] = {"effort": eff}
        return RequestSpec(p, body, {"effort": effort or "", "temperature": None}, "/v1/responses")

    if p == "openai_chat":
        body = {"model": model_name,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_prompt}],
                "temperature": temp, "response_format": {"type": "json_object"}}
        return RequestSpec(p, body, {"temperature": temp}, "/v1/chat/completions")

    if p == "gemini":
        is_35 = "gemini-3" in model_name
        if not for_batch and not is_35:
            # 단건 호출: 논문 실험과 같은 OpenAI 호환 엔드포인트
            body = {"model": model_name,
                    "messages": [{"role": "system", "content": system_prompt},
                                 {"role": "user", "content": user_prompt}],
                    "temperature": temp, "response_format": {"type": "json_object"}}
            return RequestSpec("gemini_compat", body, {"temperature": temp})
        gen = {"responseMimeType": "application/json"}
        if not is_35:
            gen["temperature"] = temp          # 3.5는 샘플링 파라미터를 쓰지 않음(기존 코드와 동일)
        if is_35 and effort and effort != "none":
            gen["thinkingConfig"] = {"thinkingLevel": effort}
        body = {"contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "generationConfig": gen}
        return RequestSpec("gemini", body, {"temperature": None if is_35 else temp, "effort": effort or ""})

    if p == "deepseek":
        body = {"model": model_name,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_prompt}],
                "temperature": temp}
        if "chat" in model_name:
            body["response_format"] = {"type": "json_object"}
        return RequestSpec(p, body, {"temperature": temp})

    raise ValueError(p)


# =========================================================================
# 응답 해석 (SDK 객체와 dict 모두 지원)
# =========================================================================
def _as_dict(obj):
    if obj is None or isinstance(obj, dict):
        return obj or {}
    for attr in ("model_dump", "to_dict", "to_json_dict"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    return dict(getattr(obj, "__dict__", {}))


def _g(d, *keys):
    """중첩 dict에서 첫 번째로 존재하는 키 경로 값 (snake / camel 혼용 대응)."""
    for k in keys:
        cur, ok = d, True
        for part in k.split("."):
            if isinstance(cur, dict) and part in cur and cur[part] is not None:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def parse_response_body(provider, body, base_meta=None):
    """Returns: {"thinking": str, "text": str, "meta": dict}"""
    d = _as_dict(body)
    meta = dict(base_meta or {})
    thinking, text = "", ""

    if provider == "openai_responses":
        for item in d.get("output", []) or []:
            item = _as_dict(item)
            if item.get("type") == "message":
                for c in item.get("content", []) or []:
                    c = _as_dict(c)
                    if c.get("type") in ("output_text", "text"):
                        text += c.get("text", "")
            elif item.get("type") == "reasoning":
                for s in item.get("summary", []) or []:
                    thinking += _as_dict(s).get("text", "")
        if not text:
            text = d.get("output_text", "") or ""
        u = d.get("usage") or {}
        meta.update(input_tokens=_g(u, "input_tokens"), output_tokens=_g(u, "output_tokens"),
                    reasoning_tokens=_g(u, "output_tokens_details.reasoning_tokens"))

    elif provider in ("openai_chat", "gemini_compat", "deepseek"):
        ch = (d.get("choices") or [{}])[0]
        msg = _as_dict(ch.get("message") if isinstance(ch, dict) else _as_dict(ch).get("message"))
        text = msg.get("content") or ""
        thinking = msg.get("reasoning_content") or ""
        u = d.get("usage") or {}
        meta.update(input_tokens=_g(u, "prompt_tokens"), output_tokens=_g(u, "completion_tokens"),
                    reasoning_tokens=_g(u, "completion_tokens_details.reasoning_tokens"))

    elif provider == "gemini":
        cands = d.get("candidates") or []
        if cands:
            content = _as_dict(_as_dict(cands[0]).get("content"))
            for part in content.get("parts", []) or []:
                part = _as_dict(part)
                if part.get("thought"):
                    thinking += part.get("text", "") or ""
                elif part.get("text"):
                    text += part["text"]
        u = _g(d, "usageMetadata", "usage_metadata") or {}
        meta.update(input_tokens=_g(u, "promptTokenCount", "prompt_token_count"),
                    output_tokens=_g(u, "candidatesTokenCount", "candidates_token_count"),
                    reasoning_tokens=_g(u, "thoughtsTokenCount", "thoughts_token_count"))
    else:
        raise ValueError(provider)

    meta["thinking_chars"] = len(thinking)
    return {"thinking": thinking, "text": text, "meta": meta}
