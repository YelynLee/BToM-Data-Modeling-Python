import time
import os
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

# gemini-3.5-flash 지원을 위한 Google GenAI 공식 SDK
from google import genai
from google.genai import types

try:
    from src.llm_requests import parse_response_body, _as_dict
except ImportError:
    from llm_requests import parse_response_body, _as_dict

# ==============================================================================
# 1. 실험 설정
# ==============================================================================
NUM_SUBJECTS = 16  # 피험자 수 (반복 횟수)
TEMPERATURE = 0.7  # 다양성을 위해 0.0보다 높게 설정 (0.7 ~ 1.0 권장)
MAX_RETRIES = 5    # Rate Limit 발생 시 재시도 횟수

# API 키 설정
# .env 파일에 있는 변수들을 시스템 환경변수로 등록
load_dotenv()

# os.getenv()를 사용하여 .env에서 키를 안전하게 가져옵니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

client_gpt = OpenAI(api_key=OPENAI_API_KEY)

client_gemini = OpenAI(api_key=GEMINI_API_KEY,
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
# Gemini 3.5 전용 공식 클라이언트
client_gemini_genai = genai.Client(api_key=GEMINI_API_KEY)

client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY,
                         base_url="https://api.deepseek.com")
client_claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ==============================================================================
# 2. 실험 진행
# ==============================================================================
THINKING_SETTINGS = ("default", "on", "off")


def build_claude_params(model_name, system_prompt, user_prompt, effort=None, thinking="default", temperature=None):
    """
    Claude Messages API 요청 파라미터와 기록용 메타 정보를 만듦 (일반 호출과 Batch 공통).

    thinking:
      - "default": 모든 모델을 기본 설정(추론 기능을 따로 켜지 않음)으로 시작한다는 원칙에 따름
                   Opus   -> "off"와 같음 (thinking disabled, temperature 0.7, max_tokens 8000)
                   Sonnet -> thinking 파라미터 없음, temperature TEMPERATURE(0.7), max_tokens 4000
                   ※ 2026-10-28 이전 코드에서 Opus default는 adaptive thinking이었음
                     (논문 End-step / Every-step Opus 결과와 그 이전 'default' 실행은 adaptive thinking)
      - "on"     : adaptive thinking (별도 옵션). thinking을 켜면 temperature는 1.0이어야 함.
      - "off"    : thinking={"type": "disabled"} 로 명시적으로 끔.
    effort: output_config.effort ("low" / "medium" / "high" / "max").
    temperature: None이면 위 기본값. thinking off에서 1.0으로 맞추고 싶을 때 지정.

    Returns: (params dict, meta dict)
      params의 output_config는 그대로 두며, 일반 호출에서는 SDK 호환을 위해 extra_body로 옮김.
    """
    is_opus = "claude-opus" in model_name
    if is_opus and thinking == "default":
        thinking = "off"   # Opus 기본 = thinking off (결과 폴더·기록도 thinking_off로 통일)
    use_thinking = thinking == "on"

    if use_thinking:
        thinking_cfg, temp, max_tokens = {"type": "adaptive"}, 1.0, 16000
    else:
        thinking_cfg = {"type": "disabled"} if thinking == "off" else None
        temp = temperature if temperature is not None else TEMPERATURE
        max_tokens = 8000 if thinking == "off" else 4000

    params = {
        "model": model_name,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temp,
    }
    if thinking_cfg is not None:
        params["thinking"] = thinking_cfg
    if effort and effort != "none":
        params["output_config"] = {"effort": effort}

    meta = {
        "thinking_setting": thinking if thinking != "default" else "default(none)",
        "temperature": temp,
        "effort": effort or "",
    }
    return params, meta


def parse_claude_message(message, meta):
    """Message 객체 -> {"thinking", "text", "meta"} (utils.process_result_json이 바로 쓰는 형식)."""
    thinking_content, text_content = "", ""
    for block in message.content:
        if block.type == "thinking":
            thinking_content += block.thinking
        elif block.type == "text":
            text_content += block.text
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "max_tokens":
        print(f"\n⚠️ [Warning] max_tokens에 도달해 응답이 잘렸을 수 있습니다.")
    usage = getattr(message, "usage", None)
    meta = dict(meta, output_tokens=getattr(usage, "output_tokens", None),
                input_tokens=getattr(usage, "input_tokens", None),
                thinking_chars=len(thinking_content), stop_reason=stop_reason)
    return {"thinking": thinking_content, "text": text_content, "meta": meta}


def _call_claude(model_name, system_prompt, user_prompt, effort=None, thinking="default", temperature=None):
    """Claude 단건 호출. Returns: {"thinking": str, "text": str, "meta": dict}"""
    params, meta = build_claude_params(model_name, system_prompt, user_prompt, effort, thinking, temperature)
    if "output_config" in params:
        # SDK 버전과 무관하게 동작하도록 extra_body로 전달
        params["extra_body"] = {"output_config": params.pop("output_config")}
    response = client_claude.messages.create(**params)
    return parse_claude_message(response, meta)


# ==============================================================================
# Message Batches API (Claude 전용): 요청마다 독립 처리, 50% 할인, 보통 24시간 이내 완료
# ==============================================================================
def submit_claude_batch(requests):
    """requests: [(custom_id, params dict), ...] -> batch id"""
    batch = client_claude.messages.batches.create(
        requests=[{"custom_id": cid, "params": params} for cid, params in requests])
    return batch.id


def get_claude_batch(batch_id):
    """Returns: (processing_status, request_counts dict)"""
    b = client_claude.messages.batches.retrieve(batch_id)
    rc = b.request_counts
    counts = {k: getattr(rc, k, None) for k in ("processing", "succeeded", "errored", "canceled", "expired")}
    return b.processing_status, counts


def iter_claude_batch_results(batch_id):
    """Yields: (custom_id, "succeeded" | "errored" | "canceled" | "expired", Message 또는 오류 문자열)"""
    for item in client_claude.messages.batches.results(batch_id):
        outcome = item.result
        if outcome.type == "succeeded":
            yield item.custom_id, "succeeded", outcome.message
        else:
            err = getattr(outcome, "error", None)
            yield item.custom_id, outcome.type, str(err) if err is not None else outcome.type


def call_model_api(model_name, system_prompt, user_prompt, effort=None, thinking="default", temperature=None):
    """
    재시도 로직이 포함된 API 호출 함수

    thinking: "default" | "on" | "off"  (현재 Claude에만 적용. 다른 모델은 effort로 추론량을 조절)
    temperature: None이면 기존 기본값(TEMPERATURE). temperature를 받는 모델(gpt-4o, gemini-2.5,
                 deepseek, thinking을 끈 Claude)에만 적용됨.

    Returns: {"thinking": str, "text": str, "meta": {input/output/reasoning 토큰 수 등}} (모든 제공사 공통)
             실패 시 None
    """
    temp = temperature if temperature is not None else TEMPERATURE
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            # ----------------------------------------
            # CASE 1-1: GPT Reasoning Models (gpt-5.4, o4-mini)
            # ----------------------------------------
            if "gpt-5.4" in model_name or "o4" in model_name:

                # 기본 kwargs 설정 (responses.create 규격)
                kwargs = {
                    "model": model_name,
                    "input": [
                        {"role": "developer", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
                
                # effort 파라미터가 명시적으로 전달된 경우 처리
                if effort:
                    # o4-mini 모델인데 none이 들어온 경우 low로 강제 다운그레이드 (에러 방지)
                    if "o4-mini" in model_name and effort == "none":
                        print("\n⚠️ [Warning] o4-mini는 'none' effort를 지원하지 않으므로 'low'로 강제 조정합니다.")
                        kwargs["reasoning"] = {"effort": "low"}
                    else:
                        kwargs["reasoning"] = {"effort": effort}

                # 새로운 responses.create 엔드포인트 사용
                response = client_gpt.responses.create(**kwargs)

                # 원문 + 토큰 수(추론 토큰 포함)를 함께 반환
                out = parse_response_body("openai_responses", response, {"effort": effort or ""})
                if getattr(response, "output_text", None):
                    out["text"] = response.output_text
                return out

            # ----------------------------------------
            # CASE 1-2: GPT Standard (gpt-4o)
            # ----------------------------------------
            elif "gpt-4o" in model_name:
                response = client_gpt.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temp,
                    response_format={"type": "json_object"}
                )
                return parse_response_body("openai_chat", response, {"temperature": temp})

            # ----------------------------------------
            # CASE 2-1: Gemini 3.5 Series (공식 SDK 사용)
            # ----------------------------------------
            elif "gemini-3.5" in model_name:
                # 3.5부터는 temperature 등 샘플링 매개변수가 권장되지 않으므로 생략하고 기본값 사용
                config_kwargs = {
                    "system_instruction": system_prompt,
                    "response_mime_type": "application/json"
                }

                # effort가 전달된 경우 3.5의 새로운 추론 설정인 thinking_level로 매핑
                if effort and effort != "none":
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=effort  # "low", "medium", "high" 등 지원
                    )

                config = types.GenerateContentConfig(**config_kwargs)

                # 새로운 공식 SDK 메서드 사용
                response = client_gemini_genai.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=config
                )
                u = _as_dict(getattr(response, "usage_metadata", None))
                return {"thinking": "", "text": response.text or "",
                        "meta": {"effort": effort or "",
                                 "input_tokens": u.get("prompt_token_count"),
                                 "output_tokens": u.get("candidates_token_count"),
                                 "reasoning_tokens": u.get("thoughts_token_count"),
                                 "thinking_chars": 0}}

            # ----------------------------------------
            # CASE 2-2: 기존 Gemini Series (OpenAI 호환 API 유지)
            # ----------------------------------------
            elif "gemini" in model_name:
                response = client_gemini.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temp,
                    response_format={"type": "json_object"}
                )
                return parse_response_body("gemini_compat", response, {"temperature": temp})
            
            # ----------------------------------------
            # CASE 3: Deepseek Series
            # ----------------------------------------
            elif "deepseek" in model_name:
                # 1. API 호출용 기본 파라미터 구성
                kwargs = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temp
                }
                
                # 2. deepseek-chat(V3)일 때만 JSON 모드 활성화 
                # (deepseek-reasoner는 강제 JSON 모드를 지원하지 않으므로 제외)
                if "chat" in model_name:
                    kwargs["response_format"] = {"type": "json_object"}
                    
                response = client_deepseek.chat.completions.create(**kwargs)
                
                # 최종 답변 + 추론 과정(reasoning_content, deepseek-reasoner) + 토큰 수
                return parse_response_body("deepseek", response, {"temperature": temp})
            
            # ----------------------------------------
            # CASE 4: Claude Series (Opus / Sonnet) - thinking / effort / temperature 제어
            # ----------------------------------------
            elif "claude" in model_name:
                return _call_claude(model_name, system_prompt, user_prompt,
                                    effort=effort, thinking=thinking, temperature=temperature)

            # ----------------------------------------
            # CASE 5: 지원하지 않는 모델 방어 로직
            # ----------------------------------------
            else:
                print(f"\n[Error] Unsupported model: {model_name}")
                return None
        
        except Exception as e:
            error_msg = str(e)

            # 1. 내 호출 한도가 초과된 경우 (길게 대기)
            if "429" in error_msg or "Quota exceeded" in error_msg:
                wait_time = 20 + (retry_count * 10)
                print(f"\n[Rate Limit] {model_name}: Retrying in {wait_time}s... ({retry_count+1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                retry_count += 1

            # 2. 서버 과부하 및 네트워크 연결 오류 - 짧게 점진적으로 대기
            elif any(err in error_msg for err in ["529", "overloaded_error", "Connection error", "ConnectError"]):
                wait_time = 2 ** retry_count  # 1초, 2초, 4초, 8초...
                print(f"\n[Network/Server Issue] {model_name}: Retrying in {wait_time}s... ({retry_count+1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                retry_count += 1

            else:
                print(f"\n[Error] {model_name}: {e}")
                return None 

    print(f'[Failed] finished: {retry_count}/5')        
    return None