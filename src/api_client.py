import time
import os
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

# gemini-3.5-flash 지원을 위한 Google GenAI 공식 SDK
from google import genai
from google.genai import types

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
def call_model_api(model_name, system_prompt, user_prompt, effort=None):
    """
    재시도 로직이 포함된 API 호출 함수
    """
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

                return response.output_text

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
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content

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
                return response.text

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
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            
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
                    "temperature": TEMPERATURE
                }
                
                # 2. deepseek-chat(V3)일 때만 JSON 모드 활성화 
                # (deepseek-reasoner는 강제 JSON 모드를 지원하지 않으므로 제외)
                if "chat" in model_name:
                    kwargs["response_format"] = {"type": "json_object"}
                    
                response = client_deepseek.chat.completions.create(**kwargs)
                
                # 결과 반환 (추론 과정은 무시하고 최종 답변만 반환)
                return response.choices[0].message.content
            
            # ----------------------------------------
            # CASE 4-1: Claude Reasoning Series
            # ----------------------------------------
            elif "claude-opus" in model_name:
                # 1. thinking 토큰 용량을 고려하여 max_tokens를 16000 이상으로 넉넉하게 설정합니다.
                # 2. adaptive thinking 모드 활성화 시 temperature는 반드시 1.0 이어야 합니다.
                thinking_config = {"type": "adaptive"}
                
                response = client_claude.messages.create(
                    model=model_name,
                    max_tokens=16000,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=1.0,
                    thinking=thinking_config
                )
                
                # response.content 배열을 돌면서 각각의 블록을 추출합니다.
                thinking_content = ""
                text_content = ""
                
                for block in response.content:
                    if block.type == "thinking":
                        thinking_content += block.thinking
                    elif block.type == "text":
                        text_content += block.text

                # utils.py에서 두 정보를 한 번에 매핑할 수 있도록 딕셔너리로 반환합니다.
                return {
                    "thinking": thinking_content,
                    "text": text_content
                }
            
            # ----------------------------------------
            # CASE 4-2: Claude Series
            # ----------------------------------------
            elif "claude-sonnet" in model_name:                
                response = client_claude.messages.create(
                    model=model_name,
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=TEMPERATURE
                )
                return response.content[0].text

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