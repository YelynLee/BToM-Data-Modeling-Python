import time
import os
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

# ==============================================================================
# 1. 실험 설정
# ==============================================================================
NUM_SUBJECTS = 16  # 피험자 수 (반복 횟수)
TEMPERATURE = 0.7  # 다양성을 위해 0.0보다 높게 설정 (0.7 ~ 1.0 권장)
MAX_RETRIES = 5    # Rate Limit 발생 시 재시도 횟수

# API 키 설정
# 🌟 [추가] .env 파일에 있는 변수들을 시스템 환경변수로 등록
load_dotenv()

# 🌟 [수정] os.getenv()를 사용하여 .env에서 키를 안전하게 가져옵니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

client_gpt = OpenAI(api_key=OPENAI_API_KEY)
client_gemini = OpenAI(api_key=GEMINI_API_KEY,
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY,
                         base_url="https://api.deepseek.com")
client_claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ==============================================================================
# 2. 실험 진행
# ==============================================================================
def call_model_api(model_name, system_prompt, user_prompt):
    """
    재시도 로직이 포함된 API 호출 함수
    """
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            # ----------------------------------------
            # CASE 1: OpenAI O1 Series (Reasoning Model)
            # ----------------------------------------
            if "o1-" in model_name: 
                # System Role 불가 -> User 프롬프트와 합침
                # Temperature 파라미터 불가 (Default 1 고정)
                combined_prompt = f"Instructions:\n{system_prompt}\n\nTask:\n{user_prompt}"
                
                response = client_gpt.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": combined_prompt}],
                    # temperature=1, # o1은 지원 안함
                    response_format={"type": "json_object"} # 미지원일 수도...
                )
                return response.choices[0].message.content

            # ----------------------------------------
            # CASE 2: Gemini Series
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
            # CASE 3: GPT Standard (4o, 4-turbo, 3.5)
            # ----------------------------------------
            elif "gpt" in model_name:
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
            # CASE 4: Deepseek Series
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
            # CASE 5: Claude Series
            # ----------------------------------------
            elif "claude" in model_name:
                response = client_claude.messages.create(
                    model=model_name,
                    max_tokens=2000,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=TEMPERATURE
                )
                return response.content[0].text

            # ----------------------------------------
            # CASE 6: 지원하지 않는 모델 방어 로직
            # ----------------------------------------
            else:
                print(f"\n[Error] Unsupported model: {model_name}")
                return None
        
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Quota exceeded" in error_msg:
                wait_time = 20 + (retry_count * 10)
                print(f"\n[Rate Limit] {model_name}: Retrying in {wait_time}s... ({retry_count+1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                retry_count += 1
            else:
                print(f"\n[Error] {model_name}: {e}")
                return None 

    print(f'[Failed] finished: {retry_count}/5')        
    return None