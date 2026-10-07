import os
import sys

# 현재 스크립트(analysis 폴더)의 상위 경로를 파이썬 탐색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # 상위 폴더 (프로젝트 루트)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import REFERENCE_PKL_DIR
from src.prepare_everystep import load_reference_everystep
from analysis.verify_mh_consistency import verify_consistency

# 1. MH 데이터 생성 (이때 가중치 fit과 스텝별 점수 계산이 모두 일어남)
df_mh = load_reference_everystep('motionheuristic')

# 2. 결과 검증
pkl_path = os.path.join(REFERENCE_PKL_DIR, 'motionheuristic', 'motionheuristic_data.pkl')
verify_consistency(df_mh, pkl_path)