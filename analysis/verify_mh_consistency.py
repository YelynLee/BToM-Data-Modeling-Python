import pandas as pd
import pickle
import numpy as np
import os

def verify_consistency(df_mh_everystep, pkl_path):
    """
    df_mh_everystep: generate_mh_scores를 통해 얻은 최종 데이터프레임
    pkl_path: 비교 대상이 될 motionheuristic_data.pkl 경로
    """
    if not os.path.exists(pkl_path):
        print(f"❌ 검증 실패: {pkl_path} 파일을 찾을 수 없습니다.")
        return

    # 1. pkl 데이터 로드
    with open(pkl_path, 'rb') as f:
        mh_pkl_data = pickle.load(f)
        
    # pkl 내의 des_inf_mean, bel_inf_mean_norm (3, 78) 추출
    pkl_des = mh_pkl_data['des_inf_mean']
    pkl_bel = mh_pkl_data['bel_inf_mean_norm']

    # 2. df_mh_everystep에서 마지막 타임스텝의 데이터만 추출 (Scenario당 마지막 행)
    # 왜냐하면 marginal data는 최종 추론 값을 담고 있기 때문
    idx_last_steps = df_mh_everystep.groupby('scenario_id')['time_step'].idxmax()
    df_last = df_mh_everystep.loc[idx_last_steps].sort_values('scenario_id')

    # 3. 수치 비교 (Tolerance: 1e-6)
    # df_last의 컬럼 순서대로 [K, L, M] / [L, M, Empty]
    mh_des_mat = df_last[['desire_K', 'desire_L', 'desire_M']].values.T
    mh_bel_mat = df_last[['belief_L', 'belief_M', 'belief_Empty']].values.T

    des_diff = np.max(np.abs(mh_des_mat - pkl_des))
    bel_diff = np.max(np.abs(mh_bel_mat - pkl_bel))

    print("-" * 50)
    print("🔍 [검증 결과]")
    print(f"Desire 최대 오차: {des_diff:.10f}")
    print(f"Belief 최대 오차:  {bel_diff:.10f}")

    if des_diff < 1e-6 and bel_diff < 1e-6:
        print("✅ 성공: Marginal 데이터와 .pkl 데이터가 수치적으로 일치합니다!")
    else:
        print("⚠️ 주의: 데이터 간의 차이가 발견되었습니다. 데이터 조립 로직을 다시 확인하세요.")
    print("-" * 50)

    diff_matrix = np.abs(mh_des_mat - pkl_des)
    max_sc_idx = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
    print(f"가장 큰 Desire 오차 발생 시나리오 인덱스: {max_sc_idx[1] + 1}")
    print(f"해당 시나리오의 픽클 값 vs 우리 계산값: {pkl_des[:, max_sc_idx[1]]} vs {mh_des_mat[:, max_sc_idx[1]]}")