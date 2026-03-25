import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
from config import BASE_RESULTS_DIR, BEHAVIOR_GROUPS
from prepare_everystep import load_btom_everystep

def calc_total_variation(series):
    """시계열 데이터의 총 변동성(Total Variation) 계산"""
    return np.sum(np.abs(np.diff(series)))

def calculate_bias_indicators(model_name, condition):
    print(f"\n🔍 [Eval Indicators] 6대 인지 편향 Raw Data 추출 시작: {model_name} - {condition}")
    
    target_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep")
    data_path = os.path.join(target_dir, "everystep_valid_only.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Valid 데이터가 없습니다. ({data_path})")
        return None
        
    df_llm = pd.read_csv(data_path)
    df_btom = load_btom_everystep()
    
    # LLM과 BToM 데이터 병합
    score_cols = ['desire_K', 'desire_L', 'desire_M', 'belief_L', 'belief_M', 'belief_Empty']
    btom_merge_cols = ['scenario_id', 'time_step'] + score_cols + \
                      ['agent_x', 'agent_y', 'goal2_x', 'goal2_y', 'K_x', 'K_y', 'L_x', 'L_y', 'M_x', 'M_y', 'wall_start_x', 'wall_start_y', 'phase']
    
    # BToM 데이터에 is_irrational 컬럼이 있다면 포함, 없다면 제외하고 병합
    if 'is_irrational' in df_btom.columns:
        btom_merge_cols.append('is_irrational')

    merged = pd.merge(df_llm, df_btom[btom_merge_cols], 
                      on=['scenario_id', 'time_step'], suffixes=('', '_btom'))

    # 0~1 확률 정규화 컬럼 생성 (LLM 1~7점 스케일을 확률로 변환)
    b_cols = ['belief_L', 'belief_M', 'belief_Empty']
    b_cols_btom = ['belief_L_btom', 'belief_M_btom', 'belief_Empty_btom']
    
    # LLM의 1~7점 척도를 0~6점 척도로 Shift (원본 MATLAB과 동일한 로직)
    # 1점을 0% 확률로, 4점을 0.33 확률로 완벽하게 맵핑하기 위함
    llm_shifted = merged[b_cols] - 1

    bel_sum = llm_shifted.sum(axis=1).replace(0, 1.0)
    merged[['prob_L', 'prob_M', 'prob_Empty']] = llm_shifted.div(bel_sum, axis=0)
    
    # BToM은 BToM은 이미 0~1 확률 상태이므로 Shift 없이 그대로 안전 정규화
    bel_sum_btom = merged[b_cols_btom].sum(axis=1).replace(0, 1.0)
    merged[['prob_L_btom', 'prob_M_btom', 'prob_Empty_btom']] = merged[b_cols_btom].div(bel_sum_btom, axis=0)

    results = []
    
    # Subject_id와 Scenario_id의 조합으로 순회 (Raw Data 단위)
    for (subj, sc_id), group_data in merged.groupby(['subject_id', 'scenario_id']):
        group_data = group_data.sort_values('time_step').reset_index(drop=True)
        group_id = group_data['group_id'].iloc[0]
        group_desc = BEHAVIOR_GROUPS.get(group_id, f"Group {group_id}")
        
        t1 = group_data.iloc[0]
        t_end = group_data.iloc[-1]

        # 시나리오의 물리적 특성 추출
        path_length = t_end['time_step']
        start_x, start_y = t1['agent_x'], t1['agent_y']
        wall_x, wall_y = t1['wall_start_x'], t1['wall_start_y']
        is_irrational = t1['is_irrational'] if 'is_irrational' in t1 else 0
        
        # 실제 G2 식별
        g2_x, g2_y = t1['goal2_x'], t1['goal2_y']
        if t1['K_x'] == g2_x and t1['K_y'] == g2_y: actual_g2 = 'K'
        elif t1['L_x'] == g2_x and t1['L_y'] == g2_y: actual_g2 = 'L'
        elif t1['M_x'] == g2_x and t1['M_y'] == g2_y: actual_g2 = 'M'
        else: actual_g2 = 'Empty'
        
        prob_col_g2 = f"prob_{actual_g2}"
        prob_col_g2_btom = f"prob_{actual_g2}_btom"
        
        # -------------------------------------------------------------
        # 0. Baseline Metrics (KL Divergence & RMSE)
        # -------------------------------------------------------------
        kl_divs = []
        desire_sq_errors = []
        for _, row in group_data.iterrows():
            # KL Divergence: BToM(P) || LLM(Q)
            p = [row['prob_L_btom'], row['prob_M_btom'], row['prob_Empty_btom']]
            q = [row['prob_L'], row['prob_M'], row['prob_Empty']]
            # P와 Q에 0이 있으면 entropy 계산 시 inf가 발생할 수 있으므로 미세한 값 추가
            kl_divs.append(entropy(pk=np.array(p)+1e-9, qk=np.array(q)+1e-9))
            
            # RMSE
            sq_err = np.mean([
                (row['desire_K'] - row['desire_K_btom'])**2,
                (row['desire_L'] - row['desire_L_btom'])**2,
                (row['desire_M'] - row['desire_M_btom'])**2
            ])
            desire_sq_errors.append(sq_err)
            
        base_kl_mean = np.mean(kl_divs)
        base_desire_rmse = np.sqrt(np.mean(desire_sq_errors))

        # -------------------------------------------------------------
        # 1. TrueBelief (G2 Present 조건 한정)
        # -------------------------------------------------------------
        tb_llm_belief, tb_btom_prob = np.nan, np.nan
        if actual_g2 != 'Empty':
            tb_llm_belief = t1[prob_col_g2]
            tb_btom_prob = t1[prob_col_g2_btom]
            
        # -------------------------------------------------------------
        # 2. NoCost (특정 Phase 길이를 Effort로 산출)
        # -------------------------------------------------------------
        nc_effort_K, nc_effort_L, nc_effort_M = np.nan, np.nan, np.nan
        
        # 예외 1: Check-Partial (G6, G7)
        if group_id in [6, 7]:
            pass_g1_len = len(group_data[group_data['phase'].str.contains('Pass G1', na=False)])
            nc_effort_L = pass_g1_len
            nc_effort_M = pass_g1_len
            
        # 예외 2: Check-GoBack (G1, G4)
        elif group_id in [1, 4]:
            approach_len = len(group_data[group_data['phase'].str.contains('Approach G1', na=False)])
            stop_len = len(group_data[group_data['phase'].str.contains('Stop', na=False)])
            return_len = len(group_data[group_data['phase'].str.contains('Return G1', na=False)])
            pass_g1_len = len(group_data[group_data['phase'].str.contains('Pass G1', na=False)])
            
            nc_effort_K = approach_len + stop_len + return_len
            nc_effort_M = pass_g1_len + stop_len + return_len
            
        # -------------------------------------------------------------
        # 3. MotionHeuristic (t=1의 거리에 따른 맹목적 가치 부여)
        # -------------------------------------------------------------
        dist_g1 = abs(t1['agent_x'] - t1['K_x']) + abs(t1['agent_y'] - t1['K_y'])
        dist_g2 = abs(t1['agent_x'] - g2_x) + abs(t1['agent_y'] - g2_y)
        mh_dist_diff = dist_g1 - dist_g2
        
        mh_llm_des_K = t1['desire_K']
        mh_btom_des_K = t1['desire_K_btom']

        # -------------------------------------------------------------
        # 4. HindsightBias (특정 구간 간 Delta 추출)
        # -------------------------------------------------------------
        hb_target, hb_llm_pre, hb_llm_post, hb_btom_pre, hb_btom_post = [np.nan] * 5
        
        if group_id not in [3, 5]: # NoCheck 제외
            pass_g1 = group_data[group_data['phase'].str.contains('Pass G1', na=False)]
            see_g2 = group_data[group_data['phase'].str.contains('See', na=False)]
            return_g1 = group_data[group_data['phase'].str.contains('Return G1', na=False)]
            
            if group_id in [1] and not see_g2.empty and not return_g1.empty:
                # Check-GoBack (P)
                hb_target = 'L'
                pre_obs = see_g2.iloc[-1]
                post_obs = return_g1.iloc[0]
            elif not pass_g1.empty and not see_g2.empty:
                # 그 외
                hb_target = 'Empty'
                pre_obs = pass_g1.iloc[-1]
                post_obs = see_g2.iloc[0]
            else:
                pre_obs, post_obs = None, None

            if pre_obs is not None and post_obs is not None:
                hb_llm_pre = pre_obs[f'belief_{hb_target}']
                hb_llm_post = post_obs[f'belief_{hb_target}']
                hb_btom_pre = pre_obs[f'prob_{hb_target}_btom']
                hb_btom_post = post_obs[f'prob_{hb_target}_btom']

        # -------------------------------------------------------------
        # 5. RationalConsistency (Pass G1 구간에서의 Expected Value와 TV)
        # -------------------------------------------------------------
        rc_llm_ev_mean, rc_llm_dk_mean, rc_llm_ev_tv = np.nan, np.nan, np.nan
        rc_btom_ev_mean, rc_btom_dk_mean, rc_btom_ev_tv = np.nan, np.nan, np.nan
        
        if group_id not in [3, 5]:
            pass_g1_data = group_data[group_data['phase'].str.contains('Pass G1', na=False)]
            if not pass_g1_data.empty:
                # LLM EV
                llm_evs = (pass_g1_data['prob_L'] * pass_g1_data['desire_L']) + \
                          (pass_g1_data['prob_M'] * pass_g1_data['desire_M'])
                rc_llm_ev_mean = llm_evs.mean()
                rc_llm_dk_mean = pass_g1_data['desire_K'].mean()
                rc_llm_ev_tv = calc_total_variation(llm_evs.values)
                
                # BToM EV (BToM 확률 * 1~7 스케일 Desire)
                btom_evs = (pass_g1_data['prob_L_btom'] * pass_g1_data['desire_L_btom']) + \
                           (pass_g1_data['prob_M_btom'] * pass_g1_data['desire_M_btom'])
                rc_btom_ev_mean = btom_evs.mean()
                rc_btom_dk_mean = pass_g1_data['desire_K_btom'].mean()
                rc_btom_ev_tv = calc_total_variation(btom_evs.values)

        # -------------------------------------------------------------
        # 6. ZeroSumBias (NoCheck 구간에서의 Desire L, M TV 및 Delta)
        # -------------------------------------------------------------
        zs_llm_tv_L = zs_llm_tv_M = zs_btom_tv_L = zs_btom_tv_M = np.nan
        zs_llm_delta_L = zs_llm_delta_M = zs_btom_delta_L = zs_btom_delta_M = np.nan
        
        if group_id in [3, 5]: # NoCheck(P, A) 한정
            # 1) Total Variation (요동치는 정도)
            zs_llm_tv_L = calc_total_variation(group_data['desire_L'].values)
            zs_llm_tv_M = calc_total_variation(group_data['desire_M'].values)
            zs_btom_tv_L = calc_total_variation(group_data['desire_L_btom'].values)
            zs_btom_tv_M = calc_total_variation(group_data['desire_M_btom'].values)
            
            # 2) Delta Desire (시작점과 끝점의 순 변화량)
            zs_llm_delta_L = t_end['desire_L'] - t1['desire_L']
            zs_llm_delta_M = t_end['desire_M'] - t1['desire_M']
            zs_btom_delta_L = t_end['desire_L_btom'] - t1['desire_L_btom']
            zs_btom_delta_M = t_end['desire_M_btom'] - t1['desire_M_btom']

        # -------------------------------------------------------------
        # 최종 Raw Data 적재
        # -------------------------------------------------------------
        results.append({
            'subject_id': subj, 'scenario_id': sc_id, 'group_id': group_id, 'group_desc': group_desc,
            
            # 물리적 환경 특성
            'path_length': path_length, 'start_x': start_x, 'start_y': start_y, 
            'wall_x': wall_x, 'wall_y': wall_y, 'is_irrational': is_irrational, 'actual_g2': actual_g2,
            
            # 0. Baseline
            'base_kl_mean': base_kl_mean, 'base_desire_rmse': base_desire_rmse,
            
            # 1. TrueBelief
            'tb_llm_belief_t1': tb_llm_belief, 'tb_btom_prob_t1': tb_btom_prob,
            
            # 2. NoCost
            'nc_effort_K': nc_effort_K, 'nc_effort_L': nc_effort_L, 'nc_effort_M': nc_effort_M,
            'nc_llm_des_K_end': t_end['desire_K'], 'nc_llm_des_L_end': t_end['desire_L'], 'nc_llm_des_M_end': t_end['desire_M'],
            
            # 3. MotionHeuristic
            'mh_dist_G1': dist_g1, 'mh_dist_G2': dist_g2, 'mh_dist_diff': mh_dist_diff,
            'mh_llm_des_K_t1': mh_llm_des_K, 'mh_btom_des_K_t1': mh_btom_des_K,
            
            # 4. HindsightBias
            'hb_target': hb_target,
            'hb_llm_pre': hb_llm_pre, 'hb_llm_post': hb_llm_post,
            'hb_btom_pre': hb_btom_pre, 'hb_btom_post': hb_btom_post,
            
            # 5. RationalConsistency
            'rc_llm_ev_mean': rc_llm_ev_mean, 'rc_llm_dk_mean': rc_llm_dk_mean, 'rc_llm_ev_tv': rc_llm_ev_tv,
            'rc_btom_ev_mean': rc_btom_ev_mean, 'rc_btom_dk_mean': rc_btom_dk_mean, 'rc_btom_ev_tv': rc_btom_ev_tv,
            
            # 6. ZeroSumBias
            'zs_llm_tv_L': zs_llm_tv_L, 'zs_llm_tv_M': zs_llm_tv_M,
            'zs_btom_tv_L': zs_btom_tv_L, 'zs_btom_tv_M': zs_btom_tv_M,
            'zs_llm_delta_L': zs_llm_delta_L, 'zs_llm_delta_M': zs_llm_delta_M,
            'zs_btom_delta_L': zs_btom_delta_L, 'zs_btom_delta_M': zs_btom_delta_M
        })
        
    df_results = pd.DataFrame(results)
    
    out_path = os.path.join(target_dir, "bias_indicators.csv")
    df_results.to_csv(out_path, index=False)
    print(f"✅ Raw Data 채점 완료 및 추출 저장됨: {out_path}")
    return df_results

if __name__ == "__main__":
    calculate_bias_indicators("gemini-2.5-flash", "vanilla")