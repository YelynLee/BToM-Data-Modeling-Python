% 함수 맨 마지막 파라미터에 t_vis 추가
function [b_sub_unique,b_ind,fwd,bwd,filter,smooth,trans_unique,o_ind,a_ind] = ...
  score_momdp_path(c_ind_seq,w_ind0,b_sub0,b_sub0_prior,Q,V,beta,s_dim,w_trans,c_trans,obs_dist,b_sub_to_g_sub,g_ind_to_b_ind,b_precision, t_vis)

% ... (초기화 및 Forward Loop 코드는 원본 그대로 유지) ...

%% backward loop

% lambda_HB: 편향의 강도를 결정하는 파라미터 (0이면 정상 BToM, 클수록 편향 심화)
lambda_HB = 2.0; 

for T=path_len:(-1):1
  
  bwd{T} = cell(1,T);
  smooth{T} = cell(1,T);
  
  % 🌟 파이썬에서 넘겨받은 t_vis를 바로 사용하여 편향 발동 여부 결정
  % T(현재 관찰 시점)가 t_vis(시야 확보 시점)에 도달하거나 넘었으면 True
  apply_hindsight = (T >= t_vis);
  
  % initialization
  bwd{T}{T} = zeros(size(fwd{T}));
  smooth{T}{T} = fwd{T} + bwd{T}{T};

  for t=(T-1):(-1):1
    n_b = size(b_sub_unique{t},2);
    bwd{T}{t} = zeros(1,n_b);
    for bi=1:n_b
      bi_index = (b_prev{t} == bi);
      if any(bi_index)
          bwd{T}{t}(bi) = logsumexp(m2v(trans{t}(bi_index)) + m2v(bwd{T}{t+1}(b_ind{t+1}(bi_index))));
      else
          bwd{T}{t}(bi) = -inf;
      end
    end
    
    if apply_hindsight
        % b_sub_unique{t}는 [n_world x n_b] 크기의 행렬.
        % w_ind0는 실제 진실인 세계(True World)의 인덱스.
        % 에이전트의 각 후보 믿음 상태(bi)가 진실(w_ind0)을 얼마나 믿고 있는지 추출 (0 ~ 1)
        agent_belief_in_truth = b_sub_unique{t}(w_ind0, :);
        hb_penalty = lambda_HB * log(agent_belief_in_truth + 1e-10);
        
        % Hindsight 발동: 과거(t)의 에이전트도 진실을 알았어야 한다고 왜곡함
        smooth{T}{t} = fwd{t} + bwd{T}{t} + hb_penalty;
    else
        % Hindsight 미발동: 진실을 모르므로 순수 BToM의 정상적인 회고를 수행함
        smooth{T}{t} = fwd{t} + bwd{T}{t};
    end
    
  end % for t=T:(-1):1
end % for T=path_len:(-1):1

% ... (아래의 policy_b_sub 함수 등은 원본 그대로 유지) ...