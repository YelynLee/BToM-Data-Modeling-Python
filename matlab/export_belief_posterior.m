%% export_belief_posterior.m
% λ(lambda_HB)를 Python에서 연속적으로 바꿔 보기 위한 사후확률 내보내기.
%
% Hindsight 모델은 smoothing 단계에서 초기 belief 후보 b 의 로그 점수에
% lambda_HB * log b(참 세계) 를 더한 것뿐이므로 (T >= t_vis 일 때),
% 한 번 계산한 사후확률 P(b | x_1:T) 를 Python에서 b(참)^Δλ 로 다시 가중하면
% 임의의 λ에 대한 예측이 정확히 나온다 (β = 2.5에서 검증: 기존 결과와 최대 차이 1e-5).
%
% 사용법 (BToM-MATLAB-Resource 폴더 맨 위에서):
%   beta_list = [2.5 3.5];          % 내보낼 β 목록 (β 민감도용이면 0.5:0.5:10)
%   export_belief_posterior
%
% 필요: data/btom/solve_momdps (MOMDP 해; β와 무관하므로 한 번만 풀면 됨)
% 출력: data/hindsight/posterior_b<β>.mat  (시나리오 78개, 수 MB)
%   logp{ns}  : (n_point x T) 시점 T까지의 경로가 주어졌을 때 초기 belief 점들의 로그 사후확률 (보상 합산)
%   points{ns}: (n_point x n_world) 초기 belief 점
%   wi(ns)    : 참 세계 인덱스, t_vis(ns): Spot 2가 처음 보이는 시점
%   lambda_used: 이 파일을 만들 때 쓴 lambda_HB (Python에서 되돌림)
%   desire_marg{ns}: (3 x T) desire 주변분포 (λ와 무관)

addpath([pwd '/code']);
if ~exist('beta_list','var'), beta_list = 2.5; end

for beta_score = beta_list
  lambda_HB = 0;     % BToM 그대로 계산 (Python에서 임의의 λ로 바꿈)
  outfile = sprintf('data/hindsight/posterior_b%1.1f.mat', beta_score);
  if exist(outfile,'file')
    fprintf('skip %s\n', outfile); continue;
  end
  fprintf('beta = %1.1f\n', beta_score);

  if ~exist('data/btom/solve_momdps','dir')
    btom_solve_momdps;               % 한 번만 필요 (β와 무관)
  end
  btom_score_scenarios;              % data/hindsight/score_scenarios 에 덮어씀
  btom_results;                      % workspace에 smooth_joint_post, belief_sub, reward_marg 생성

  load('data/hindsight/t_vis_data.mat','t_vis_array');
  n_s = length(scenario);
  logp = cell(1,n_s); points = cell(1,n_s); desire_marg = cell(1,n_s);
  wi = zeros(1,n_s); t_vis = t_vis_array(:)';
  for ns = 1:n_s
    T = length(scenario{ns}.path);
    points{ns} = belief_sub{ns}{1};
    lp = zeros(size(points{ns},1), T);
    for t = 1:T
      S = smooth_joint_post{ns}{t}{1};          % (n_goal_reward x n_point), 로그
      lp(:,t) = logsumexp(S, 1)';                % 보상에 대해 합산
    end
    logp{ns} = lp;
    wi(ns) = scenario{ns}.condition(2);
    desire_marg{ns} = reward_marg{ns};
  end
  lambda_used = lambda_HB;
  save(outfile, 'logp','points','wi','t_vis','lambda_used','beta_score','desire_marg');
  fprintf('saved %s\n', outfile);
end
