"""
탐색적 분석: Spot 2 내용(참 상태) 쪽으로 끌리는 정도를 Spot 2가 보이기 전 / 후 / 끝내 안 보임으로 나눠 계산.
  - truth-leak index: (LLM - BToM)을 (참 one-hot - BToM) 방향으로 사영, 시나리오별 Σnum/Σden. 0 = BToM, 1 = 참 확신
  - 보이기 전(t < t_vis) 값은 미래 정보 누출, 끝내 안 보이는 시나리오는 음성 대조
  - Spot 2가 보인 뒤 BToM / Hindsight / 현재 belief(보인 뒤 = 참) 레퍼런스와의 평균 절대오차
기본 조건은 Opus(Every 논문 16회, Single-trunc, Prefix). 다른 조건은 인자로 추가:
  python analysis/truth_leak_explore.py "Every-step-single=results/claude-opus-4-6/vanilla/singlestep/thinking_off"
출력: results/claude-opus-4-6/vanilla/compare_trunc/truth_leak_index.csv, truth_leak_steps.csv
"""
import sys, os, glob, numpy as np, pandas as pd, scipy.io as sio
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [REPO, REPO + '/src']; os.chdir(REPO)
from src.dataset import df_btom
R = 'results/claude-opus-4-6/vanilla'
COND = {'Every (paper)': R + '/everystep_paper/everystep_valid_only.csv',
        'Single-trunc': R + '/singlestep_trunc/thinking_off',
        'Prefix-step': R + '/prefixstep/thinking_off'}
EXTRA = sys.argv[1:]  # optional label=path pairs (e.g. Single)
for e in EXTRA:
    k, v = e.split('='); COND[k] = v
OPT = ['L', 'M', 'Empty']

def load(path):
    if path.endswith('.csv'):
        d = pd.read_csv(path)
    else:
        d = pd.concat([pd.read_csv(f).assign(subject_id=i) for i, f in enumerate(sorted(glob.glob(path + '/subject_*.csv')))])
    d = d[['subject_id', 'scenario_id', 'time_step'] + [f'belief_{o}' for o in OPT]].dropna()
    m = d.groupby(['scenario_id', 'time_step'])[[f'belief_{o}' for o in OPT]].mean()
    x = np.maximum(m.values - 1, 0); s = x.sum(1, keepdims=True); s[s == 0] = np.nan
    return pd.DataFrame(x / s, index=m.index, columns=OPT)

def mat(name, f):
    b = sio.loadmat(f'data/{name}/{f}', squeeze_me=True)['belief_marg']
    rows = []
    for ns in range(78):
        a = np.atleast_2d(b[ns]); a = a.reshape(3, -1) if a.shape[0] != 3 else a
        rows.append(a)
    return rows
BT = mat('btom', 'btom_everystep_beta2.5.mat'); HS = mat('hindsight', 'hindsight_everystep_beta3.5.mat')
TB = mat('truebelief', 'truebelief_everystep_beta9.0.mat')

traj = df_btom[df_btom.is_irrational == 0] if 'is_irrational' in df_btom else df_btom
info = {}
for sc, g in traj.groupby('scenario_id'):
    T = int(g.time_step.max()); vis = g.loc[g.visible_goal2 == 1, 'time_step']
    tv = int(vis.min()) if len(vis) else None
    truth = TB[sc - 1][:, 0]
    info[sc] = dict(T=T, tvis=tv, truth=truth, group=int(g.group_id.iloc[0]))

def ref(rows, sc, t):
    a = rows[sc - 1]; return a[:, min(t - 1, a.shape[1] - 1)]

def proj(llm, b, target):
    d = target - b; den = d @ d
    return (np.nan, 0.0) if den < 0.05 ** 2 else ((llm - b) @ d, den)

data = {k: load(v) for k, v in COND.items()}
rows = []
for k, df in data.items():
    for (sc, t), v in df.iterrows():
        if sc not in info: continue
        I = info[sc]; b = ref(BT, sc, t); llm = v.values
        cur = I['truth'] if (I['tvis'] is not None and t >= I['tvis']) else b   # agent's current belief proxy
        num_l, den_l = proj(llm, b, I['truth'])
        rows.append(dict(cond=k, sc=sc, t=t, T=I['T'], tvis=I['tvis'], group=I['group'],
                         phase=('pre' if I['tvis'] is None or t < I['tvis'] else 'post'),
                         seen=I['tvis'] is not None, num=num_l, den=den_l,
                         p_truth=llm @ I['truth'], p_truth_btom=b @ I['truth'],
                         err_btom=np.abs(llm - b).mean(), err_cur=np.abs(llm - cur).mean(),
                         err_hs=np.abs(llm - ref(HS, sc, t)).mean(), remain=I['T'] - t))
L = pd.DataFrame(rows)

def scen_index(sub):
    s = sub.groupby('sc').apply(lambda g: g.num.sum() / g.den.sum() if g.den.sum() > 0 else np.nan, include_groups=False)
    return s.dropna()
rng = np.random.default_rng(0)
def boot(s, n=2000):
    v = s.values; return np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)], [2.5, 97.5])
def pperm(d, n=10000):
    d = d[~np.isnan(d)]; s = rng.choice([-1, 1], (n, len(d))); null = (s * d).mean(1)
    return (np.sum(np.abs(null) >= abs(d.mean())) + 1) / (n + 1)

out = []
print('\n=== Truth-leak index (0 = BToM, 1 = certain of true Spot-2 content) ===')
seg = {'pre-vis (seen later, t < t_vis)': lambda L: L[(L.phase == 'pre') & L.seen],
       'post-vis (t >= t_vis)': lambda L: L[L.phase == 'post'],
       'never seen (No-check; negative control)': lambda L: L[~L.seen]}
idx = {}
for name, f in seg.items():
    for k in COND:
        s = scen_index(f(L[L.cond == k])); idx[(name, k)] = s
        lo, hi = boot(s) if len(s) > 2 else (np.nan, np.nan)
        print(f'{name:42s} {k:16s} {s.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]  n_scen={len(s)}')
        out.append(dict(segment=name, cond=k, est=s.mean(), lo=lo, hi=hi, n=len(s)))
print('\n--- paired differences (scenario sign-flip) ---')
conds = list(COND)
for name in seg:
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            a, b = idx[(name, conds[i])], idx[(name, conds[j])]; d = (a - b).dropna()
            if len(d) > 2:
                lo, hi = boot(d); print(f'{name:42s} {conds[i]} - {conds[j]}: {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] p={pperm(d.values):.4f}')

print('\n=== Mean abs error vs references, post-vis steps (lower = closer) ===')
for k in COND:
    sub = L[(L.cond == k) & (L.phase == 'post')].groupby('sc')[['err_btom', 'err_hs', 'err_cur']].mean()
    print(f'{k:16s} BToM {sub.err_btom.mean():.3f} | Hindsight {sub.err_hs.mean():.3f} | Current-belief(truth) {sub.err_cur.mean():.3f}')

print('\n=== Pre-vis leak vs remaining future steps (T - t), seen scenarios ===')
for k in COND:
    sub = L[(L.cond == k) & (L.phase == 'pre') & L.seen].copy()
    sub['leak'] = sub.p_truth - sub.p_truth_btom
    sub['steps_to_vis'] = sub.tvis - sub.t
    from scipy.stats import spearmanr
    # within-scenario rank correlation, averaged
    rs = [spearmanr(g.steps_to_vis, g.leak)[0] for _, g in sub.groupby('sc') if g.steps_to_vis.nunique() > 2]
    rs = pd.Series(rs).dropna()
    print(f'{k:16s} mean leak(p_truth - BToM) {sub.leak.mean():+.3f}; within-scen rho(steps until visible, leak) {rs.mean():+.2f} (n={len(rs)})')
pd.DataFrame(out).to_csv(R + '/compare_trunc/truth_leak_index.csv', index=False)
L.to_csv(R + '/compare_trunc/truth_leak_steps.csv', index=False)
