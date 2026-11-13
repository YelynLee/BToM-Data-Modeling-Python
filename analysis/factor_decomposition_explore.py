"""탐색적: (1) t = T 재현성 r(Single_T, Single-trunc_T), (2) Every -> Prefix hindsight index 감소의 단계별 분해(형식/정보/문구), (3) Opus run 수준 회귀(정보, 문구, log 토큰; 시나리오x스텝 고정효과). 경로 S는 각 모델 stats zip을 푼 위치로 바꿔 사용."""
import sys, os, glob, numpy as np, pandas as pd, scipy.io as sio, pickle
REPO='/home/claude/btom-data-modeling-python'; sys.path[:0]=[REPO,REPO+'/src',REPO+'/analysis']; os.chdir(REPO)
S='/tmp/claude-0/-home-claude/c0a402c9-dd5b-57dd-b992-6b42f996bdfb/scratchpad'
rng=np.random.default_rng(0); NB=2000
R='results/claude-opus-4-6/vanilla'
# ---------- (1) reliability at t=T: Single vs Single-trunc (identical prompt) ----------
from src.config import get_group_indices
from stats_utils import pooled_r
ids=sorted(s for g in get_group_indices(include_irrational=False) for s in g); c=np.array(ids)-1
a=pickle.load(open(R+'/singlestep/thinking_off/model_data.pkl','rb')); b=pickle.load(open(R+'/singlestep_trunc/thinking_off/model_data.pkl','rb'))
for k in ['bel_inf_mean_norm','des_inf_mean']:
    x,y=a[k][:,c],b[k][:,c]; est=pooled_r(x,y)
    bs=[pooled_r(x[:,i],y[:,i]) for i in (rng.integers(0,len(c),len(c)) for _ in range(NB))]
    print(f'[t=T reliability] r(Single_T, Single-trunc_T) {k}: {est:.2f} [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]')
# ---------- (2) decomposition of hindsight index (scenario level) ----------
def sl(path,m='hindsight_index'):
    d=pd.read_csv(path); d=d[d.metric==m]
    cols=[x for x in d.columns if x not in ('scenario_id','comparison','metric','diff')]
    return d.groupby('scenario_id')[cols].first()
op=sl(S+'/opus_stats/compare_2x2/stepwise_scenario_level.csv'); tr=sl(R+'/compare_trunc/stepwise_scenario_level.csv')
OP=pd.concat([op['Every-step'],op['Every-step-single'],tr['Single-trunc'],op['Prefix-step']],axis=1,keys=['E','S','T','P']).dropna()
ge=sl(S+'/z_ce6a0f09-gemini-2.5-pro_stats/compare_2x2/stepwise_scenario_level.csv'); GE=ge.rename(columns={'Every-step':'E','Every-step-single':'S','Prefix-step':'P'})[['E','S','P']].dropna()
gp=sl(S+'/z_9549d842-gpt-5.4_stats/compare_2x2/stepwise_scenario_level.csv'); GP=gp.rename(columns={'Every-step':'E','Every-step-single':'S','Prefix-step':'P'})[['E','S','P']].dropna()
def decomp(df,steps):
    def f(d):
        tot=d['E'].mean()-d['P'].mean(); out={}
        for name,(x,y) in steps.items():
            v=d[x].mean()-d[y].mean(); out[name]=v; out[name+'_share']=v/tot
        out['total']=tot; return out
    est=f(df); bs=[f(df.iloc[rng.integers(0,len(df),len(df))]) for _ in range(NB)]
    rows=[]
    for k in est: v=[b[k] for b in bs]; rows.append((k,est[k],np.percentile(v,2.5),np.percentile(v,97.5)))
    return pd.DataFrame(rows,columns=['term','est','lo','hi'])
print('\n[decomposition of hindsight index, Every -> Prefix; scenario bootstrap]')
for name,df,steps in [('Opus',OP,{'format (E-S)':('E','S'),'information (S-T)':('S','T'),'wording (T-P)':('T','P')}),
                      ('Gemini',GE,{'format (E-S)':('E','S'),'information+wording (S-P)':('S','P')}),
                      ('GPT-5.4',GP,{'format (E-S)':('E','S'),'information+wording (S-P)':('S','P')})]:
    d=decomp(df,steps); d.insert(0,'model',name); d['n_scen']=len(df); print(d.round(3).to_string(index=False)); d.to_csv(f'{R}/compare_single3/decomp_{name}.csv',index=False)
# ---------- (3) run-level regression with cell fixed effects (Opus Single / Trunc / Prefix) ----------
from src.dataset import df_btom
from src.prepare_everystep import everystep_column
TLEN=df_btom.groupby('scenario_id').time_step.max().to_dict()
def mat(n,f):
    b=sio.loadmat(f'data/{n}/{f}',squeeze_me=True)['belief_marg']; return [np.atleast_2d(b[i]).reshape(3,-1) for i in range(78)]
BT=mat('btom','btom_everystep_beta2.5.mat'); HS=mat('hindsight','hindsight_everystep_beta3.5.mat')
irr={int(s) for s in df_btom.loc[df_btom.is_irrational==1,'scenario_id'].unique()} if 'is_irrational' in df_btom else set()
rows=[]
for lab,path,info,word in [('Single',R+'/singlestep/thinking_off',1,1),('Trunc',R+'/singlestep_trunc/thinking_off',0,1),('Prefix',R+'/prefixstep/thinking_off',0,0)]:
    for i,f in enumerate(sorted(glob.glob(path+'/subject_*.csv'))):
        d=pd.read_csv(f).dropna(subset=['belief_L','belief_M','belief_Empty'])
        for sc,t,l,m,e,tok in zip(d.scenario_id,d.time_step,d.belief_L,d.belief_M,d.belief_Empty,d.output_tokens):
            if sc in irr: continue
            x=np.maximum(np.array([l,m,e])-1,0); s=x.sum()
            if s==0: continue
            x=x/s; j=everystep_column(t,TLEN[sc],BT[sc-1].shape[1]); bt=BT[sc-1][:,j]; dd=HS[sc-1][:,j]-bt; den=dd@dd
            if den<0.05**2: continue
            rows.append(dict(cond=lab,run=i,sc=sc,t=t,f_info=info,f_word=word,proj=(x-bt)@dd/den,tok=tok))
D=pd.DataFrame(rows)
# per-step hindsight projection (1 = Hindsight lambda=2), cell = scenario x step
D['cell']=D.sc.astype(str)+'_'+D.t.astype(str)
D['ltok']=np.log(D.tok)
# within condition x cell centring of log tokens -> compute effect net of item difficulty and condition
D['ltok_w']=D.ltok-D.groupby(['cond','cell']).ltok.transform('mean')
import numpy.linalg as la
def fit(d):
    y=d.proj-d.groupby('cell').proj.transform('mean')
    X=np.column_stack([d.f_info,d.f_word,d.ltok_w]); X=X-pd.DataFrame(X).groupby(d.cell.values).transform('mean').values
    return la.lstsq(X,y.values,rcond=None)[0]
est=fit(D); scs=D.sc.unique(); g={s:D[D.sc==s] for s in scs}; bs=[]
for _ in range(500):
    pick=rng.choice(scs,len(scs)); dd=pd.concat([g[s].assign(cell=g[s].cell+f'_{k}') for k,s in enumerate(pick)]); bs.append(fit(dd))
bs=np.array(bs)
print('\n[run-level regression, Opus Single/Trunc/Prefix, outcome = per-step hindsight projection (0=BToM, 1=Hindsight), scenario x step fixed effects, scenario bootstrap 500]')
for k,n in enumerate(['information (full log = 1)','wording (Every = 1)','log tokens (within condition x cell; per 1 log-unit = 2.7x tokens)']):
    print(f'  {n:70s} {est[k]:+.3f} [{np.percentile(bs[:,k],2.5):+.3f}, {np.percentile(bs[:,k],97.5):+.3f}]')
print('  n rows', len(D), ' n scenarios', len(scs))
# within-cell token-projection correlation per condition
for lab in ['Single','Trunc','Prefix']:
    d=D[D.cond==lab]; y=d.proj-d.groupby('cell').proj.transform('mean')
    print(f'  within-cell r(log tokens, projection) {lab}: {np.corrcoef(d.ltok_w,y)[0,1]:+.3f}  (SD of log tokens within cell {d.ltok_w.std():.3f})')
D.to_csv(R+'/compare_single3/runlevel_projection.csv',index=False)
