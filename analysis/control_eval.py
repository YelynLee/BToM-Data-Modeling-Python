"""
Control 실험 채점.

main_experiment.py --mode control 결과(subject_*.csv)를 읽어,
행마다 저장된 probe 시점으로 정답을 다시 계산하고 문항별로 채점함.
(정답 정의는 src/control_items.py 하나에서 관리되며 프롬프트와 공유됨)

이전 버전(9문항 v1, 13문항 v2) 결과도 현재 번호(v3)로 변환해 채점함 (없는 문항은 건너뜀).

출력 (결과 폴더 안):
  control_scored.csv   : 행(피험자 x 시나리오)별 응답, 정답, 정오, 오차, q12 해석 선택
  control_summary.csv  : 문항별 정확도 / 평균 절대오차 / q12 해석 선택 비율 (전체 + 행동 그룹별)
  control_ci.csv       : 위 지표의 95% 시나리오 bootstrap CI (행동 그룹별 층화)
                         --tom_link: 시나리오별 사실 문항 정확도 vs End-step |LLM - 사람| Spearman 상관

사용 예)
  python analysis/control_eval.py --model claude-opus-4-6 --condition vanilla --variant thinking_off --tom_link
  python analysis/control_eval.py --input results/claude-opus-4-6/vanilla/control/thinking_off
"""
import os
import sys
import glob
import json
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.append(p)

from src.dataset import df_btom
from src.config import BASE_RESULTS_DIR, BEHAVIOR_GROUPS
from src.control_items import (ground_truth, score_response, upgrade_legacy_row, result_version,
                                ITEMS, NUMERIC_ITEMS, PROBE_COLS)
sys.path.append(os.path.join(ROOT, "analysis"))
from stats_utils import stratified_boot_indices, percentile_ci
import numpy as np
import pickle
from scipy.stats import spearmanr

# 좌표·가시성·이동 등 사실 이해 문항 (질문 해석 q12는 따로 집계)
FACT_ITEMS = [k for k in ITEMS if not k.startswith('q12')]
MEANING = ('q12a_question_meaning', 'q12b_question_meaning')
SHORTEST = ('q7_shortest_moves_spot1', 'q8_shortest_moves_spot2')
CHOICE_NAMES = {'a': '정답', 'b': 'current 해석', 'c': 'q12a: 실제 트럭 / q12b: hindsight 해석', 'd': '기타'}


def load_results(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "subject_*.csv")))
    if not files:
        raise FileNotFoundError(f"No subject_*.csv in {input_dir}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["subject_id"] = int(os.path.basename(f).split("_")[1].split(".")[0])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def score_all(df):
    scen = {sc: g for sc, g in df_btom.groupby('scenario_id')}
    group_of = df_btom.groupby('scenario_id')['group_id'].first()
    rows = []
    for r in df.to_dict('records'):
        r = upgrade_legacy_row(r)  # 이전 버전(9문항 / 13문항) 결과는 현재 번호로 변환
        base = {'subject_id': r['subject_id'], 'scenario_id': int(r['scenario_id']),
                'group_id': int(group_of[int(r['scenario_id'])])}
        if isinstance(r.get('error'), str) or pd.isna(r.get('probe_t5a')):
            base['parse_error'] = True
            rows.append(base)
            continue
        probes = {c: r[c] for c in PROBE_COLS if c in r and not pd.isna(r[c])}
        gt = ground_truth(scen[base['scenario_id']], probes)
        sc = score_response(r, gt)
        out = dict(base, parse_error=False, **probes)
        for k in ITEMS:
            if gt.get(k) is None or k not in r:
                continue  # 이전 버전 결과에는 없는 문항
            out[f'{k}_pred'] = r.get(k)
            out[f'{k}_gt'] = json.dumps(gt[k]) if isinstance(gt[k], dict) else gt[k]
        for k in SHORTEST:
            if k in r:
                out[f'{k}_gt_unbounded'] = gt[f'{k}_unbounded']
        out.update(sc)
        for c in ('thinking_setting', 'temperature', 'effort', 'output_tokens', 'thinking_chars'):
            if c in r:
                out[c] = r[c]
        rows.append(out)
    return pd.DataFrame(rows)


def summarize(scored):
    ok = scored[~scored['parse_error']]
    group_names = {gid: name for gid, name in BEHAVIOR_GROUPS.items()}

    def block(d, label):
        rec = {'scope': label, 'n_rows': len(d)}
        for k in ITEMS:
            if f'{k}_gt' in d.columns:
                rec[f'{k}_acc'] = d.loc[d[f'{k}_gt'].notna(), f'{k}_correct'].mean()
        # 질문 해석 문항: 어떤 해석을 골랐는지 비율 (CHOICE_NAMES 참고)
        for k in MEANING:
            if f'{k}_choice' in d.columns:
                for cid in 'abcd':
                    rec[f'{k}_choice_{cid}'] = (d[f'{k}_choice'] == cid).mean()
        for k in NUMERIC_ITEMS:
            if f'{k}_abs_err' in d.columns:
                rec[f'{k}_mae'] = pd.to_numeric(d[f'{k}_abs_err'], errors='coerce').mean()
        for k in SHORTEST:
            if f'{k}_correct_unbounded' not in d.columns:
                continue
            rec[f'{k}_acc_unbounded'] = d[f'{k}_correct_unbounded'].mean()
            rec[f'{k}_acc_either'] = d[f'{k}_correct_either'].mean()
        rec['overall_acc'] = d['n_correct'].sum() / d['n_items'].sum()
        return rec

    recs = [block(ok, 'all')]
    for gid, d in ok.groupby('group_id'):
        recs.append(block(d, group_names.get(gid, f'G{gid}')))
    summ = pd.DataFrame(recs)
    summ.insert(1, 'parse_errors', [int(scored['parse_error'].sum())] + [None] * (len(summ) - 1))
    return summ


# -------------------------------------------------------------------------
# 신뢰구간: 시나리오 단위 bootstrap (행동 그룹별 층화, 한 시나리오의 모든 피험자 행이 함께 뽑힘)
# -------------------------------------------------------------------------
def bootstrap_cis(scored, n_boot=2000, seed=0):
    ok = scored[~scored['parse_error']].copy()
    scen = sorted(ok['scenario_id'].unique())
    group_of = ok.groupby('scenario_id')['group_id'].first().to_dict()
    B = stratified_boot_indices(scen, group_of, n_boot, seed)
    pos = {s: i for i, s in enumerate(scen)}
    sidx = ok['scenario_id'].map(pos).to_numpy()

    metrics = {}
    for k in ITEMS:
        if f'{k}_gt' in ok.columns:
            m = ok[f'{k}_gt'].notna().to_numpy()
            metrics[f'{k}_acc'] = (ok[f'{k}_correct'].to_numpy(float), m)
    for k in MEANING:
        if f'{k}_choice' in ok.columns:
            m = ok[f'{k}_gt'].notna().to_numpy() if f'{k}_gt' in ok.columns else np.ones(len(ok), bool)
            for cid in 'abcd':
                metrics[f'{k}_choice_{cid}'] = ((ok[f'{k}_choice'] == cid).to_numpy(float), m)
    fact_cols = [f'{k}_correct' for k in FACT_ITEMS if f'{k}_gt' in ok.columns]
    metrics['fact_items_acc'] = (np.nanmean(ok[fact_cols].to_numpy(float), axis=1), np.ones(len(ok), bool))

    rows = []
    for name, (v, m) in metrics.items():
        point = v[m].mean()
        boots = []
        for ix in B:
            w = np.bincount(ix, minlength=len(scen))[sidx] * m
            boots.append((w * v).sum() / w.sum() if w.sum() > 0 else np.nan)
        lo, hi = percentile_ci(boots)
        rows.append(dict(metric=name, estimate=point, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# 파싱 정확도와 ToM 오차의 관계 (시나리오 단위)
# -------------------------------------------------------------------------
def tom_link(scored, model, condition, n_boot=2000, seed=0):
    """
    시나리오별 사실 문항 정확도 vs End-step ToM 오차 |LLM - 사람| (옵션 평균), desire / belief 각각.
    irrational 제외 시나리오만. Spearman r + 시나리오 bootstrap CI.
    음의 상관 = 좌표/궤적을 잘못 이해한 시나리오일수록 ToM 오차가 큼.
    """
    from src.config import HUMAN_PKL_PATH, get_group_indices
    end_pkl = os.path.join(BASE_RESULTS_DIR, model, condition, "model_data.pkl")
    if not os.path.exists(end_pkl):
        print(f"[warn] End-step 결과가 없어 ToM 연결 분석을 건너뜁니다: {end_pkl}")
        return None
    llm = pickle.load(open(end_pkl, 'rb'))
    human = pickle.load(open(HUMAN_PKL_PATH, 'rb'))
    rational = set(s for g in get_group_indices(include_irrational=False) for s in g)

    ok = scored[~scored['parse_error'] & scored['scenario_id'].isin(rational)]
    fact_cols = [f'{k}_correct' for k in FACT_ITEMS if f'{k}_gt' in ok.columns]
    acc = ok.assign(acc=np.nanmean(ok[fact_cols].to_numpy(float), axis=1)).groupby('scenario_id')['acc'].mean()
    scen = acc.index.to_numpy()
    group_of = ok.groupby('scenario_id')['group_id'].first().to_dict()
    B = stratified_boot_indices(list(scen), group_of, n_boot, seed)
    rows = []
    if acc.nunique() < 2:
        print("[info] 시나리오별 사실 문항 정확도가 모두 같아(예: 전부 정답) ToM 오차와의 상관을 정의할 수 없습니다.")
        return pd.DataFrame([dict(metric="spearman(fact accuracy, |LLM-human|)", estimate=np.nan, ci_low=np.nan,
                                  ci_high=np.nan, n_scenarios=len(scen), note="accuracy constant across scenarios")])
    for dim, key in (('desire', 'des_inf_mean'), ('belief', 'bel_inf_mean_norm')):
        err = np.abs(np.asarray(llm[key], float) - np.asarray(human[key], float)).mean(axis=0)[scen - 1]
        r = spearmanr(acc.values, err).correlation
        boots = [spearmanr(acc.values[ix], err[ix]).correlation for ix in B]
        lo, hi = percentile_ci(boots)
        rows.append(dict(metric=f"spearman(fact accuracy, |LLM-human| {dim})", estimate=r, ci_low=lo, ci_high=hi,
                         n_scenarios=len(scen)))
    return pd.DataFrame(rows)


def print_report(summ, scored):
    allrow = summ.iloc[0]
    print("\n" + "=" * 60)
    print(f"📋 Control 채점 결과 (rows={int(allrow['n_rows'])}, parse errors={int(scored['parse_error'].sum())})")
    print("=" * 60)
    for k in ITEMS:
        if f'{k}_acc' not in allrow or pd.isna(allrow[f'{k}_acc']):
            continue
        line = f"  {k:<28} acc = {allrow[f'{k}_acc']:.3f}"
        if k in NUMERIC_ITEMS:
            line += f"   MAE = {allrow[f'{k}_mae']:.2f}"
        if k in SHORTEST:
            line += f"   (unbounded acc = {allrow[f'{k}_acc_unbounded']:.3f})"
        print(line)
    print(f"  {'overall':<28} acc = {allrow['overall_acc']:.3f}")
    for k in MEANING:
        if f'{k}_choice_a' in allrow and pd.notna(allrow[f'{k}_choice_a']):
            dist = ", ".join(f"{cid}({CHOICE_NAMES[cid]})={allrow[f'{k}_choice_{cid}']:.2f}" for cid in 'abcd')
            print(f"  {k} 선택 분포: {dist}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="control 결과 폴더 (subject_*.csv)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--condition", default="vanilla")
    ap.add_argument("--variant", default="", help="하위 폴더 (예: thinking_off, effort_low)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--tom_link", action="store_true",
                    help="시나리오별 사실 문항 정확도와 End-step ToM 오차(|LLM-사람|)의 상관도 계산")
    args = ap.parse_args()

    input_dir = args.input or os.path.join(BASE_RESULTS_DIR, args.model, args.condition, "control", args.variant)
    df = load_results(input_dir)
    ver = result_version(df.columns)
    if ver != 'v3':
        print(f"ℹ️ 이전 버전({ver}) 결과입니다. 현재 문항 번호로 변환해 채점하며, 없는 문항은 건너뜁니다.")
    scored = score_all(df)
    summ = summarize(scored)

    scored.to_csv(os.path.join(input_dir, "control_scored.csv"), index=False)
    summ.to_csv(os.path.join(input_dir, "control_summary.csv"), index=False)
    print_report(summ, scored)

    ci = bootstrap_cis(scored, args.n_boot)
    model = args.model or (df['model'].dropna().iloc[0] if 'model' in df.columns else None)
    if args.tom_link and model:
        link = tom_link(scored, model, args.condition, args.n_boot)
        if link is not None:
            ci = pd.concat([ci, link], ignore_index=True)
    ci.to_csv(os.path.join(input_dir, "control_ci.csv"), index=False)
    print("\n📐 95% 시나리오 bootstrap CI")
    for r in ci.itertuples():
        print(f"  {r.metric:52} {r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]")
    print(f"\n✅ saved: {os.path.join(input_dir, 'control_ci.csv')}")
    print(f"✅ saved: {os.path.join(input_dir, 'control_scored.csv')}")
    print(f"✅ saved: {os.path.join(input_dir, 'control_summary.csv')}")


if __name__ == "__main__":
    main()
