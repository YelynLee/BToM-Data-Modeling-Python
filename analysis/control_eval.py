"""
Control 실험 채점.

main_experiment.py --mode control 결과(subject_*.csv)를 읽어,
행마다 저장된 probe 시점으로 정답을 다시 계산하고 문항별로 채점함.
(정답 정의는 src/control_items.py 하나에서 관리되며 프롬프트와 공유됨)

출력 (결과 폴더 안):
  control_scored.csv   : 행(피험자 x 시나리오)별 응답, 정답, 정오, 오차
  control_summary.csv  : 문항별 정확도 / 평균 절대오차 (전체 + 행동 그룹별)

사용 예)
  python analysis/control_eval.py --input results/claude-opus-4-6/vanilla/control
  python analysis/control_eval.py --model claude-opus-4-6 --condition vanilla --variant thinking_off
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
from src.control_items import ground_truth, score_response, ITEMS, NUMERIC_ITEMS

PROBE_COLS = ['probe_t1', 'probe_t2a', 'probe_t2b', 'probe_t9']


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
        base = {'subject_id': r['subject_id'], 'scenario_id': int(r['scenario_id']),
                'group_id': int(group_of[int(r['scenario_id'])])}
        if isinstance(r.get('error'), str) or any(pd.isna(r.get(c)) for c in PROBE_COLS):
            base['parse_error'] = True
            rows.append(base)
            continue
        probes = {c: int(r[c]) for c in PROBE_COLS}
        gt = ground_truth(scen[base['scenario_id']], probes)
        sc = score_response(r, gt)
        out = dict(base, parse_error=False, **probes)
        for k in ITEMS:
            out[f'{k}_pred'] = r.get(k)
            out[f'{k}_gt'] = json.dumps(gt[k]) if isinstance(gt[k], dict) else gt[k]
        for k in ('q5_shortest_moves_spot1', 'q6_shortest_moves_spot2'):
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
            rec[f'{k}_acc'] = d[f'{k}_correct'].mean()
        for k in NUMERIC_ITEMS:
            rec[f'{k}_mae'] = pd.to_numeric(d[f'{k}_abs_err'], errors='coerce').mean()
        for k in ('q5_shortest_moves_spot1', 'q6_shortest_moves_spot2'):
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


def print_report(summ, scored):
    allrow = summ.iloc[0]
    print("\n" + "=" * 60)
    print(f"📋 Control 채점 결과 (rows={int(allrow['n_rows'])}, parse errors={int(scored['parse_error'].sum())})")
    print("=" * 60)
    for k in ITEMS:
        line = f"  {k:<28} acc = {allrow[f'{k}_acc']:.3f}"
        if k in NUMERIC_ITEMS:
            line += f"   MAE = {allrow[f'{k}_mae']:.2f}"
        if k in ('q5_shortest_moves_spot1', 'q6_shortest_moves_spot2'):
            line += f"   (unbounded acc = {allrow[f'{k}_acc_unbounded']:.3f})"
        print(line)
    print(f"  {'overall':<28} acc = {allrow['overall_acc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="control 결과 폴더 (subject_*.csv)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--condition", default="vanilla")
    ap.add_argument("--variant", default="", help="하위 폴더 (예: thinking_off, effort_low)")
    args = ap.parse_args()

    input_dir = args.input or os.path.join(BASE_RESULTS_DIR, args.model, args.condition, "control", args.variant)
    df = load_results(input_dir)
    scored = score_all(df)
    summ = summarize(scored)

    scored.to_csv(os.path.join(input_dir, "control_scored.csv"), index=False)
    summ.to_csv(os.path.join(input_dir, "control_summary.csv"), index=False)
    print_report(summ, scored)
    print(f"\n✅ saved: {os.path.join(input_dir, 'control_scored.csv')}")
    print(f"✅ saved: {os.path.join(input_dir, 'control_summary.csv')}")


if __name__ == "__main__":
    main()
