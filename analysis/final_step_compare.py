"""
마지막 스텝(t = T) 비교: 조건별 최종 판단이 사람 / BToM / 같은 모델의 End-step과 얼마나 맞는지.

t = T에서는 End-step, Every-step, Every-step-single, Prefix-step이 모두 같은 정보(전체 로그)를 봄.
따라서 이 시점의 차이는 정보 접근이 아니라 응답 형식·질문 해석에서 나온 것으로 볼 수 있음.

입력: 결과 폴더(subject_*.csv가 있는 곳) 또는 model_data.pkl.
  - 폴더에 model_data.pkl이 없으면 data_processor로 만들어서 사용 (stepwise는 실제 마지막 스텝 행만 사용)
  - 폴더 경로에서 mode(everystep / prefixstep / singlestep)를 찾음
지표: 73개 시나리오(irrational 제외), 옵션 3개를 펼친 pooled Pearson r, 시나리오 층화 bootstrap 95% CI

사용 예)
  python analysis/final_step_compare.py --model claude-opus-4-6 \\
      --input results/claude-opus-4-6/vanilla/everystep \\
              results/claude-opus-4-6/vanilla/singlestep \\
              results/claude-opus-4-6/vanilla/prefixstep \\
      --labels "Every-step" "Every-step-single" "Prefix-step" --out results/claude-opus-4-6/vanilla/compare/final_step.csv
  (End-step은 results/<model>/<condition>/model_data.pkl을 자동으로 추가. --no_end로 끔)
"""
import os
import sys
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "analysis")):
    if p not in sys.path:
        sys.path.append(p)

from src.config import (BASE_RESULTS_DIR, HUMAN_PKL_PATH, REFERENCE_PKL_DIR, STEPWISE_MODES,
                        get_group_indices)
from stats_utils import pooled_r, stratified_boot_indices, percentile_ci

KEY = {"desire": "des_inf_mean", "belief": "bel_inf_mean_norm"}


def find_mode(path):
    for part in reversed(os.path.normpath(path).split(os.sep)):
        for m in STEPWISE_MODES:
            if part.startswith(m):
                return m
    return "normal"


def load_pkl(path):
    if path.endswith(".pkl"):
        return pickle.load(open(path, "rb"))
    pkl = os.path.join(path, "model_data.pkl")
    if not os.path.exists(pkl):
        from src.data_processor import process_model_results
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            process_model_results(path, mode=find_mode(path))
    return pickle.load(open(pkl, "rb"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="End-step 위치를 찾을 모델 이름")
    ap.add_argument("--condition", default="vanilla")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--no_end", action="store_true", help="End-step을 추가하지 않음")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="결과 csv 경로")
    args = ap.parse_args()
    assert len(args.input) == len(args.labels), "--input과 --labels 개수가 같아야 합니다."

    groups = get_group_indices(include_irrational=False)
    group_of = {s: g for g, ids in enumerate(groups) for s in ids}
    scen = sorted(group_of)
    idx = np.array(scen) - 1
    boot = stratified_boot_indices(scen, group_of, args.n_boot, args.seed)

    runs = {}
    if not args.no_end:
        runs["End-step"] = load_pkl(os.path.join(BASE_RESULTS_DIR, args.model, args.condition, "model_data.pkl"))
    for p, lab in zip(args.input, args.labels):
        runs[lab] = load_pkl(p)
    refs = {"Human": pickle.load(open(HUMAN_PKL_PATH, "rb")),
            "BToM": pickle.load(open(os.path.join(REFERENCE_PKL_DIR, "btom", "btom_data.pkl"), "rb"))}
    if "End-step" in runs:
        refs["End-step"] = runs["End-step"]

    rows = []
    for dim, key in KEY.items():
        for lab, d in runs.items():
            x = d[key][:, idx]
            for rname, rd in refs.items():
                if rname == lab:
                    continue
                y = rd[key][:, idx]
                est = pooled_r(x, y)
                lo, hi = percentile_ci([pooled_r(x[:, b], y[:, b]) for b in boot])
                rows.append(dict(dimension=dim, condition=lab, versus=rname, r=est, ci_low=lo, ci_high=hi,
                                 n_scenarios=int(np.sum(~np.isnan(x).any(axis=0)))))
        y = refs["BToM"][key][:, idx]
        h = refs["Human"][key][:, idx]
        lo, hi = percentile_ci([pooled_r(y[:, b], h[:, b]) for b in boot])
        rows.append(dict(dimension=dim, condition="BToM", versus="Human", r=pooled_r(y, h), ci_low=lo, ci_high=hi,
                         n_scenarios=len(scen)))
    res = pd.DataFrame(rows)
    res["text"] = res.apply(lambda r: f"{r.r:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]", axis=1)
    for dim in KEY:
        print(f"\n== {dim} (final step, {len(scen)} scenarios) ==")
        print(res[res.dimension == dim].pivot(index="condition", columns="versus", values="text")
              .reindex(list(runs) + ["BToM"]).fillna("").to_string())
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        res.drop(columns="text").to_csv(args.out, index=False)
        print(f"\n✅ saved: {args.out}")


if __name__ == "__main__":
    main()
