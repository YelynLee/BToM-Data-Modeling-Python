"""
Spot 2의 실제 내용(참 상태) 쪽으로 끌리는 정도(pull)를 Spot 2가 보이기 전 / 보인 뒤 / 끝내 안 보임으로 나눠 계산.

  pull: 각 스텝에서 (LLM - BToM)을 (참 one-hot - BToM) 방향으로 사영해 시나리오별 Σnum/Σden. 0 = BToM, 1 = 참 확신
        TrueBelief 모델의 belief는 모든 스텝에서 참 one-hot이므로 pull = TrueBelief 방향 사영 지수와 같음
  구간: pre  = t < t_vis (나중에 Spot 2를 보는 시나리오). TrueBelief는 1, Hindsight는 0을 예측 → 두 모델을 가르는 구간
        post = t >= t_vis. Hindsight와 참 방향이 거의 같음(코사인 약 0.99)
        never = Spot 2를 끝내 못 보는 시나리오 (음성 대조)
  추가: Spot 2가 보인 뒤 BToM / Hindsight / 현재 belief(보인 뒤 = 참)와의 평균 절대오차

사용 예)
  R=results/claude-opus-4-6/vanilla
  python analysis/truth_leak_explore.py \\
      --input $R/everystep/thinking_off/rerun $R/singlestep/thinking_off $R/singlestep_trunc/thinking_off $R/prefixstep/thinking_off \\
      --labels "Every-step" "Every-step-single" "Single-trunc" "Prefix-step" --out_dir $R/compare_4cond
출력: truth_leak_index.csv (구간 x 조건 추정치와 CI), truth_leak_pairs.csv (조건 쌍 차이, 순열 p), truth_leak_steps.csv
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.prepare_everystep import everystep_column

OPT = ["L", "M", "Empty"]
SEGMENTS = {"pre": "before Spot 2 visible (t < t_vis)", "post": "after Spot 2 visible (t >= t_vis)",
            "never": "Spot 2 never visible (negative control)"}


def load_ratings(path):
    if path.endswith(".csv"):
        d = pd.read_csv(path)
    else:
        d = pd.concat([pd.read_csv(f).assign(subject_id=i)
                       for i, f in enumerate(sorted(glob.glob(os.path.join(path, "subject_*.csv"))))])
    cols = [f"belief_{o}" for o in OPT]
    m = d.dropna(subset=cols).groupby(["scenario_id", "time_step"])[cols].mean()
    x = np.maximum(m.values - 1, 0)
    s = x.sum(1, keepdims=True)
    s[s == 0] = np.nan
    return pd.DataFrame(x / s, index=m.index, columns=OPT)


def load_refs():
    def mat(name, f):
        b = sio.loadmat(os.path.join(ROOT, "data", name, f), squeeze_me=True)["belief_marg"]
        return [np.atleast_2d(b[i]).reshape(3, -1) for i in range(len(b))]
    return (mat("btom", "btom_everystep_beta2.5.mat"), mat("hindsight", "hindsight_everystep_beta3.5.mat"),
            mat("truebelief", "truebelief_everystep_beta9.0.mat"))


def scenario_info():
    from src.dataset import df_btom
    _, _, TB = load_refs()
    info = {}
    for sc, g in df_btom.groupby("scenario_id"):
        if "is_irrational" in g and int(g.is_irrational.iloc[0]) == 1:
            continue
        vis = g.loc[g.visible_goal2 == 1, "time_step"]
        info[int(sc)] = dict(T=int(g.time_step.max()), tvis=int(vis.min()) if len(vis) else None, truth=TB[sc - 1][:, 0])
    return info


def step_table(conds):
    BT, HS, _ = load_refs()
    info = scenario_info()
    rows = []
    for lab, path in conds.items():
        for (sc, t), v in load_ratings(path).iterrows():
            if sc not in info or v.isna().any():
                continue
            I = info[sc]
            j = everystep_column(t, I["T"], BT[sc - 1].shape[1])
            b, h = BT[sc - 1][:, j], HS[sc - 1][:, j]
            seen = I["tvis"] is not None
            seg = "never" if not seen else ("pre" if t < I["tvis"] else "post")
            d = I["truth"] - b
            den = float(d @ d)
            cur = I["truth"] if seg == "post" else b
            rows.append(dict(cond=lab, sc=sc, t=t, segment=seg,
                             num=float((v.values - b) @ d) if den >= 0.05 ** 2 else 0.0,
                             den=den if den >= 0.05 ** 2 else 0.0,
                             err_btom=np.abs(v.values - b).mean(), err_hindsight=np.abs(v.values - h).mean(),
                             err_current=np.abs(v.values - cur).mean()))
    return pd.DataFrame(rows)


def scenario_pull(L):
    g = L.groupby(["cond", "segment", "sc"])[["num", "den"]].sum()
    g = g[g.den > 0]
    return (g.num / g.den).rename("pull").reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert len(a.input) == len(a.labels)
    rng = np.random.default_rng(a.seed)
    conds = dict(zip(a.labels, a.input))
    L = step_table(conds)
    P = scenario_pull(L)

    def boot(v):
        return np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(a.n_boot)], [2.5, 97.5])

    out, pairs = [], []
    for seg in SEGMENTS:
        for lab in a.labels:
            v = P[(P.segment == seg) & (P.cond == lab)].pull.values
            lo, hi = boot(v)
            out.append(dict(segment=seg, segment_label=SEGMENTS[seg], condition=lab, estimate=v.mean(),
                            ci_low=lo, ci_high=hi, n_scenarios=len(v)))
        W = P[P.segment == seg].pivot(index="sc", columns="cond", values="pull")
        for i, x in enumerate(a.labels):
            for y in a.labels[i + 1:]:
                d = (W[y] - W[x]).dropna().values
                lo, hi = boot(d)
                null = (rng.choice([-1, 1], (a.n_perm, len(d))) * d).mean(1)
                p = (np.sum(np.abs(null) >= abs(d.mean())) + 1) / (a.n_perm + 1)
                pairs.append(dict(segment=seg, comparison=f"{y} - {x}", estimate=d.mean(), ci_low=lo, ci_high=hi,
                                  p_perm=p, n_scenarios=len(d)))
    out, pairs = pd.DataFrame(out), pd.DataFrame(pairs)
    err = (L[L.segment == "post"].groupby(["cond", "sc"])[["err_btom", "err_hindsight", "err_current"]].mean()
           .groupby("cond").mean().reindex(a.labels))
    os.makedirs(a.out_dir, exist_ok=True)
    out.to_csv(os.path.join(a.out_dir, "truth_leak_index.csv"), index=False)
    pairs.to_csv(os.path.join(a.out_dir, "truth_leak_pairs.csv"), index=False)
    err.to_csv(os.path.join(a.out_dir, "truth_leak_post_errors.csv"))
    L.to_csv(os.path.join(a.out_dir, "truth_leak_steps.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    print(pairs.round(4).to_string(index=False))
    print("\nMean abs error after Spot 2 visible (lower = closer):")
    print(err.round(3).to_string())


if __name__ == "__main__":
    main()
