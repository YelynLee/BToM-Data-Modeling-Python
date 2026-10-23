"""
응답 원문(response_text)을 시나리오 그룹 × phase로 층화 추출해 읽기 쉬운 Markdown으로 저장함.
+ 모든 응답에 대해 간단한 휴리스틱 표시를 붙인 CSV를 함께 저장함.

목적
  - Prefix-step 응답이 실제로 '그 시점까지의 정보'만으로 추론하는지 직접 읽어 확인
  - Spot 2가 아직 보이지 않았는데(seen_spot2=False) Spot 2의 실제 트럭을 단정하는 문장이 있는지
    (Map Configuration을 보고 정답을 투사하는 'knowledge projection'의 직접 증거) 빠르게 찾기

휴리스틱 (사람이 읽고 확인할 후보를 고르는 용도. 통계 근거로 쓰지 말 것)
  mentions_true_spot2 : 'Spot 2' 주변(같은 문장) 에 실제 Spot 2의 내용(L / M / empty)이 함께 언급됨
  -> seen_spot2=False 인 응답에서 이 비율이 높으면 투사 가능성. 표본을 읽어 판단하세요.

사용 예
  python analysis/sample_responses.py --dir results/claude-opus-4-8/vanilla/prefixstep/thinking_off \\
      --n 2 --out results/claude-opus-4-8/vanilla/prefixstep/thinking_off/response_samples.md
  옵션: --scenarios rational (기본) / all, --groups 1,4 (특정 그룹만), --seed 0
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

from src.dataset import df_btom  # noqa: E402
from src.config import get_scenario_subset  # noqa: E402


# -------------------------------------------------------------------------
# 시나리오 정보: Spot 2의 실제 내용, Spot 2가 처음 보이는 시점, phase
# -------------------------------------------------------------------------
def scenario_info():
    d = df_btom.copy()
    info = {}
    for sc, g in d.groupby("scenario_id"):
        r = g.iloc[0]
        truth = "Empty"
        for tr in ("L", "M"):
            if (r[f"{tr}_x"], r[f"{tr}_y"]) == (r["goal2_x"], r["goal2_y"]) and (r[f"{tr}_x"] or r[f"{tr}_y"]):
                truth = tr
        vis = g.loc[g["visible_goal2"] == 1, "time_step"]
        info[int(sc)] = {"spot2": truth, "t_vis": int(vis.min()) if len(vis) else None,
                         "group_id": int(r["group_id"]), "group_desc": r["group_desc"],
                         "T": int(g["time_step"].max())}
    return info


def phase_table():
    """(scenario, t) -> phase 이름 (prepare_everystep의 phase 정의를 그대로 사용)"""
    try:
        from src.prepare_everystep import apply_phase_labeling
    except Exception as e:  # pragma: no cover
        print(f"⚠️ phase 계산 생략: {e}")
        return {}
    d = df_btom.copy()
    d["subject_id"] = 1
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        d = apply_phase_labeling(d)
    return {(int(a), int(b)): p for a, b, p in zip(d["scenario_id"], d["time_step"], d["phase"])}


# -------------------------------------------------------------------------
def mentions_true_spot2(text, truth):
    """같은 문장 안에 'Spot 2'와 실제 내용(L / M / empty)이 함께 나오는지"""
    if not isinstance(text, str) or not text:
        return False
    target = r"\bempty\b" if truth == "Empty" else rf"(?<![A-Za-z]){truth}(?![A-Za-z])"
    for sent in re.split(r"(?<=[.!?\n])\s+", text):
        if re.search(r"spot\s*2", sent, re.I) and re.search(target, sent, re.I if truth == "Empty" else 0):
            return True
    return False


def load_runs(path):
    frames = []
    for f in sorted(glob.glob(os.path.join(path, "subject_*.csv"))):
        df = pd.read_csv(f)
        df["subject"] = int(os.path.basename(f)[8:10])
        frames.append(df)
    if not frames:
        sys.exit(f"{path}에 subject_*.csv가 없습니다.")
    return pd.concat(frames, ignore_index=True)


def unit_table(df, info, phases):
    """응답 1개 = 1행. prefixstep은 (subject, scenario, t), 그 외는 (subject, scenario)"""
    if "response_text" not in df.columns:
        sys.exit("response_text 열이 없습니다. 새로 실행했거나 fetch_batch_responses.py로 원문을 채운 결과인지 확인하세요.")
    is_prefix = (df.get("mode", pd.Series(["?"])).astype(str) == "prefixstep").any()
    df = df[df["response_text"].notna()].copy()
    rows = []
    for r in df.itertuples():
        sc = int(r.scenario_id)
        t = int(r.time_step) if is_prefix and pd.notna(getattr(r, "time_step", np.nan)) else info[sc]["T"]
        i = info[sc]
        bel = {k: getattr(r, f"belief_{k}", np.nan) for k in ("L", "M", "Empty")}
        rows.append({
            "subject": r.subject, "scenario_id": sc, "time_step": t, "T": i["T"],
            "group_id": i["group_id"], "group_desc": i["group_desc"],
            "phase": phases.get((sc, t), "?") if is_prefix else "(whole)",
            "spot2_truth": i["spot2"], "t_vis_spot2": i["t_vis"],
            "seen_spot2": i["t_vis"] is not None and t >= i["t_vis"],
            "belief_L": bel["L"], "belief_M": bel["M"], "belief_Empty": bel["Empty"],
            "output_tokens": getattr(r, "output_tokens", np.nan),
            "mentions_true_spot2": mentions_true_spot2(r.response_text, i["spot2"]),
            "response_text": r.response_text,
        })
    return pd.DataFrame(rows), is_prefix


def stratified(units, n, seed):
    rng = np.random.default_rng(seed)
    picks = []
    for _, g in units.groupby(["group_id", "phase"], sort=True):
        idx = rng.choice(len(g), size=min(n, len(g)), replace=False)
        picks.append(g.iloc[sorted(idx)])
    return pd.concat(picks) if picks else units.iloc[:0]


def to_markdown(units, sample, src, is_prefix):
    out = [f"# 응답 표본 ({src})", ""]
    out.append(f"- 응답 {len(units)}개 중 그룹 × phase별로 층화 추출한 {len(sample)}개")
    if is_prefix:
        unseen = units[~units["seen_spot2"]]
        seen = units[units["seen_spot2"]]
        out.append(f"- 휴리스틱 'Spot 2 문장에 실제 내용 언급' 비율: Spot 2를 보기 전 "
                   f"{unseen['mentions_true_spot2'].mean():.1%} (n={len(unseen)}), 본 후 "
                   f"{seen['mentions_true_spot2'].mean():.1%} (n={len(seen)})")
        out.append("  - 보기 전에 높다면 정답 투사 후보. 아래 ⚠️ 표시 응답을 직접 읽어 확인하세요.")
    out.append("")
    for (gid, gdesc), g in sample.groupby(["group_id", "group_desc"], sort=True):
        out.append(f"## Group {gid}: {gdesc}")
        for r in g.sort_values(["phase", "scenario_id", "time_step"]).itertuples():
            flag = " ⚠️" if (r.mentions_true_spot2 and not r.seen_spot2) else ""
            seen = "본 후" if r.seen_spot2 else "보기 전"
            out.append(f"### sc {r.scenario_id}, t={r.time_step}/{r.T} · phase {r.phase} · subject {r.subject}{flag}")
            out.append(f"- Spot 2 실제: **{r.spot2_truth}** (처음 보이는 시점 t={r.t_vis_spot2}, 현재 {seen})")
            out.append(f"- 응답 belief: L={r.belief_L}, M={r.belief_M}, Empty={r.belief_Empty}"
                       + (f" · 출력 토큰 {int(r.output_tokens)}" if pd.notna(r.output_tokens) else ""))
            out.append("")
            out.append("```text")
            out.append(str(r.response_text).strip())
            out.append("```")
            out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=2, help="그룹 × phase 칸마다 뽑을 응답 수")
    ap.add_argument("--scenarios", default="rational", choices=["rational", "all", "check", "goback"])
    ap.add_argument("--groups", default=None, help="예: 1,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Markdown 경로 (기본: <dir>/response_samples.md)")
    args = ap.parse_args()

    info = scenario_info()
    df = load_runs(args.dir)
    keep = get_scenario_subset(args.scenarios)   # 'all'이면 None
    if keep is not None:
        df = df[df["scenario_id"].isin(set(keep))]
    units, is_prefix = unit_table(df, info, phase_table())
    if args.groups:
        units = units[units["group_id"].isin([int(x) for x in args.groups.split(",")])]
    if units.empty:
        sys.exit("원문이 있는 응답이 없습니다.")

    sample = stratified(units, args.n, args.seed)
    out_md = args.out or os.path.join(args.dir, "response_samples.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(to_markdown(units, sample, args.dir, is_prefix))
    out_csv = os.path.splitext(out_md)[0] + "_all.csv"
    units.to_csv(out_csv, index=False)

    print(f"✅ 표본 {len(sample)}개 -> {out_md}")
    print(f"✅ 전체 응답 + 휴리스틱 표시 {len(units)}개 -> {out_csv}")
    if is_prefix:
        tab = units.groupby(["seen_spot2"])["mentions_true_spot2"].agg(["mean", "size"])
        print("\n'Spot 2 문장에 실제 내용 언급' 비율 (seen_spot2별):")
        print(tab.to_string())


if __name__ == "__main__":
    main()
