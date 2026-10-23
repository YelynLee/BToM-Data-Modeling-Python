"""
공통 통계 함수 (numpy / scipy만 사용).

- 시나리오 단위 bootstrap: 행동 그룹별 층화, 모든 조건에 같은 리샘플을 써서 paired 비교
- 상관 비교: Williams t (공유 변수가 있는 두 종속 상관), Steiger Z
- paired 순열 검정 (시나리오 단위 sign-flip)
- 다중비교: Holm
- 사람 데이터 신뢰도: split-half (Spearman-Brown), leave-one-subject-out
"""
import numpy as np
from scipy import stats

ALPHA = 0.05


# -------------------------------------------------------------------------
# 상관
# -------------------------------------------------------------------------
def pooled_r(x, y):
    """여러 옵션 x 시나리오 값을 펼쳐서 Pearson r (NaN 무시)."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def williams_t(r_jk, r_jh, r_kh, n):
    """
    Williams (1959) t: 공통 변수 j에 대한 두 상관 r_jk, r_jh가 다른지 검정.
    예) j = 사람, k = LLM, h = BToM  ->  r(사람, LLM) vs r(사람, BToM), r_kh = r(LLM, BToM)
    Returns: t, df, two-sided p
    """
    det = 1 - r_jk ** 2 - r_jh ** 2 - r_kh ** 2 + 2 * r_jk * r_jh * r_kh
    rbar = (r_jk + r_jh) / 2
    num = (r_jk - r_jh) * np.sqrt((n - 1) * (1 + r_kh))
    den = np.sqrt(2 * ((n - 1) / (n - 3)) * det + rbar ** 2 * (1 - r_kh) ** 3)
    t = num / den
    df = n - 3
    return float(t), df, float(2 * stats.t.sf(abs(t), df))


def steiger_z(r_jk, r_jh, r_kh, n):
    """Steiger (1980) Z (Fisher z 기반, 공통 변수 j). Returns: z, two-sided p"""
    z_jk, z_jh = np.arctanh(r_jk), np.arctanh(r_jh)
    rm2 = ((r_jk + r_jh) / 2) ** 2
    f = min((1 - r_kh) / (2 * (1 - rm2)), 1.0)
    h = (1 - f * rm2) / (1 - rm2)
    z = (z_jk - z_jh) * np.sqrt((n - 3) / (2 * (1 - r_kh) * h))
    return float(z), float(2 * stats.norm.sf(abs(z)))


# -------------------------------------------------------------------------
# Bootstrap
# -------------------------------------------------------------------------
def stratified_boot_indices(scenario_ids, group_of, n_boot=2000, seed=0):
    """
    행동 그룹별로 층화한 시나리오 복원추출.
    Returns: (n_boot, n_scen) 정수 배열 — 각 행은 scenario_ids 안의 위치(index)들.
    모든 조건에 같은 배열을 쓰면 paired bootstrap이 됨.
    """
    rng = np.random.default_rng(seed)
    scenario_ids = list(scenario_ids)
    by_group = {}
    for i, s in enumerate(scenario_ids):
        by_group.setdefault(group_of[s], []).append(i)
    out = np.empty((n_boot, len(scenario_ids)), dtype=int)
    for b in range(n_boot):
        out[b] = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in by_group.values()])
    return out


def indices_to_weights(idx, n):
    """리샘플 인덱스 -> 시나리오별 뽑힌 횟수."""
    return np.bincount(idx, minlength=n).astype(float)


def percentile_ci(samples, alpha=ALPHA):
    a = np.asarray(samples, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan, np.nan
    return float(np.percentile(a, 100 * alpha / 2)), float(np.percentile(a, 100 * (1 - alpha / 2)))


def boot_p_two_sided(diff_samples):
    """paired bootstrap 차이 분포가 0을 넘나드는 비율로 구한 양측 p (최소 1/n_boot)."""
    d = np.asarray(diff_samples, float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return np.nan
    p = 2 * min((d > 0).mean(), (d < 0).mean())
    return float(max(p, 1.0 / d.size))


# -------------------------------------------------------------------------
# 순열 검정
# -------------------------------------------------------------------------
def paired_signflip_test(diffs, n_perm=10000, seed=0):
    """
    시나리오 단위 paired 차이(d_s = 조건B - 조건A)의 평균이 0인지 sign-flip 순열 검정.
    시나리오 안의 여러 스텝은 먼저 시나리오 평균으로 묶어서 넣어야 함(반복측정 처리).
    Returns: mean diff, two-sided p
    """
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
    return float(obs), float(p)


# -------------------------------------------------------------------------
# 다중비교
# -------------------------------------------------------------------------
def holm(pvals):
    """Holm-Bonferroni 보정 p (입력 순서 유지, NaN은 그대로)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    m = len(ok)
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


# -------------------------------------------------------------------------
# 사람 데이터 신뢰도 (상한선)
# -------------------------------------------------------------------------
def split_half_reliability(data_3d, mask, n_split=1000, seed=0, normalize_belief=False):
    """
    data_3d: (3 options, 78 scenarios, n_subj) 개인 평정.
    피험자를 무작위로 반씩 나눠 두 반의 평균끼리 pooled r을 구하고 Spearman-Brown 보정.
    Returns: dict(r_half=mean r, r_sb=보정 신뢰도, ci_low, ci_high, noise_ceiling=sqrt(r_sb))
    noise_ceiling은 '그룹 평균과 상관할 수 있는 최대치'의 근사로 흔히 쓰는 sqrt(신뢰도).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(data_3d, float)[:, mask, :]
    n = X.shape[2]
    rs = []
    for _ in range(n_split):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        ma, mb = np.nanmean(X[:, :, a], axis=2), np.nanmean(X[:, :, b], axis=2)
        if normalize_belief:
            ma, mb = _norm_belief(ma), _norm_belief(mb)
        rs.append(pooled_r(ma, mb))
    rs = np.asarray(rs)
    sb = 2 * rs / (1 + rs)
    lo, hi = percentile_ci(sb)
    return {"r_half": float(np.nanmean(rs)), "r_sb": float(np.nanmean(sb)), "ci_low": lo, "ci_high": hi,
            "noise_ceiling": float(np.sqrt(max(np.nanmean(sb), 0)))}


def loso_r(data_3d, mask, normalize_belief=False):
    """한 명씩 빼고 나머지 평균과의 pooled r (개인 수준 일치도)의 평균과 범위."""
    X = np.asarray(data_3d, float)[:, mask, :]
    rs = []
    for s in range(X.shape[2]):
        one = X[:, :, s]
        rest = np.nanmean(np.delete(X, s, axis=2), axis=2)
        if normalize_belief:
            one, rest = _norm_belief(one), _norm_belief(rest)
        rs.append(pooled_r(one, rest))
    rs = np.asarray(rs)
    return {"mean": float(np.nanmean(rs)), "min": float(np.nanmin(rs)), "max": float(np.nanmax(rs))}


def _norm_belief(m):
    """1~7 belief 평균 (3, n) -> (x-1)/합 (논문 bel_inf_mean_norm과 같은 처리)."""
    s = np.maximum(np.asarray(m, float) - 1, 0)
    tot = np.nansum(s, axis=0)
    tot[tot == 0] = 1.0
    return s / tot[None, :]
