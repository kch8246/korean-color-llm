#!/usr/bin/env python3
"""논문용 그림 4종 (폭 130mm = 학위논문 본문 규격, 흑백 인쇄에서도 구분되도록 마커·선종류 병용)
사용: python make_figures.py <scored_rep0.csv> [출력폴더]
"""
import sys, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

scored = sys.argv[1] if len(sys.argv) > 1 else "results/scored_rep0.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "results/figures"

for p in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in p or "NotoSansCJK-Bold" in p:
        fm.fontManager.addfont(p)
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "font.size": 7.5, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300})
W = 130 / 25.4
INK, MUT, GRID = "#1a1a1a", "#666666", "#dddddd"
BLUE, ORANGE = "#2a78d6", "#eb6834"

CORE = ["claude-haiku-5-5", "gemini-3.8-flash", "claude-sonnet-5-5", "deepseek-v4-pro", "perplexity-sonar",
        "gemma-4-31b-it", "gpt-5.4", "HCX-007", "gpt-4o", "HCX-005"]
SHORT = {"claude-haiku-5-5": "Claude Haiku 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5", "gemini-3.8-flash": "Gemini 3.8 Flash",
         "deepseek-v4-pro": "DeepSeek V4 Pro", "perplexity-sonar": "Perplexity Sonar", "gemma-4-31b-it": "Gemma 4 31B",
         "gpt-5.4": "GPT-5.4", "HCX-007": "HCX-007", "gpt-4o": "GPT-4o", "HCX-005": "HCX-005"}

s = pd.read_csv(scored)
s = s[s.model_label.isin(CORE) | s.model_label.str.contains("websearch")]

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

def save(fig, name):
    fig.savefig(f"{out}/{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---------------- 그림 1: 모델 x 하위과제 정답률 히트맵
cols = [("S1a 모음 단서", lambda d: d.subset == "S1a_vowel"),
        ("S1b 된소리", lambda d: d.subset == "S1b_fortis"),
        ("S1c 접사 순위", lambda d: d.subset == "S1c_affix_rank"),
        ("S1e 무의미어", lambda d: d.subset == "S1e_nonce_vowel"),
        ("S1g 5단계 순위", lambda d: d.subset == "S1g_rank5"),
        ("S3a 상위어 없음", lambda d: (d.subset == "S3a_referent") & (d.variant == "nohyper")),
        ("S3a 상위어 있음", lambda d: (d.subset == "S3a_referent") & (d.variant == "hyper")),
        ("S3b 어휘 함정", lambda d: d.subset == "S3b_lexical_trap"),
        ("S3c 어원", lambda d: d.subset == "S3c_etymology"),
        ("S4 관용 표현", lambda d: d.set_id == "S4")]
base = s[s.model_label.isin(CORE) & (s.set_id.isin(["S1", "S3", "S4"])) & s.correct.notna()]
M = np.full((len(CORE), len(cols)), np.nan); N = []
for j, (_, f) in enumerate(cols):
    sub = base[f(base)]
    N.append(sub.item_id.nunique())
    for i, m in enumerate(CORE):
        g = sub[sub.model_label == m]
        if len(g): M[i, j] = g.correct.mean()
fig, ax = plt.subplots(figsize=(W, 3.5))
im = ax.imshow(M, cmap="Greys_r", vmin=0, vmax=1.18, aspect="auto")  # 높을수록 밝음 → 어두운 칸 = 약점
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if np.isnan(v): continue
        ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", va="center", fontsize=6.8,
                color="white" if v < 0.55 else INK, fontweight="bold" if v < 0.85 else "normal")
ax.set_xticks(range(len(cols)))
LAB = ["S1a\n모음 단서", "S1b\n된소리", "S1c\n접사 순위", "S1e\n무의미어", "S1g\n5단계 순위", "S3a\n상위어 無", "S3a\n상위어 有", "S3b\n어휘 함정", "S3c\n어원", "S4\n관용 표현"]
ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(LAB, N)], fontsize=6.0)
ax.set_yticks(range(len(CORE))); ax.set_yticklabels([SHORT[m] for m in CORE], fontsize=7)
ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True); ax.set_yticks(np.arange(-.5, len(CORE), 1), minor=True)
ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.ax.tick_params(labelsize=6, length=2); cb.set_label("정답률", fontsize=6.5)
cb.outline.set_visible(False); cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
save(fig, "fig1_heatmap_accuracy")

# ---------------- 그림 2: S8 편향 문서 용량-반응
conds = ["support", "irrelevant", "no_doc", "biased_1", "biased_3"]
clab = ["지지\n문서", "무관\n문서", "문서\n없음", "편향 문서\n1건", "편향 문서\n3건"]
import os
R = pd.read_csv(os.path.join(os.path.dirname(scored), "table_s8_hyper_rate_wide.csv"), index_col=0, encoding="utf-8-sig")[conds].loc[CORE]
fig, ax = plt.subplots(figsize=(W, 3.1))
x = np.arange(len(conds))
for m in CORE:
    ax.plot(x, R.loc[m].values, color="#b5b5b5", lw=0.8, zorder=1)
hi = {"gemma-4-31b-it": (0.0, 0.015), "deepseek-v4-pro": (0.0, -0.04), "claude-haiku-5-5": (0, 0), "gpt-5.4": (0, 0)}
for m in ["gemma-4-31b-it", "claude-haiku-5-5", "gpt-5.4"]:
    ax.plot(x, R.loc[m].values, color=INK, lw=1.0, ls={"gemma-4-31b-it": "-", "claude-haiku-5-5": "--", "gpt-5.4": ":"}[m],
            marker={"gemma-4-31b-it": "s", "claude-haiku-5-5": "^", "gpt-5.4": "o"}[m], ms=3.5, zorder=3)
    ax.text(x[-1] + 0.12, R.loc[m].values[-1] + 0.0, SHORT[m], fontsize=6.3, va="center", color=INK)
mu = R.mean()
ax.plot(x, mu.values, color=BLUE, lw=2.2, marker="o", ms=5, zorder=4)
for xi, v in zip(x, mu.values):
    ax.text(xi, v + 0.065, f"{v:.2f}", ha="center", fontsize=6.8, color=BLUE, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=6)
ax.text(x[0] - 0.2, 0.92, "굵은 파란선 = 10개 모델 평균\n회색선 = 개별 모델", fontsize=6.3, color=MUT, va="top")
ax.set_xticks(x); ax.set_xticklabels(clab, fontsize=6.8)
ax.set_ylim(-0.03, 1.05); ax.set_xlim(-0.3, 5.1)
ax.set_ylabel("상위어(편향 답) 선택 비율", fontsize=7)
ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.axvline(2, color=GRID, lw=0.8, ls="--")
save(fig, "fig2_s8_dose_response")

# ---------------- 그림 3: S3a 상위어 유무 대비 (모델별 대응 비교)
s3 = s[(s.subset == "S3a_referent") & s.model_label.isin(CORE) & s.correct.notna()]
A = s3.groupby(["model_label", "variant"]).correct.mean().unstack().loc[CORE]
n = s3[s3.model_label == CORE[0]].groupby("variant").item_id.nunique().to_dict()
order = A.sort_values("hyper").index.tolist()
fig, ax = plt.subplots(figsize=(W, 3.0))
for i, m in enumerate(order):
    ax.plot([A.loc[m, "hyper"], A.loc[m, "nohyper"]], [i, i], color="#999999", lw=1.2, zorder=1)
    ax.scatter(A.loc[m, "nohyper"], i, s=26, marker="o", color=BLUE, edgecolor="white", lw=0.6, zorder=3)
    ax.scatter(A.loc[m, "hyper"], i, s=26, marker="s", color=ORANGE, edgecolor="white", lw=0.6, zorder=3)
    ax.text(1.06, i, f"{A.loc[m,'hyper']-A.loc[m,'nohyper']:+.2f}", va="center", fontsize=6.5, color=INK)
ax.text(1.06, len(order) - 0.35, "차이", fontsize=6.5, color=MUT)
ax.set_yticks(range(len(order))); ax.set_yticklabels([SHORT[m] for m in order], fontsize=7)
ax.set_xlim(0.5, 1.03); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_xlabel("정답률", fontsize=7)
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.scatter([], [], s=26, marker="o", color=BLUE, label="상위어 없음 (nohyper)")
ax.scatter([], [], s=26, marker="s", color=ORANGE, label="상위어 있음 (hyper)")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.5, frameon=False, handletextpad=0.3)
save(fig, "fig3_s3a_hypernym")

# ---------------- 그림 4: 웹 검색 효과
sr = s[(s.set_id.isin(["S0", "S3", "S4"])) & s.correct.notna()]
pairs = [("gpt-4o", "gpt-4o+websearch", "GPT-4o"), ("claude-sonnet-5-5", "claude-sonnet-5-5+websearch", "Claude Sonnet 5.5")]
fig, ax = plt.subplots(figsize=(W * 0.72, 2.5))
xs = []
for i, (a, b, lab) in enumerate(pairs):
    ids = set(sr[sr.model_label == b].item_id)
    for k, (mm, mk, col, dx) in enumerate([(a, "o", "#555555", -0.12), (b, "D", BLUE, 0.12)]):
        g = sr[(sr.model_label == mm) & sr.item_id.isin(ids)]
        kk, nn = int(g.correct.sum()), len(g); p = kk / nn; lo, hi_ = wilson(kk, nn)
        ax.errorbar(i + dx, p, yerr=[[p - lo], [hi_ - p]], fmt=mk, color=col, ms=5, capsize=2.5, lw=1, zorder=3)
        ax.text(i + dx + (0.07 if k else -0.07), p, f"{p:.3f}", ha="left" if k else "right", va="center", fontsize=6.5)
    xs.append(i)
ax.set_xticks(xs); ax.set_xticklabels([p[2] for p in pairs], fontsize=7)
ax.set_xlim(-0.6, 1.6); ax.set_ylim(0.78, 1.02); ax.set_ylabel("정답률 (95% 신뢰구간)", fontsize=7)
ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.errorbar([], [], fmt="o", color="#555555", label="검색 없음"); ax.errorbar([], [], fmt="D", color=BLUE, label="웹 검색 사용")
ax.legend(loc="lower right", fontsize=6.5, frameon=False)
save(fig, "fig4_search_effect")
print("done", out)
