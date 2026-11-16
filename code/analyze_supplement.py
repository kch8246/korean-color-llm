#!/usr/bin/env python3
"""보충 분석(2026년 11월): 6회 반복 자료, 검색 6회 대 6회, 온도 0, 검색 기록, 정답 기준 점검.

저장소 루트에서 실행:  python code/analyze_supplement.py > results/supplement_result.txt

입력(모두 results/ 안):
  all_trials_scored_compact.csv  본 실험과 보충 호출의 채점 결과(프롬프트·응답 원문 제외)
  temp0_scored_compact.csv       온도 0 호출의 채점 결과
  search_log_summary.csv         검색 기록(7회차) 요약
  human_item_summary.csv, human_s3a2_items.csv  모어 화자 응답의 문항별 집계

논문의 표 6, 표 7, 표 8, 표 13(32쌍 열), 표 15와 4.5, 4.6, 4.8, 4.11, 4.13절의 보충 수치를 다시 계산한다.
읽어 내지 못한 택일 응답은 오답으로 센다. 다만 S3a 반복 분석(4.11, 4.12, 4.13절)은 논문과 같이
10월에 수집한 호출만 쓰고 읽어 내지 못한 응답은 뺀다.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

R = "results/"
MODELS = ["gpt-4o", "gpt-5.4", "claude-sonnet-5-5", "claude-haiku-5-5", "gemini-3.8-flash",
          "gemma-4-31b-it", "deepseek-v4-pro", "HCX-005", "HCX-007", "perplexity-sonar"]
SEARCH = {"gpt-4o": "gpt-4o+websearch", "claude-sonnet-5-5": "claude-sonnet-5-5+websearch"}


def holm(ps):
    ps = np.array(ps, float); n = len(ps); order = np.argsort(ps); adj = np.empty(n); run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (n - i) * ps[k])); adj[k] = run
    return adj


def perm_p(diffs, unit=600):
    """문항(쌍)별 차이의 부호를 뒤바꾸는 정확한 양측 순열 검정."""
    ks = [int(round(v * unit)) for v in diffs if abs(v) > 1e-12]
    if not ks:
        return 1.0
    dist = {0: 1}
    for k in ks:
        nd = defaultdict(int)
        for s, c in dist.items():
            nd[s + k] += c; nd[s - k] += c
        dist = nd
    obs = abs(sum(ks))
    return sum(c for s, c in dist.items() if abs(s) >= obs) / 2 ** len(ks)


def boot_ci(values, n=5000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(values, float)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def section(title):
    print("\n" + "=" * 8 + " " + title)


d = pd.read_csv(R + "all_trials_scored_compact.csv", encoding="utf-8-sig", low_memory=False)
d["search"] = d.search.astype(str).str.lower().eq("true")
d["c"] = d.correct
d.loc[(d.task_type == "choice") & d.gold.notna() & d.c.isna(), "c"] = 0.0
core = d[d.version_added.astype(str) != "0.3"]
base = core[core.model_label.isin(MODELS) & ~core.search & core.rep.between(1, 6)]

# ---------------------------------------------------------------- 표 6: S7 6회
section("표 6. 프롬프트 언어(S7, 107쌍, 6회 평균)")
en = base[base.set_id == "S7"][["model_label", "pair_id", "rep", "c"]].rename(columns={"c": "en"})
ko = base[["model_label", "item_id", "rep", "c"]].rename(columns={"item_id": "pair_id", "c": "ko"})
j = en.merge(ko, on=["model_label", "pair_id", "rep"]).dropna(subset=["en", "ko"])
rows = []
for m in MODELS:
    g = j[j.model_label == m].groupby("pair_id")[["ko", "en"]].mean(); diff = g.ko - g.en
    rows.append(dict(model=m, pairs=len(g), ko=g.ko.mean(), en=g.en.mean(), diff=diff.mean(),
                     ko_lower=int((diff < 0).sum()), ko_higher=int((diff > 0).sum()), p=perm_p(diff.values, 6)))
t = pd.DataFrame(rows); t["p_holm"] = holm(t.p)
print(t.round(3).to_string(index=False))

# ---------------------------------------------------------------- 표 7: 검색 6회 대 6회
section("표 7. 검색 없음과 있음(6회 대 6회)")
x = core[core.rep.between(1, 6)]
rows = []


def compare(name, df, col):
    for m, s in SEARCH.items():
        b = df[df.model_label == m].groupby("item_id")[col].mean()
        w = df[df.model_label == s].groupby("item_id")[col].mean()
        jj = pd.concat([b, w], axis=1, keys=["off", "on"]).dropna(); diff = jj.on - jj.off
        rows.append(dict(task=name, model=m, items=len(jj), off=jj.off.mean(), on=jj.on.mean(), diff=diff.mean(),
                         up=int((diff > 0).sum()), down=int((diff < 0).sum()), p=perm_p(diff.values, 6)))


s34 = x[x.set_id.isin(["S3", "S4"]) & x.c.notna()]
s8 = x[x.set_id == "S8"]
compare("S3·S4 정답률", s34, "c")
compare("S8 편향 1·3건 상위어 선택률", s8[s8.variant.isin(["biased_1", "biased_3"])], "chose_hyper")
t = pd.DataFrame(rows); t["p_holm"] = holm(t.p)
print(t.round(3).to_string(index=False))
rows = []
compare("S3a 정답률", s34[s34.subset == "S3a_referent"], "c")
for v in ["biased_1", "biased_3", "irrelevant", "support"]:
    compare("S8 " + v, s8[s8.variant == v], "chose_hyper")
print("-- 세부(보정하지 않음)"); print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------------------------------------------------------- S8 6회, 20개 검정
section("4.13절. S8 편향 문서 20개 검정(6회 반복, 문서 없음과 비교)")
nodoc = base[(base.subset == "S3a_referent") & (base.variant == "hyper")].rename(columns={"item_id": "key"})
doc = base[base.set_id == "S8"].rename(columns={"base_item_id": "key"})
rows = []
for v in ["biased_1", "biased_3"]:
    for m in MODELS:
        a = nodoc[nodoc.model_label == m].groupby("key").chose_hyper.mean()
        b = doc[(doc.model_label == m) & (doc.variant == v)].groupby("key").chose_hyper.mean()
        jj = pd.concat([a, b], axis=1, keys=["nodoc", "doc"]).dropna(); diff = jj.doc - jj.nodoc
        rows.append(dict(cond=v, model=m, items=len(jj), nodoc=jj.nodoc.mean(), doc=jj.doc.mean(),
                         up=int((diff > 0).sum()), down=int((diff < 0).sum()), p=perm_p(diff.values, 6)))
t = pd.DataFrame(rows); t["p_holm20"] = holm(t.p)
print(t.round(4).to_string(index=False)); print("보정 후 유의:", int((t.p_holm20 < .05).sum()), "/ 20")

# ---------------------------------------------------------------- S9 6회
section("4.8절. S9 반박 비율(키워드 추정), 첫 호출과 6회 평균")
s9 = base[base.set_id == "S9"]
print(pd.concat([s9[s9.rep == 1].groupby(["model_label", "variant"]).rebuts.mean().unstack()[["justify_wrong", "justify_gold"]],
                 s9.groupby(["model_label", "variant"]).rebuts.mean().unstack()[["justify_wrong", "justify_gold"]]],
                axis=1, keys=["첫 호출", "6회"]).loc[MODELS].round(2).to_string())

# ---------------------------------------------------------------- S3a 32쌍(10월 호출, 읽지 못한 응답 제외)
section("표 13의 32쌍 열과 표 15의 S3a 32쌍 열")
s3a = d[d.model_label.isin(MODELS) & ~d.search & (d.subset == "S3a_referent") & d.rep.between(1, 6)
        & ~d.run_tag.astype(str).str.startswith("sup") & d.correct.notna()]
print("분석 호출 수", len(s3a))
print(s3a.groupby(["model_label", "variant"]).correct.mean().unstack().loc[MODELS].round(3).to_string())
print("전체", s3a.groupby("variant").correct.mean().round(3).to_dict())
cell = s3a.groupby(["model_label", "group_id", "variant"]).correct.mean().unstack()
rows = []
for m in MODELS:
    diff = (cell.loc[m].nohyper - cell.loc[m].hyper).dropna()
    rows.append(dict(model=m, pairs=len(diff), drop=diff.mean(), p=perm_p(diff.values)))
t = pd.DataFrame(rows); t["p_holm"] = holm(t.p); print(t.round(4).to_string(index=False))

# ---------------------------------------------------------------- 정답 기준 점검
section("4.13절. 모어 화자 응답으로 나눈 쌍의 하락폭")
pool = {i["item_id"]: i["group_id"] for i in json.load(open("items/item_pool_v0.3.json", encoding="utf-8"))["items"]}
hum = {}
h1 = pd.read_csv(R + "human_item_summary.csv", encoding="utf-8-sig"); h1 = h1[h1.subset == "S3a_referent"]
for _, r in h1.iterrows():
    dist = json.loads(r["dist"]); gold_n = dist.get(r.draft_gold, 0)
    hum[pool[r.item_id]] = (gold_n, max(v for k, v in dist.items() if k != r.draft_gold) if len(dist) > 1 else 0)
h2 = pd.read_csv(R + "human_s3a2_items.csv", encoding="utf-8-sig")
for _, r in h2.iterrows():
    hum[r.group] = (r.gold_n, r.hyp_n)
H = pd.DataFrame(hum, index=["gold_n", "hyper_n"]).T
drop = (cell.groupby("group_id").mean()); drop = drop.nohyper - drop.hyper
for name, idx in [("상위어 다수(7명 이상) 9쌍을 뺀 쌍", H.index[H.hyper_n < 7]), ("초안 정답 7명 이상인 쌍", H.index[H.gold_n >= 7])]:
    v = drop.reindex(idx); lo, hi = boot_ci(v.values)
    md = cell[cell.index.get_level_values("group_id").isin(idx)].groupby("model_label").mean()
    print(f"{name}: {len(v)}쌍, 하락폭 {v.mean():.3f} [{lo:.2f}, {hi:.2f}], 하락 {int((v > 1e-9).sum())}쌍, 같음 {int((v.abs() < 1e-9).sum())}쌍, "
          f"p = {perm_p(v.values):.2g}, 하락 모델 {int(((md.nohyper - md.hyper) > 0).sum())}개")

# ---------------------------------------------------------------- 온도 0
section("4.11절. 온도 0 (5개 모델, S3a 64문항, 3회)")
t0 = pd.read_csv(R + "temp0_scored_compact.csv", encoding="utf-8-sig"); t0 = t0[t0.correct.notna()]
m5 = sorted(t0.model_label.unique())
a = t0.groupby(["model_label", "variant"]).correct.mean().unstack(); a["drop"] = a.nohyper - a.hyper
b = s3a[s3a.model_label.isin(m5)].groupby(["model_label", "variant"]).correct.mean().unstack(); b["drop"] = b.nohyper - b.hyper
print(pd.concat([b, a], axis=1, keys=["기본 온도", "온도 0"]).round(3).to_string())
print("평균", b.mean().round(3).to_dict(), a.mean().round(3).to_dict(), "분석 호출 수", len(t0))
print("3회 응답이 모두 같은 비율", t0.groupby(["model_label", "item_id"]).correct.agg(lambda s: s.nunique() == 1).groupby("model_label").mean().round(2).to_dict())

# ---------------------------------------------------------------- 표 8: 검색 기록
section("표 8과 4.6절. 검색 기록(7회차)")
g = pd.read_csv(R + "search_log_summary.csv", encoding="utf-8-sig").fillna("")
for m, x in g.groupby("model_label"):
    print(m, "응답", len(x), "응답당 검색어", round(x.n_queries.mean(), 2), "검색어 평균 길이", round(x.query_chars.mean(), 1),
          "JSON 지시 포함", round(x.query_has_json_instruction.mean(), 3), "응답당 결과", round(x.n_results.mean(), 1),
          "인용이 기록된 응답", int((x.n_citations > 0).sum()))


def cat(h):
    if "wikipedia.org" in h: return "위키백과"
    if "namu.wiki" in h: return "나무위키"
    if "wordrow" in h: return "단어장 사이트(wordrow.kr)"
    if "wiktionary" in h: return "위키낱말사전"
    if h == "krdict.korean.go.kr": return "국립국어원 한국어기초사전"
    if h in ("stdict.korean.go.kr", "opendict.korean.go.kr"): return "표준국어대사전·우리말샘"
    return "그 밖의 사이트"


cl = g[g.model_label.str.startswith("claude")]
n_res = defaultdict(int); n_resp = defaultdict(int); total = 0; others = set()
for hosts in cl.result_hosts:
    hs = [h for h in hosts.split(";") if h]; total += len(hs)
    for h in hs:
        n_res[cat(h)] += 1
        if cat(h) == "그 밖의 사이트": others.add(h)
    for c in {cat(h) for h in hs}: n_resp[c] += 1
for c in ["위키백과", "나무위키", "단어장 사이트(wordrow.kr)", "위키낱말사전", "국립국어원 한국어기초사전", "표준국어대사전·우리말샘", "그 밖의 사이트"]:
    print(f"{c}: 결과 {n_res[c]} ({n_res[c] / total * 100:.1f}%), 응답 {n_resp[c]} ({n_resp[c] / len(cl) * 100:.1f}%)")
print("합계 결과", total, "응답", len(cl), "그 밖의 사이트 수", len(others))

# ---------------------------------------------------------------- 선택지 위치(사후 점검, 5.1절)
section("5.1절. S3a 상위어 있음 문항에서 선택지 위치와 상위어 선택률")
pos = pd.read_csv(R + "s3a_option_position.csv", encoding="utf-8-sig")
sp = s3a[s3a.variant == "hyper"].merge(pos, on="trial_id")
sp["after"] = sp.hyper_pos > sp.gold_pos
print("분석 호출 수", len(sp))
print("상위어 위치별 선택률", sp.groupby("hyper_pos").chose_hyper.mean().round(3).to_dict())
def pos_test(x, name):
    g = x.groupby(["model_label", "item_id", "after"]).chose_hyper.mean().unstack().dropna()
    it = (g[True] - g[False]).groupby("item_id").mean()
    print(f"{name}: 상위어가 정답 뒤 {x[x.after].chose_hyper.mean():.3f}, 앞 {x[~x.after].chose_hyper.mean():.3f}, "
          f"문항 {len(it)}개, 짝지은 차이 {it.mean():.3f}, p = {perm_p(it.values, unit=360000):.3g}")
pos_test(sp, "10개 모델"); pos_test(sp[sp.model_label == "gpt-4o"], "GPT-4o"); pos_test(sp[sp.model_label != "gpt-4o"], "GPT-4o 제외 9개")
