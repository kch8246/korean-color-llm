#!/usr/bin/env python3
"""
파일럿/본 실험 결과 채점·요약 (pandas + numpy만 사용)

사용(주피터):
    import analyze_pilot as ap
    ap.main(trials="results/trials.jsonl", pool="item_pool_v0.1.json", tag="pilot5", out="analysis")

만드는 파일(out 폴더):
    scored.csv            호출 단위 채점 결과
    acc_by_model.csv      모델별 정답률 + Wilson 95% CI (choice/rank 전체)
    acc_by_set.csv        모델 x 세트/하위세트 정답률
    consistency.csv       같은 문항 반복(rep) 간 일관성
    contrasts.csv         조건 대비(hyper/nohyper, trap/control, ko/en, S8, S9)
    rating_means.csv      평정 문항 모델별 평균/SD (설문 평균과 비교할 자료)
    quality.csv           오류·빈 응답·파싱 실패·잘림
"""
import json
import math
import os
import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd

FREE_SETS = {"S0", "S9"}


# ---------------------------------------------------------------- 유틸
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def norm(s):
    return re.sub(r"\s+", "", str(s)).strip(" .,'\"「」『』")


def kendall_tau(pred, gold):
    """pred, gold: 같은 원소의 순서 리스트. 반환 -1~1, 원소가 다르면 nan"""
    if set(pred) != set(gold) or len(pred) < 2:
        return np.nan
    pos = {x: i for i, x in enumerate(pred)}
    conc = disc = 0
    for a, b in combinations(gold, 2):  # gold 순서상 a가 b보다 앞
        if pos[a] < pos[b]:
            conc += 1
        else:
            disc += 1
    return (conc - disc) / (conc + disc)


# ---------------------------------------------------------------- 채점
def load(trials, pool, tag=None):
    rows = [json.loads(l) for l in open(trials, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    if tag:
        df = df[df.run_tag == tag].copy()
    pj = json.load(open(pool, encoding="utf-8"))
    p = pd.DataFrame(pj["items"]).rename(columns={"item_id": "item_id"})
    keep = ["item_id", "gold", "gold_confidence", "gold_type", "hypothesis_id", "pair_id", "base_item_id", "options", "premise_option"]
    df = df.merge(p[keep].rename(columns={"options": "pool_options", "variant": "v"}), on="item_id", how="left")
    # 성공 기록만 남기고 같은 trial이 중복이면 마지막 것을 사용
    df["is_err"] = df.error.notna()
    df = df.sort_values("ts").drop_duplicates("trial_id", keep="last")
    return df


def score_row(r):
    if r.is_err or not isinstance(r.parsed, dict):
        return pd.Series({"correct": np.nan, "tau": np.nan, "answer": None})
    t = r.task_type
    g = r.gold if isinstance(r.gold, str) else ""
    if t == "choice" and g:
        a = r.parsed.get("answer")
        return pd.Series({"correct": float(norm(a) == norm(g)) if a is not None else np.nan, "tau": np.nan, "answer": a})
    if t == "rank" and g:
        rk = r.parsed.get("ranking")
        gold = [norm(x) for x in g.split(">")]
        if isinstance(rk, list):
            pred = [norm(x) for x in rk]
            tau = kendall_tau(pred, gold)
            return pd.Series({"correct": float(pred == gold) if set(pred) == set(gold) else np.nan, "tau": tau, "answer": ">".join(pred)})
        return pd.Series({"correct": np.nan, "tau": np.nan, "answer": None})
    if t == "rating":
        return pd.Series({"correct": np.nan, "tau": np.nan, "answer": r.parsed.get("score")})
    return pd.Series({"correct": np.nan, "tau": np.nan, "answer": None})


def table_acc(d, by):
    out = []
    for key, g in d.groupby(by):
        n = int(g.correct.notna().sum())
        k = int(g.correct.sum())
        lo, hi = wilson(k, n)
        row = dict(zip(by if isinstance(by, list) else [by], key if isinstance(key, tuple) else (key,)))
        row.update(n=n, correct=k, acc=round(k / n, 3) if n else np.nan, ci_lo=round(lo, 3), ci_hi=round(hi, 3))
        out.append(row)
    return pd.DataFrame(out)


def consistency(d):
    """문항별 5회 반복의 최빈 응답 비율의 평균(1이면 완전 일관)."""
    rows = []
    for (m, it), g in d[d.answer.notna() & (d.task_type.isin(["choice", "rank"]))].groupby(["model_label", "item_id"]):
        vc = g.answer.astype(str).value_counts()
        rows.append(dict(model_label=m, item_id=it, n=len(g), modal_share=vc.iloc[0] / len(g)))
    c = pd.DataFrame(rows)
    return c.groupby("model_label").agg(items=("item_id", "nunique"), mean_modal_share=("modal_share", "mean"),
                                        frac_fully_consistent=("modal_share", lambda s: (s == 1).mean())).round(3).reset_index()


def contrasts(d):
    """조건 대비: 같은 모델 안에서 variant 간 정답률 차이."""
    rows = []
    cond_sets = [("S3", "a"), ("S3", "b")]
    d = d.copy()
    d["subset_key"] = d.set_id + d.subset.fillna("").astype(str)
    for (m, sk), g in d[d.task_type == "choice"].groupby(["model_label", "subset_key"]):
        if g.variant.nunique() < 2 or g.correct.notna().sum() == 0:
            continue
        acc = g.groupby("variant").correct.agg(["mean", "count"]).round(3)
        rows.append(dict(model_label=m, subset=sk, **{f"acc_{v}": acc.loc[v, "mean"] for v in acc.index},
                         **{f"n_{v}": int(acc.loc[v, "count"]) for v in acc.index}))
    return pd.DataFrame(rows)


def s8_hyper_rate(d):
    """S8: 문서 조건별 상위어 선택률. gold(구체색) 아닌 답 중 상위어(options 마지막이 아닌 gold 외 첫 선택지)는 사전에 정의 필요 → 오답률로 대체."""
    g = d[(d.set_id == "S8") & d.correct.notna()]
    if g.empty:
        return pd.DataFrame()
    return g.groupby(["model_label", "variant"]).correct.agg(acc="mean", n="count").round(3).reset_index()


def quality(df):
    df = df.copy()
    df["empty"] = (~df.is_err) & (df.response_text.fillna("").str.strip() == "")
    df["pf"] = (~df.is_err) & (~df.set_id.isin(FREE_SETS)) & (df.parse_ok == False)  # noqa: E712
    df["trunc"] = df.finish_reason.isin(["max_tokens", "length", "MAX_TOKENS"])
    return df.groupby("model_label").agg(n=("trial_id", "size"), errors=("is_err", "sum"), empty=("empty", "sum"),
                                         parse_fail=("pf", "sum"), truncated=("trunc", "sum"),
                                         tok_out=("tokens_out", "mean"), latency=("latency_s", "mean")).round(2).reset_index()


# ---------------------------------------------------------------- 실행
def main(trials="results/trials.jsonl", pool="item_pool_v0.1.json", tag=None, out="analysis"):
    os.makedirs(out, exist_ok=True)
    df = load(trials, pool, tag)
    sc = df.apply(score_row, axis=1)
    df = pd.concat([df, sc], axis=1)

    keep = ["trial_id", "model_label", "item_id", "set_id", "subset", "variant", "rep", "task_type", "answer", "gold", "correct", "tau", "parse_ok", "finish_reason"]
    df[keep].to_csv(f"{out}/scored.csv", index=False, encoding="utf-8-sig")

    gradable = df[df.task_type.isin(["choice", "rank"]) & df.gold.notna() & (df.gold != "")]
    a1 = table_acc(gradable, ["model_label"])
    a1.to_csv(f"{out}/acc_by_model.csv", index=False, encoding="utf-8-sig")
    a2 = table_acc(gradable, ["model_label", "set_id", "subset"])
    a2.to_csv(f"{out}/acc_by_set.csv", index=False, encoding="utf-8-sig")
    cons = consistency(df)
    cons.to_csv(f"{out}/consistency.csv", index=False, encoding="utf-8-sig")
    con = contrasts(df)
    con.to_csv(f"{out}/contrasts.csv", index=False, encoding="utf-8-sig")
    rat = (df[df.task_type == "rating"].assign(score=lambda x: pd.to_numeric(x.answer, errors="coerce"))
           .groupby(["model_label", "item_id"]).score.agg(["mean", "std", "count"]).round(2).reset_index())
    rat.to_csv(f"{out}/rating_means.csv", index=False, encoding="utf-8-sig")
    q = quality(df)
    q.to_csv(f"{out}/quality.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    print("\n[품질]\n", q.to_string(index=False))
    print("\n[모델별 정답률 (choice+rank 정답이 있는 문항)]\n", a1.sort_values("acc", ascending=False).to_string(index=False))
    print("\n[반복 일관성]\n", cons.to_string(index=False))
    print("\n[조건 대비]\n", con.to_string(index=False) if len(con) else "(없음)")
    pv = a2.pivot_table(index=["set_id", "subset"], columns="model_label", values="acc")
    print("\n[세트·하위세트별 정답률]\n", pv.round(2).to_string())
    return df


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(a[:4]))
