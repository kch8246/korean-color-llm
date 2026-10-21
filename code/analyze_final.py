#!/usr/bin/env python3
"""
최종 분석: 여러 태그의 결과를 합쳐 rep=1(모든 문항·모델 공통)으로 비교표 생성.
사용: python analyze_final.py trials.jsonl item_pool_v0.2.json out_dir
출력(CSV): table_overall, table_by_subset, table_contrasts, table_s7_ko_en, table_s8_hyper_follow,
           table_search_effect, table_s0_anchor, table_ratings_s2, table_ratings_s6, table_ratings_corr, table_s9
"""
import json, math, os, re, sys
from itertools import combinations
import numpy as np
import pandas as pd

OPEN_SETS = {"S0", "S9"}

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))

def norm(s): return re.sub(r"\s+", "", str(s)).strip(" .,'\"")

def exact_sign_p(b, c):
    """McNemar 정확 검정(양측): b, c = 불일치 쌍 수"""
    n = b+c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k+1))/2**n
    return min(1.0, 2*p)

def tau(pred, gold):
    if set(pred) != set(gold) or len(pred) < 2: return np.nan
    pos = {x: i for i, x in enumerate(pred)}; c = d = 0
    for a, b in combinations(gold, 2):
        if pos[a] < pos[b]: c += 1
        else: d += 1
    return (c-d)/(c+d)

def load(trials, pool_path):
    df = pd.read_json(trials, lines=True)
    pool = json.load(open(pool_path, encoding="utf-8"))["items"]
    P = pd.DataFrame(pool)[["item_id","gold","gold_confidence","pair_id","base_item_id","options","version_added"]].rename(columns={"options":"pool_options"})
    df = df[df.model_label != "qwen3.8-27b"]
    df = df[df.error.isna()].sort_values("ts").drop_duplicates("trial_id", keep="last")
    df = df[df.rep == 1].merge(P, on="item_id", how="left")
    return df, {i["item_id"]: i for i in pool}

def score(df):
    cor, taus, ans = [], [], []
    for r in df.itertuples():
        p = r.parsed if isinstance(r.parsed, dict) else {}
        g = r.gold if isinstance(r.gold, str) else ""
        c = t = np.nan; a = None
        if r.task_type == "choice":
            a = p.get("answer")
            if g and a is not None: c = float(norm(a) == norm(g))
        elif r.task_type == "rank":
            rk = p.get("ranking")
            if isinstance(rk, list):
                pred = [norm(x) for x in rk]; a = ">".join(pred)
                if g:
                    gold = [norm(x) for x in g.split(">")]
                    t = tau(pred, gold)
                    c = float(pred == gold) if set(pred) == set(gold) else np.nan
        elif r.task_type == "rating":
            a = p.get("score")
        cor.append(c); taus.append(t); ans.append(a)
    df = df.copy(); df["correct"], df["tau"], df["answer"] = cor, taus, ans
    return df

def acc_table(d, by):
    rows = []
    for key, g in d.groupby(by):
        n = int(g.correct.notna().sum()); k = int(g.correct.sum())
        lo, hi = wilson(k, n)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "n": n, "correct": k,
                     "acc": round(k/n, 3) if n else np.nan, "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    return pd.DataFrame(rows)

def main(trials, pool_path, out):
    os.makedirs(out, exist_ok=True)
    df, by_id = load(trials, pool_path)
    df = score(df)
    df["subset_full"] = df.set_id + ":" + df.subset
    gradable = df[df.correct.notna() & ~df.set_id.isin(["S8"])]
    # 1) 전체 정답률 (S8 제외: S8은 조건 연구)
    t_over = acc_table(gradable[gradable.set_id.isin(["S1","S3","S4"])], ["model_label"])
    t_over.to_csv(f"{out}/table_overall_S1S3S4.csv", index=False, encoding="utf-8-sig")
    # 2) 하위 세트별
    t_sub = acc_table(gradable, ["subset_full", "model_label"])
    t_sub.pivot(index="subset_full", columns="model_label", values="acc").to_csv(f"{out}/table_by_subset.csv", encoding="utf-8-sig")
    # 3) 쌍 대비: S3a hyper vs nohyper, S3b trap vs control (모델별, 틀 단위 McNemar)
    rows = []
    for (ss, a_var, b_var) in [("S3a_referent","hyper","nohyper"),("S3b_lexical_trap","trap","control")]:
        for m, g in df[(df.subset==ss)&df.correct.notna()].groupby("model_label"):
            wide = g.pivot_table(index="group_id", columns="variant", values="correct", aggfunc="first").dropna()
            x, y = wide[a_var], wide[b_var]
            b = int(((x==1)&(y==0)).sum()); c = int(((x==0)&(y==1)).sum())
            rows.append(dict(subset=ss, model_label=m, cond_a=a_var, acc_a=round(x.mean(),3), cond_b=b_var, acc_b=round(y.mean(),3),
                             diff=round(x.mean()-y.mean(),3), n_groups=len(wide), a_ok_b_fail=b, a_fail_b_ok=c, p_mcnemar=round(exact_sign_p(b,c),4)))
    pd.DataFrame(rows).to_csv(f"{out}/table_contrasts.csv", index=False, encoding="utf-8-sig")
    # 4) S7 한국어 vs 영어 지시문 (pair_id로 짝)
    s7 = df[df.set_id=="S7"][["model_label","item_id","pair_id","correct","tau"]].rename(columns={"correct":"c_en"})
    ko = df[["model_label","item_id","correct"]].rename(columns={"item_id":"pair_id","correct":"c_ko"})
    j = s7.merge(ko, on=["model_label","pair_id"]).dropna(subset=["c_en","c_ko"])
    rows = []
    for m, g in j.groupby("model_label"):
        b = int(((g.c_ko==1)&(g.c_en==0)).sum()); c = int(((g.c_ko==0)&(g.c_en==1)).sum())
        rows.append(dict(model_label=m, n_pairs=len(g), acc_ko=round(g.c_ko.mean(),3), acc_en=round(g.c_en.mean(),3),
                         diff_ko_minus_en=round(g.c_ko.mean()-g.c_en.mean(),3), ko_ok_en_fail=b, ko_fail_en_ok=c, p_mcnemar=round(exact_sign_p(b,c),4)))
    pd.DataFrame(rows).to_csv(f"{out}/table_s7_ko_en.csv", index=False, encoding="utf-8-sig")
    # 5) S8: 상위어 추종률 (상위어 = hyper 선택지 - nohyper 선택지)
    hyper_of = {}
    for it in by_id.values():
        if it["subset"]=="S3a_referent" and it["variant"]=="hyper":
            nh = next(x for x in by_id.values() if x["subset"]=="S3a_referent" and x["group_id"]==it["group_id"] and x["variant"]=="nohyper")
            hyper_of[it["item_id"]] = (set(it["options"]) - set(nh["options"])).pop()
    s8 = df[df.set_id=="S8"].copy()
    s8["hyper_word"] = s8.base_item_id.map(hyper_of)
    s8["chose_hyper"] = [float(norm(a)==norm(h)) if a is not None and isinstance(h,str) else np.nan for a, h in zip(s8.answer, s8.hyper_word)]
    s8["chose_gold"] = s8.correct
    # 기준선(문서 없음): 같은 모델의 S3a hyper 문항
    base = df[(df.subset=="S3a_referent")&(df.variant=="hyper")][["model_label","item_id","answer"]].rename(columns={"item_id":"base_item_id","answer":"base_answer"})
    base["base_hyper"] = base.base_item_id.map(hyper_of)
    base["chose_hyper"] = [float(norm(a)==norm(h)) for a, h in zip(base.base_answer, base.base_hyper)]
    base["variant"] = "no_doc"
    b2 = base.merge(df[(df.subset=="S3a_referent")&(df.variant=="hyper")][["model_label","item_id","correct"]].rename(columns={"item_id":"base_item_id"}), on=["model_label","base_item_id"])
    b2 = b2.rename(columns={"correct":"chose_gold"})
    allc = pd.concat([s8[["model_label","base_item_id","variant","chose_hyper","chose_gold"]], b2[["model_label","base_item_id","variant","chose_hyper","chose_gold"]]])
    order = ["no_doc","irrelevant","support","biased_1","biased_3"]
    t8 = allc.groupby(["model_label","variant"]).agg(n=("chose_gold","size"), acc_gold=("chose_gold","mean"), hyper_rate=("chose_hyper","mean")).round(3).reset_index()
    t8["variant"] = pd.Categorical(t8.variant, order, ordered=True)
    t8 = t8.sort_values(["model_label","variant"])
    t8.to_csv(f"{out}/table_s8_hyper_follow.csv", index=False, encoding="utf-8-sig")
    t8.pivot(index="model_label", columns="variant", values="hyper_rate").to_csv(f"{out}/table_s8_hyper_rate_wide.csv", encoding="utf-8-sig")
    # 6) 검색 효과 (S3, S4, 동일 문항 짝)
    rows = []
    for plain, srch in [("gpt-4o","gpt-4o+websearch"),("claude-sonnet-5-5","claude-sonnet-5-5+websearch")]:
        a = df[(df.model_label==plain)&df.set_id.isin(["S3","S4"])][["item_id","correct"]].rename(columns={"correct":"c_plain"})
        b = df[(df.model_label==srch)&df.set_id.isin(["S3","S4"])][["item_id","correct","search_calls"]].rename(columns={"correct":"c_search"})
        j = a.merge(b, on="item_id").dropna(subset=["c_plain","c_search"])
        x = int(((j.c_plain==1)&(j.c_search==0)).sum()); y = int(((j.c_plain==0)&(j.c_search==1)).sum())
        rows.append(dict(model=plain, n=len(j), acc_plain=round(j.c_plain.mean(),3), acc_search=round(j.c_search.mean(),3),
                         search_used_rate=round((j.search_calls>0).mean(),3), plain_ok_search_fail=x, plain_fail_search_ok=y, p_mcnemar=round(exact_sign_p(x,y),4)))
    pd.DataFrame(rows).to_csv(f"{out}/table_search_effect.csv", index=False, encoding="utf-8-sig")
    # 7) S0 앵커 (1차 논문 재현)
    s0 = df[df.set_id=="S0"][["model_label","item_id","answer","response_text"]]
    q1 = s0[s0.item_id=="S0-001"].copy()
    def q1order(t):
        t = str(t); idx = {w: t.find(w) for w in ["까맣다","검다","거무스름하다"]}
        if min(idx.values()) < 0: return "추출 불가"
        return ">".join(w for w, _ in sorted(idx.items(), key=lambda x: x[1]))
    q1["first_mention_order"] = q1.response_text.map(q1order)
    q4 = s0[s0.item_id=="S0-004"].copy()
    def q4ans(t):
        t = str(t)
        for w in ["밤색","갈색","회색","은색"]:
            if w in t[:60]: return w
        return "추출 불가"
    q4["first_mention"] = q4.response_text.map(q4ans)
    out0 = q1[["model_label","first_mention_order"]].merge(q4[["model_label","first_mention"]], on="model_label")
    out0.to_csv(f"{out}/table_s0_anchor.csv", index=False, encoding="utf-8-sig")
    # 8) 평정: S2 변이형 평균, S6 어종x상황, 모델 간 상관
    rt = df[df.task_type=="rating"].copy(); rt["score"] = pd.to_numeric(rt.answer, errors="coerce")
    s2 = rt[rt.set_id=="S2"].groupby(["subset","variant","model_label"]).score.mean().round(2).unstack("model_label")
    s2.to_csv(f"{out}/table_ratings_s2.csv", encoding="utf-8-sig")
    s6 = rt[rt.set_id=="S6"].copy(); s6["ctx"] = s6.group_id.str.split("-").str[1]; s6["color"] = s6.group_id.str.split("-").str[0]
    s6.groupby(["ctx","variant","model_label"]).score.mean().round(2).unstack("model_label").to_csv(f"{out}/table_ratings_s6.csv", encoding="utf-8-sig")
    wide = rt.pivot_table(index="item_id", columns="model_label", values="score")
    wide.corr(method="spearman").round(2).to_csv(f"{out}/table_ratings_corr.csv", encoding="utf-8-sig")
    # 9) S9 (휴리스틱): 거짓 전제 반박 여부, 답 변경
    s9 = df[df.set_id=="S9"].copy()
    rebut = re.compile(r"(정답은 .{0,20}(아닙니다|아니라)|사실과 다르|오류|잘못|엄밀히|실제로는|그렇지 않|동의하기 어렵|오히려|반대)")
    s9["rebuts"] = s9.response_text.map(lambda t: bool(rebut.search(str(t))))
    t9 = s9.groupby(["model_label","variant"]).rebuts.mean().round(2).unstack("variant")
    t9.to_csv(f"{out}/table_s9_rebuttal_heuristic.csv", encoding="utf-8-sig")
    # 품질
    q = df.groupby("model_label").agg(n=("trial_id","size"), parse_fail=("parse_ok", lambda s: int((s==False).sum())),
        empty=("response_text", lambda s: int((s.fillna("").str.strip()=="").sum())),
        truncated=("finish_reason", lambda s: int(s.isin(["max_tokens","length","MAX_TOKENS"]).sum()))).reset_index()
    q.to_csv(f"{out}/table_quality.csv", index=False, encoding="utf-8-sig")
    df[["trial_id","model_label","item_id","set_id","subset","variant","group_id","task_type","answer","gold","correct","tau"]].to_csv(f"{out}/scored_rep0.csv", index=False, encoding="utf-8-sig")
    return dict(overall=t_over, sub=t_sub, contrasts=pd.DataFrame(rows), s8=t8, quality=q)

if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 3: print(__doc__); sys.exit(1)
    main(*a[:3])
