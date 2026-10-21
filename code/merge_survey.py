#!/usr/bin/env python3
"""
설문 응답(JSON) 취합 -> 문항 풀에 human_* 열 채우기 + 정답 점검 표

    python merge_survey.py <응답 폴더> [item_pool_v0.3.json] [출력 폴더]

필요: pandas, numpy

만드는 파일(출력 폴더, 기본 survey_out):
    item_pool_v0.2_human.json   human_mean/sd/n(평정), human_gold(선택·순위) 채운 문항 풀
    human_item_summary.csv      문항별 요약(분포, 최빈 응답, 일치율, 모르는 표현 비율)
    gold_audit.csv              연구자 초안 gold와 모어 화자 최빈 응답이 다른 문항(점검 대상)
    respondents.csv             응답자별 소요 시간, 주의 확인 통과, 제외 사유
"""
import glob
import json
import os
import sys
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

MIN_SECONDS = 240  # 이보다 빨리 끝낸 응답은 성의 없는 응답으로 제외
MIN_AGREE_N = 10   # 이보다 응답자가 적은 문항은 정답 판정을 보류


def norm(s):
    return "".join(str(s).split())


def tau(pred, gold):
    if set(pred) != set(gold) or len(pred) < 2:
        return np.nan
    pos = {x: i for i, x in enumerate(pred)}
    c = d = 0
    for a, b in combinations(gold, 2):
        if pos[a] < pos[b]:
            c += 1
        else:
            d += 1
    return (c - d) / (c + d)


def load_responses(folder):
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "*.json")) + glob.glob(os.path.join(folder, "*.txt"))):
        try:
            raw = open(p, encoding="utf-8-sig").read()
            r = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])  # 카톡/메일에 붙여 넣은 텍스트도 허용
            r["_file"] = os.path.basename(p)
            rows.append(r)
        except Exception as e:  # noqa: BLE001
            print("읽기 실패:", p, e)
    return rows


def screen(resp):
    """응답자 단위 제외 규칙. 반환: (통과 여부, 사유 리스트, 소요 초)"""
    why = []
    m = resp.get("meta", {})
    if not (m.get("consent") and m.get("native_ko") and m.get("adult")):
        why.append("동의/모어/성인 확인 누락")
    ans = {a["item_id"]: a for a in resp.get("answers", [])}
    if resp.get("form") != "core" and ans.get("ATTN-R", {}).get("answer") != 2:
        why.append("주의 확인(평정) 실패")
    if ans.get("ATTN-C", {}).get("answer") != "파란색":
        why.append("주의 확인(선택) 실패")
    try:
        t0 = pd.Timestamp(resp["started"])
        t1 = pd.Timestamp(resp["finished"])
        secs = (t1 - t0).total_seconds()
    except Exception:  # noqa: BLE001
        secs = np.nan
    if secs == secs and secs < MIN_SECONDS:
        why.append(f"너무 빠름({secs:.0f}초)")
    return (not why), why, secs


def main(folder, pool_path="item_pool_v0.3.json", out="survey_out"):
    os.makedirs(out, exist_ok=True)
    pool = json.load(open(pool_path, encoding="utf-8"))
    items = pool["items"]
    by_id = {i["item_id"]: i for i in items}

    resps = load_responses(folder)
    rrows, good = [], []
    for r in resps:
        ok, why, secs = screen(r)
        rrows.append(dict(file=r["_file"], respondent=r.get("respondent_id"), form=r.get("form"), age=r.get("meta", {}).get("age"),
                          seconds=round(secs) if secs == secs else None, n_answers=len(r.get("answers", [])), included=ok,
                          excluded_because=";".join(why)))
        if ok:
            good.append(r)
    rdf = pd.DataFrame(rrows)
    rdf.to_csv(os.path.join(out, "respondents.csv"), index=False, encoding="utf-8-sig")
    print(f"응답 {len(resps)}명 중 포함 {len(good)}명")
    if not good:
        print("포함할 응답이 없어 종료합니다. respondents.csv의 제외 사유를 확인하세요.")
        return

    # 문항 단위로 모으기
    long = []
    for r in good:
        for a in r["answers"]:
            if a["item_id"].startswith("ATTN"):
                continue
            typ = a.get("type") or a.get("tpe") or by_id.get(a["item_id"], {}).get("task_type")  # 키 오타(tpe)가 섞인 응답 허용
            long.append(dict(respondent=r["respondent_id"], item_id=a["item_id"], type=typ, answer=a["answer"],
                             unfamiliar=a.get("unfamiliar", False), rt_ms=a.get("rt_ms")))
    L = pd.DataFrame(long)

    summ = []
    for iid, g in L.groupby("item_id"):
        it = by_id[iid]
        n = len(g)
        row = dict(item_id=iid, subset=it["subset"], variant=it["variant"], task_type=it["task_type"], n=n,
                   unfamiliar_rate=round(g.unfamiliar.mean(), 3), median_rt_s=round(g.rt_ms.median() / 1000, 1),
                   draft_gold=it["gold"], gold_confidence=it["gold_confidence"])
        if it["task_type"] == "rating":
            sc = pd.to_numeric(g.answer, errors="coerce").dropna()
            row.update(human_mean=round(sc.mean(), 3), human_sd=round(sc.std(ddof=1), 3) if len(sc) > 1 else np.nan,
                       human_gold="", agree=np.nan, dist=json.dumps(sc.value_counts().sort_index().to_dict()))
        elif it["task_type"] == "choice":
            c = Counter(g.answer)
            top, k = c.most_common(1)[0]
            row.update(human_mean=np.nan, human_sd=np.nan, human_gold=top, agree=round(k / n, 3), dist=json.dumps(dict(c), ensure_ascii=False))
        else:  # rank: 평균 순위로 집계
            ranks = {}
            for a in g.answer:
                for pos, o in enumerate(a):
                    ranks.setdefault(o, []).append(pos + 1)
            mean_rank = {o: np.mean(v) for o, v in ranks.items()}
            order = [o for o, _ in sorted(mean_rank.items(), key=lambda x: x[1])]
            taus = [tau([x for x in a], order) for a in g.answer]
            row.update(human_mean=np.nan, human_sd=np.nan, human_gold=">".join(order), agree=round(float(np.nanmean(taus)), 3),
                       dist=json.dumps({o: round(v, 2) for o, v in mean_rank.items()}, ensure_ascii=False))
            if it["gold"]:
                gt = it["gold"].split(">")
                row["tau_human_vs_draft"] = round(tau(order, gt), 3)
        summ.append(row)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(out, "human_item_summary.csv"), index=False, encoding="utf-8-sig")

    # 정답 점검: 선택 문항은 최빈 응답, 순위 문항은 순서 비교
    audit = []
    for _, r in S.iterrows():
        if not r.draft_gold or r.n < MIN_AGREE_N:
            continue
        if r.task_type == "choice" and norm(r.human_gold) != norm(r.draft_gold):
            audit.append(dict(item_id=r.item_id, subset=r.subset, draft_gold=r.draft_gold, human_modal=r.human_gold, agree=r.agree, n=r.n,
                              note="모어 화자 최빈 응답이 초안 정답과 다름"))
        elif r.task_type == "choice" and r.agree < 0.6:
            audit.append(dict(item_id=r.item_id, subset=r.subset, draft_gold=r.draft_gold, human_modal=r.human_gold, agree=r.agree, n=r.n,
                              note="초안과 같지만 일치율 60% 미만: 정답이 불분명"))
        elif r.task_type == "rank" and norm(r.human_gold) != norm(r.draft_gold):
            audit.append(dict(item_id=r.item_id, subset=r.subset, draft_gold=r.draft_gold, human_modal=r.human_gold, agree=r.agree, n=r.n,
                              note=f"순위가 다름(τ={r.get('tau_human_vs_draft')})"))
    A = pd.DataFrame(audit)
    A.to_csv(os.path.join(out, "gold_audit.csv"), index=False, encoding="utf-8-sig")

    # 문항 풀에 채우기
    sm = S.set_index("item_id")
    for it in items:
        iid = it["item_id"]
        if iid in sm.index:
            r = sm.loc[iid]
            it["human_n"] = int(r.n)
            if it["task_type"] == "rating":
                it["human_mean"], it["human_sd"] = float(r.human_mean), (None if pd.isna(r.human_sd) else float(r.human_sd))
            else:
                it["human_gold"] = r.human_gold
                it["human_agree"] = None if pd.isna(r.agree) else float(r.agree)
    pool["items"] = items
    pool["human_survey"] = {"respondents_included": len(good), "respondents_total": len(resps)}
    json.dump(pool, open(os.path.join(out, "item_pool_v0.2_human.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"문항 요약 {len(S)}행 / 정답 점검 대상 {len(A)}문항 -> {out}")
    if len(A):
        print(A.to_string(index=False))
    return S, A


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(1)
    main(*a[:3])
