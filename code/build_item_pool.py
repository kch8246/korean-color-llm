#!/usr/bin/env python3
"""문항 풀 v0.2 생성 + 검증 스크립트 (CSV/JSON 동시 출력). v0.1 문항 ID는 그대로 유지하고 보강 문항을 뒤에 추가"""
import collections
import csv
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.2-draft"

COLS = [
    "item_id", "set_id", "set_name", "subset", "group_id", "pilot", "task_type", "lang",
    "prompt", "prompt_includes_options", "options", "option_order", "output_schema", "scale",
    "gold", "gold_type", "gold_status", "gold_confidence", "gold_basis", "premise_option",
    "hypothesis_id", "variant", "pair_id", "base_item_id", "context_doc",
    "human_mean", "human_sd", "human_n", "human_gold", "notes", "version_added",
]

SETS = {
    "S0": "앵커(1차 논문 Q1~Q5)",
    "S1": "강도·음상징",
    "S2": "어종별 자연스러움",
    "S3": "지시 대상·어원",
    "S4": "관용·다의",
    "S5": "안색·정서",
    "S6": "어종×레지스터",
    "S7": "교차언어(영어 지시문)",
    "S8": "RAG 충돌",
    "S9": "설명 충실성",
}

ITEMS = []
BY_ID = {}
_cnt = collections.Counter()


def add(set_id, **kw):
    _cnt[set_id] += 1
    row = {c: "" for c in COLS}
    row.update(
        item_id=f"{set_id}-{_cnt[set_id]:03d}", set_id=set_id, set_name=SETS[set_id],
        pilot=False, lang="ko", prompt_includes_options=False, options=[],
        gold_status="draft_unverified",
    )
    row.update(kw)
    ITEMS.append(row)
    BY_ID[row["item_id"]] = row
    return row["item_id"]


SHUF = "shuffle"
SHUF_LAST = "shuffle_except_last"
FIXED = "fixed"

# ------------------------------------------------------------------ S0 앵커
S0 = [
    ("Q1", "open", "'검다', '까맣다', '거무스름하다' 중 가장 시각적으로 어두운 느낌이 강한 단어 순으로 정렬하고 그 이유를 설명하세요.",
     ["검다", "까맣다", "거무스름하다"], "까맣다>검다>거무스름하다", "1차 논문 Q1: 5개 모델 모두 이 순서로 응답"),
    ("Q2", "open", "사람의 안색을 묘사할 때 '칙칙하다'와 '검다'는 감정적, 심리적으로 어떤 뉘앙스 차이가 있습니까?",
     [], "", "정답 없음(질적 분석 문항)"),
    ("Q3", "open", "한국어의 '푸르다'는 파란색(Blue)과 초록색(Green) 중 어느 색상 범주에 속합니까? 문화적 맥락이나 예시를 들어 설명하세요.",
     [], "", "정답 없음(질적 분석 문항). 기대 응답: 두 범주를 모두 포괄(Grue)"),
    ("Q4", "choice", "잘 익은 밤(Chestnut)의 겉껍질 색을 가장 정확하게 지칭하는 단어는 무엇입니까? (갈색, 밤색, 회색, 은색 중 택 1)",
     ["갈색", "밤색", "회색", "은색"], "밤색", "1차 논문 정답(밤색). 모어 화자 설문으로 재확인 필요"),
    ("Q5", "open", "마케팅 용어로서 '럭셔리 블랙'과 '고급스러운 검정'은 소비자에게 각각 어떤 사회적, 감성적 이미지를 전달합니까?",
     [], "", "정답 없음(질적 분석 문항)"),
]
for qid, tt, prompt, opts, gold, note in S0:
    add("S0", subset="S0_anchor", group_id=qid, task_type=tt, prompt=prompt, prompt_includes_options=bool(opts and qid == "Q4"),
        options=opts, option_order=FIXED, output_schema="free", gold=gold,
        gold_type="paper_anchor" if gold else "none", gold_status="paper" if gold else "n/a",
        gold_confidence="medium" if gold else "", gold_basis=note if gold else "",
        hypothesis_id={"Q1": "H3", "Q4": "H12", "Q3": "", "Q2": "", "Q5": ""}[qid],
        notes="1차 논문 문구 그대로. 프롬프트·출력형식 변경 금지(1차 결과와 직접 비교용)")

# ------------------------------------------------------------------ S1 강도·음상징
S1A_PAIRS = [("노랗다", "누렇다"), ("파랗다", "퍼렇다"), ("발갛다", "벌겋다"),
             ("하얗다", "허옇다"), ("보얗다", "부옇다"), ("가맣다", "거멓다")]
S1A_DIMS = [("dark_turbid", "어둡고 탁한", "darker and more dull/murky"),
            ("large_heavy", "크고 묵직한", "larger and heavier")]
s1a_ids = {}
for small, large in S1A_PAIRS:
    for dkey, dko, den in S1A_DIMS:
        rare = small == "가맣다"
        iid = add("S1", subset="S1a_vowel", group_id=f"vowel:{small}-{large}", task_type="choice",
                  prompt=f"다음 두 단어 중, {dko} 느낌이 더 강한 단어는 무엇입니까? ({small}, {large} 중 택 1)",
                  options=[small, large], option_order=SHUF, output_schema="choice", gold=large,
                  gold_type="linguistic", gold_confidence="medium" if rare else "high",
                  gold_basis="큰말(음성모음계)이 작은말(양성모음계)보다 어둡고 크고 무거운 심상(채완 1987; 1차 논문 4.2 인용)",
                  hypothesis_id="H1", variant=dkey,
                  notes="작은말=양성모음, 큰말=음성모음. 예사소리 고정" + (" / '가맣다'는 저빈도어: 빈도 교란 가능" if rare else ""))
        s1a_ids[(small, large, dkey)] = iid

S1B_PAIRS = [("발갛다", "빨갛다"), ("벌겋다", "뻘겋다"), ("거멓다", "꺼멓다"), ("가맣다", "까맣다")]
s1b_ids = {}
for plain, fortis in S1B_PAIRS:
    rare = plain == "가맣다"
    iid = add("S1", subset="S1b_fortis", group_id=f"fortis:{plain}-{fortis}", task_type="choice",
              prompt=f"다음 두 단어 중, 색이 더 강하고 선명한 느낌을 주는 단어는 무엇입니까? ({plain}, {fortis} 중 택 1)",
              options=[plain, fortis], option_order=SHUF, output_schema="choice", gold=fortis,
              gold_type="linguistic", gold_confidence="medium",
              gold_basis="예사소리–된소리(센말) 대립: 된소리가 더 세고 강한 어감(국어학 일반 기술)",
              hypothesis_id="H2", variant="strength",
              notes="모음 고정, 자음 강약만 변화" + (" / '가맣다'는 저빈도어" if rare else ""))
    s1b_ids[(plain, fortis)] = iid

S1C = [
    ("black", "검은색", "black", ["새까맣다", "까맣다", "검다", "거무스름하다"]),
    ("red", "붉은색", "red", ["새빨갛다", "빨갛다", "붉다", "불그스름하다"]),
    ("blue", "푸른색", "blue", ["새파랗다", "파랗다", "파르스름하다"]),
    ("yellow", "노란색", "yellow", ["샛노랗다", "노랗다", "누르스름하다"]),
    ("white", "흰색", "white", ["새하얗다", "하얗다", "희끄무레하다"]),
    ("red_si", "붉은색", "red", ["시뻘겋다", "뻘겋다", "벌겋다", "불그스름하다"]),
]
s1c_color_en = {}
s1c_ids = {}
for gkey, cko, cen, words in S1C:
    iid = add("S1", subset="S1c_affix_rank", group_id=f"rank:{gkey}", task_type="rank",
              prompt=f"다음 한국어 색채 형용사들은 모두 {cko} 계열입니다. 해당 색이 더 짙고 강하게 느껴지는 순서(강한 것 → 약한 것)로 정렬하세요.",
              options=words, option_order=SHUF, output_schema="rank", gold=">".join(words),
              gold_type="linguistic", gold_confidence="medium",
              gold_basis="강조 접두사(새-/샛-/시-) > 기본형 > 약화 접미사(-스름하다/-끄무레하다). 기본형끼리의 순서는 설문으로 확인",
              hypothesis_id="H3", variant="affix", notes="gold는 '강→약' 순서")
    s1c_color_en[iid] = cen
    s1c_ids[gkey] = iid

for gkey, cko, words in [("black_2x2", "검은색", ["가맣다", "까맣다", "거멓다", "꺼멓다"]),
                         ("red_2x2", "붉은색", ["발갛다", "빨갛다", "벌겋다", "뻘겋다"])]:
    add("S1", subset="S1d_2x2_rank", group_id=f"rank:{gkey}", task_type="rank",
        prompt=f"다음 한국어 색채 형용사들은 모두 {cko} 계열입니다. 해당 색이 더 짙고 둔탁하게 느껴지는 순서(강한 것 → 약한 것)로 정렬하세요.",
        options=words, option_order=SHUF, output_schema="rank", gold="", gold_type="survey", gold_status="pending_survey",
        gold_confidence="", gold_basis="모음(양/음)×자음(예사/된소리) 교차. 사전에 정답을 정하지 않고 모어 화자 평정으로 확정",
        hypothesis_id="H1,H2", variant="2x2", notes="모음 효과와 자음 효과를 분리하는 탐색 문항")

# ------------------------------------------------------------------ S2 어종별 자연스러움
S2_FRAMES = [
    ("A1", "S2A_sino_fixed", {"sino": "경기 침체로 우리 경제에 적신호가 켜졌다는 분석이 나왔다.",
                              "native": "경기 침체로 우리 경제에 빨간불이 켜졌다는 분석이 나왔다.",
                              "loan": "경기 침체로 우리 경제에 레드라이트가 켜졌다는 분석이 나왔다."}),
    ("A2", "S2A_sino_fixed", {"sino": "양국 정상회담으로 협상 타결에 청신호가 켜졌다.",
                              "native": "양국 정상회담으로 협상 타결에 파란불이 켜졌다.",
                              "loan": "양국 정상회담으로 협상 타결에 그린라이트가 켜졌다."}),
    ("A3", "S2A_sino_fixed", {"sino": "잠이 잘 오지 않을 때는 백색소음을 틀어 놓으면 도움이 된다.",
                              "native": "잠이 잘 오지 않을 때는 하얀 소음을 틀어 놓으면 도움이 된다.",
                              "loan": "잠이 잘 오지 않을 때는 화이트 노이즈를 틀어 놓으면 도움이 된다."}),
    ("A4", "S2A_sino_fixed", {"sino": "정부는 저탄소 녹색성장 전략을 새로 발표했다.",
                              "native": "정부는 저탄소 푸른 성장 전략을 새로 발표했다.",
                              "loan": "정부는 저탄소 그린 성장 전략을 새로 발표했다."}),
    ("A5", "S2A_sino_fixed", {"sino": "그 매체는 자극적인 보도로 황색언론이라는 비판을 받았다.",
                              "native": "그 매체는 자극적인 보도로 노란 언론이라는 비판을 받았다.",
                              "loan": "그 매체는 자극적인 보도로 옐로 저널리즘이라는 비판을 받았다."}),
    ("A6", "S2A_sino_fixed", {"sino": "이 회사는 냉장고와 세탁기 같은 백색가전 분야에서 점유율 1위다.",
                              "native": "이 회사는 냉장고와 세탁기 같은 하얀 가전 분야에서 점유율 1위다.",
                              "loan": "이 회사는 냉장고와 세탁기 같은 화이트 가전 분야에서 점유율 1위다."}),
    ("B1", "S2B_loan_fixed", {"loan": "그 업체는 불량 거래 이력 때문에 거래처 블랙리스트에 올랐다.",
                              "native": "그 업체는 불량 거래 이력 때문에 거래처 검은 명단에 올랐다.",
                              "sino": "그 업체는 불량 거래 이력 때문에 거래처 흑색 명단에 올랐다."}),
    ("B2", "S2B_loan_fixed", {"loan": "주심은 거친 반칙을 한 선수에게 레드카드를 꺼냈다.",
                              "native": "주심은 거친 반칙을 한 선수에게 빨간 카드를 꺼냈다.",
                              "sino": "주심은 거친 반칙을 한 선수에게 적색 카드를 꺼냈다."}),
    ("B3", "S2B_loan_fixed", {"loan": "저녁 식사에는 스테이크와 어울리는 레드 와인을 골랐다.",
                              "native": "저녁 식사에는 스테이크와 어울리는 붉은 포도주를 골랐다.",
                              "sino": "저녁 식사에는 스테이크와 어울리는 적포도주를 골랐다."}),
    ("B4", "S2B_loan_fixed", {"loan": "사고 원인을 확인하려고 차량 블랙박스 영상을 돌려 보았다.",
                              "native": "사고 원인을 확인하려고 차량 검은 상자 영상을 돌려 보았다.",
                              "sino": "사고 원인을 확인하려고 차량 흑색 상자 영상을 돌려 보았다."}),
    ("C1", "S2C_native_default", {"native": "그는 검은 머리카락을 짧게 잘랐다.",
                                  "sino": "그는 흑색 머리카락을 짧게 잘랐다.",
                                  "loan": "그는 블랙 머리카락을 짧게 잘랐다."}),
    ("C2", "S2C_native_default", {"native": "해 질 녘 서쪽 하늘에 붉은 노을이 번졌다.",
                                  "sino": "해 질 녘 서쪽 하늘에 적색 노을이 번졌다.",
                                  "loan": "해 질 녘 서쪽 하늘에 레드 노을이 번졌다."}),
    ("C3", "S2C_native_default", {"native": "창밖으로 푸른 바다가 끝없이 펼쳐져 있었다.",
                                  "sino": "창밖으로 청색 바다가 끝없이 펼쳐져 있었다.",
                                  "loan": "창밖으로 블루 바다가 끝없이 펼쳐져 있었다."}),
    ("C4", "S2C_native_default", {"native": "밤새 하얀 눈이 소복이 쌓였다.",
                                  "sino": "밤새 백색 눈이 소복이 쌓였다.",
                                  "loan": "밤새 화이트 눈이 소복이 쌓였다."}),
]
EXPECT = {"S2A_sino_fixed": "sino", "S2B_loan_fixed": "loan", "S2C_native_default": "native"}
for fid, subset, variants in S2_FRAMES:
    for vtype in ["sino", "native", "loan"]:
        add("S2", subset=subset, group_id=fid, task_type="rating", variant=vtype,
            prompt=("다음 문장이 한국어 모어 화자에게 얼마나 자연스럽게 들립니까? "
                    "1(매우 부자연스럽다)부터 7(매우 자연스럽다)까지 점수로 평가하세요.\n문장: " + variants[vtype]),
            option_order=FIXED, output_schema="rating", scale="1-7", gold="", gold_type="survey", gold_status="pending_survey",
            hypothesis_id="H4",
            notes=f"사전 예상 최고 평정 변이형: {EXPECT[subset]} (가설일 뿐 정답 아님). 변이형 어종: {vtype}")

# ------------------------------------------------------------------ S3 지시 대상·어원
S3A = [
    ("chestnut", "잘 익은 밤(Chestnut)의 겉껍질", "the outer shell of a ripe chestnut", "밤색", "갈색", ["회색", "은색"], "연두색"),
    ("mugwort", "쑥(봄나물)의 잎", "the leaves of mugwort (ssuk), a spring herb", "쑥색", "초록색", ["분홍색", "보라색"], "은색"),
    ("apricot", "잘 익은 살구(열매)", "a ripe apricot", "살구색", "주황색", ["남색", "회색"], "은색"),
    ("tangerine", "귤의 껍질", "the peel of a tangerine", "귤색", "주황색", ["보라색", "회색"], "남색"),
    ("sky", "맑은 날 낮 하늘", "the sky on a clear day", "하늘색", "파란색", ["갈색", "분홍색"], "회색"),
    ("indigo", "쪽(염료 식물)으로 물들인 천", "cloth dyed with indigo plant (jjok)", "쪽빛", "파란색", ["노란색", "분홍색"], "갈색"),
    ("blood", "사람의 피(혈액)", "human blood", "핏빛", "빨간색", ["연두색", "은색"], "회색"),
    ("mouse", "쥐의 털", "a mouse's fur", "쥐색", "회색", ["분홍색", "주황색"], "연두색"),
]
s3a_ids = {}
s3a_en = {}
for gid, phrase, phrase_en, spec, hyper, ds, d3 in S3A:
    stem = f"{phrase} 색을 가장 정확하게 지칭하는 단어는 무엇입니까?"
    for variant, opts, conf in [("hyper", [spec, hyper] + ds, "medium"), ("nohyper", [spec] + ds + [d3], "high")]:
        iid = add("S3", subset="S3a_referent", group_id=gid, task_type="choice", prompt=stem, options=opts,
                  option_order=SHUF, output_schema="choice", gold=spec, gold_type="referent", gold_confidence=conf,
                  gold_basis=("대상 이름에서 유래한 하위 색이름이 상위어({})보다 정확하다고 본 연구자 정의. 설문 확인 필요".format(hyper)
                              if variant == "hyper" else "오답 선택지가 모두 명백히 다른 색이라 정답이 하나로 정해짐(통제 조건)"),
                  hypothesis_id="H5", variant=variant,
                  notes=("상위어 선택지 포함: 1차 Q4와 같은 구조" if variant == "hyper" else "상위어 제거: 지시 대상 지식만 측정"))
        s3a_ids[(gid, variant)] = iid
        s3a_en[iid] = phrase_en

S3B = [
    ("persimmon", "잘 익은 감(열매)의 껍질 색을 가장 정확하게 지칭하는 단어는 무엇입니까?", "the skin of a ripe persimmon",
     ["감색", "주황색", "회색", "은색"], ["주황색", "회색", "은색", "연두색"], "주황색", "감색", "high",
     "감색(紺色)은 짙은 남색으로 감 열매와 무관. 잘 익은 감 껍질은 주황색"),
    ("nightsky", "깊은 밤, 구름 한 점 없는 하늘(밤하늘)의 색을 가장 정확하게 지칭하는 단어는 무엇입니까?", "the sky on a deep, cloudless night",
     ["밤색", "남색", "주황색", "연두색"], ["남색", "주황색", "연두색", "분홍색"], "남색", "밤색", "medium",
     "밤색은 밤(栗)의 색이지 밤(夜)의 색이 아님(동형어 함정). 밤하늘은 짙은 남색 계열로 보고 정답 설정. 설문 확인 필요"),
]
s3b_ids = {}
for gid, stem, en, trap_opts, ctrl_opts, gold, trap, conf, basis in S3B:
    for variant, opts in [("trap", trap_opts), ("control", ctrl_opts)]:
        iid = add("S3", subset="S3b_lexical_trap", group_id=gid, task_type="choice", prompt=stem, options=opts,
                  option_order=SHUF, output_schema="choice", gold=gold, gold_type="referent", gold_confidence=conf,
                  gold_basis=basis, hypothesis_id="H6", variant=variant,
                  notes=(f"함정 선택지: {trap} (대상 이름과 글자가 겹치지만 오답)" if variant == "trap" else "통제: 함정 선택지 제거"))
        s3b_ids[(gid, variant)] = iid
        s3a_en[iid] = en

S3C = [
    ("밤색", "‘밤색’의 ‘밤’은 무엇을 가리킵니까?", ["밤나무의 열매(밤, 栗)", "밤(夜), 어두운 시간", "둘 다 가능하다", "알 수 없다"], 0,
     "밤색은 밤(栗) 껍질 색에서 유래(1차 논문 Q4 전제)", "high"),
    ("감색", "‘감색’의 ‘감’은 무엇에서 온 말입니까?", ["감나무의 열매(감)", "한자 紺(짙은 남색)", "‘감다’(눈을 감다)의 어간", "알 수 없다"], 1,
     "감색(紺色): 紺의 한국 한자음 ‘감’", "high"),
    ("갈색", "‘갈색’의 ‘갈’은 무엇에서 온 말입니까?", ["갈대(식물)", "한자 褐(거칠게 짠 털옷·베옷)", "‘갈다’(밭을 갈다)의 어간", "알 수 없다"], 1,
     "1차 논문 각주 4: 갈색(褐色)의 어원은 거친 모직물(털옷)을 뜻하는 한자이며 갈대 유래설은 오류", "high"),
    ("회색", "‘회색’의 ‘회’는 무엇에서 온 말입니까?", ["한자 灰(재)", "한자 膾(날고기, 회)", "한자 會(모이다)", "알 수 없다"], 0,
     "회색(灰色): 재의 빛깔", "high"),
    ("곤색", "‘곤색’이라는 표기는 무엇에서 온 말입니까?", ["한자 紺(짙은 남색)의 일본어 발음 ‘こん(곤)’", "곤충의 몸 색", "고유어 ‘곤’", "알 수 없다"], 0,
     "‘곤색’은 일본어 紺色(こんいろ)에서 온 표기이며 표준어는 감색", "high"),
    ("미색", "‘미색’(연한 노란빛이 도는 흰색)의 ‘미’는 어떤 한자입니까?", ["美(아름다울 미)", "米(쌀 미)", "味(맛 미)", "알 수 없다"], 1,
     "미색(米色): 쌀의 빛깔. 美色(아름다운 얼굴)과 동음이의어", "high"),
]
for gid, stem, opts, gi, basis, conf in S3C:
    add("S3", subset="S3c_etymology", group_id=gid, task_type="choice", prompt=stem, options=opts, option_order=SHUF_LAST,
        output_schema="choice", gold=opts[gi], gold_type="etymology", gold_confidence=conf, gold_basis=basis,
        hypothesis_id="H7", variant="etymology",
        notes="'알 수 없다'는 항상 마지막. 응답의 부가 어원 주장은 별도로 사전 대조(환각 코딩)")

# ------------------------------------------------------------------ S4 관용·다의
S4 = [
    # (expr, kind, group, sentence, opts_ko, opts_en, gold_idx, conf)
    ("하늘이 노랗다", "idiom", "yellow-sky", "충격적인 소식을 들은 순간 그는 하늘이 노랗게 보였다.",
     ["하늘이 실제로 노란색을 띠었다", "몹시 충격을 받아 정신이 아찔해졌다", "날씨가 맑아 기분이 좋아졌다", "판단할 수 없다"],
     ["The sky actually turned yellow", "He was so shocked that he felt dizzy and faint", "The weather was clear and he felt cheerful", "Cannot be determined"], 1, "high"),
    ("노랗다(literal)", "literal", "yellow-sky", "저녁이 되자 서쪽 하늘이 노랗게 물들었다.",
     ["노을 때문에 하늘이 실제로 노란빛을 띠었다", "몹시 충격을 받아 정신이 아찔해졌다", "날씨가 맑아 기분이 좋아졌다", "판단할 수 없다"],
     ["The sky really took on a yellow glow because of the sunset", "He was so shocked that he felt dizzy and faint", "The weather was clear and he felt cheerful", "Cannot be determined"], 0, "high"),
    ("파랗게 질리다", "idiom", "blue-face", "귀신 이야기를 듣던 아이의 얼굴이 파랗게 질렸다.",
     ["얼굴에 파란 물감이 묻었다", "겁에 질려 얼굴빛이 창백해졌다", "추워서 입술이 텄다", "판단할 수 없다"],
     ["Blue paint got on the child's face", "The child's face turned pale with fear", "The child's lips chapped from the cold", "Cannot be determined"], 1, "high"),
    ("파랗다(literal)", "literal", "blue-face", "아이는 도화지에 하늘을 파랗게 칠했다.",
     ["파란색 물감으로 하늘을 칠했다", "겁에 질려 얼굴빛이 창백해졌다", "추워서 입술이 텄다", "판단할 수 없다"],
     ["The child painted the sky with blue paint", "The child's face turned pale with fear", "The child's lips chapped from the cold", "Cannot be determined"], 0, "high"),
    ("속이 검다", "idiom", "black-inside", "그는 겉으로는 웃지만 속이 검어서 믿을 수가 없다.",
     ["실제로 내장이 검은색이다", "마음이 음흉하고 흉악하다", "몸이 아파 보인다", "판단할 수 없다"],
     ["His internal organs are actually black", "He is sly and ill-intentioned at heart", "He looks sick", "Cannot be determined"], 1, "high"),
    ("검다(literal)", "literal", "black-inside", "냄비를 오래 태웠더니 속이 새카맣게 그을렸다.",
     ["냄비 안쪽이 그을음으로 검게 변했다", "마음이 음흉하고 흉악하다", "몸이 아파 보인다", "판단할 수 없다"],
     ["The inside of the pot turned black with soot", "He is sly and ill-intentioned at heart", "He looks sick", "Cannot be determined"], 0, "high"),
    ("머릿속이 하얘지다", "idiom", "white-mind", "시험 문제를 보는 순간 머릿속이 하얘졌다.",
     ["머리카락이 하얗게 세었다", "아무 생각도 떠오르지 않았다", "눈이 부셔서 앞이 안 보였다", "판단할 수 없다"],
     ["His hair turned white", "His mind went completely blank", "He was dazzled and could not see", "Cannot be determined"], 1, "high"),
    ("하얗다(literal)", "literal", "white-mind", "밤새 눈이 내려 온 마을이 하얗게 덮였다.",
     ["눈이 쌓여 마을이 실제로 하얗게 덮였다", "아무 생각도 떠오르지 않았다", "눈이 부셔서 앞이 안 보였다", "판단할 수 없다"],
     ["Snow piled up and the village was literally covered in white", "His mind went completely blank", "He was dazzled and could not see", "Cannot be determined"], 0, "high"),
    ("하얗게 불태우다", "idiom", "white-burn", "마지막 경기에서 선수들은 가진 힘을 하얗게 불태웠다.",
     ["불이 나서 모든 것이 하얀 재가 되었다", "남은 힘 없이 전부 쏟아부었다", "경기가 취소되었다", "판단할 수 없다"],
     ["A fire turned everything into white ash", "They poured out all their strength without holding anything back", "The match was cancelled", "Cannot be determined"], 1, "medium"),
    ("백안시", "idiom", "white-eye", "그 사건 이후 동료들은 그를 백안시했다.",
     ["눈이 하얗게 보이도록 쳐다보았다", "업신여기며 냉대했다", "존경하며 따랐다", "판단할 수 없다"],
     ["They stared at him until their eyes looked white", "They looked down on him and treated him coldly", "They respected and followed him", "Cannot be determined"], 1, "high"),
    ("누렇게 뜨다", "idiom", "yellow-face", "며칠 밤을 새웠더니 얼굴이 누렇게 떴다.",
     ["얼굴에 노란 칠을 했다", "건강이 나빠져 병색이 돌게 되었다", "햇볕에 곱게 그을렸다", "판단할 수 없다"],
     ["He painted his face yellow", "His health deteriorated and he looked sickly", "He got a nice tan from the sun", "Cannot be determined"], 1, "high"),
    ("눈을 시퍼렇게 뜨다", "idiom", "blue-eyes", "내가 눈을 시퍼렇게 뜨고 있는 한 그런 일은 용납할 수 없다.",
     ["눈동자가 실제로 시퍼런 색이다", "내가 멀쩡히 살아 있는 동안에는", "눈이 아파서 잘 뜨지 못하는 동안에는", "판단할 수 없다"],
     ["My eyes are literally deep blue", "As long as I am alive and well", "While my eyes hurt and I can hardly open them", "Cannot be determined"], 1, "high"),
    ("푸른 꿈", "idiom", "blue-dream", "그는 푸른 꿈을 안고 서울로 올라왔다.",
     ["푸른색으로 된 꿈을 꾸었다", "희망에 찬 젊은 이상을 품었다", "고향을 그리워했다", "판단할 수 없다"],
     ["He had a dream that was blue in color", "He held youthful, hopeful ideals", "He missed his hometown", "Cannot be determined"], 1, "high"),
    ("눈앞이 캄캄하다", "idiom", "dark-eyes", "갑작스러운 해고 통보에 눈앞이 캄캄했다.",
     ["실제로 불이 꺼져 어두워졌다", "앞날이 막막하고 절망스러웠다", "졸음이 쏟아졌다", "판단할 수 없다"],
     ["The lights actually went out and it became dark", "The future felt bleak and hopeless", "He was overcome by drowsiness", "Cannot be determined"], 1, "high"),
    ("붉으락푸르락", "idiom", "red-blue", "그는 몹시 화가 나서 얼굴이 붉으락푸르락했다.",
     ["얼굴에 번갈아 물감을 칠했다", "몹시 화가 나서 얼굴빛이 변했다", "술에 취해 잠이 들었다", "판단할 수 없다"],
     ["He alternately painted his face different colors", "He was so angry that his face changed color", "He got drunk and fell asleep", "Cannot be determined"], 1, "high"),
]
s4_ids = []
s4_en = {}
for expr, kind, gid, sent, ko, en, gi, conf in S4:
    iid = add("S4", subset="S4_idiom" if kind == "idiom" else "S4_literal_control", group_id=gid, task_type="choice",
              prompt=f"다음 문장에서 색채 표현이 어떤 의미로 쓰였는지 가장 알맞은 것을 고르세요.\n문장: {sent}",
              options=ko, option_order=SHUF_LAST, output_schema="choice", gold=ko[gi], gold_type="idiom",
              gold_confidence=conf, gold_basis=f"표현 '{expr}'의 관용/문자 의미. 사전·모어 화자 대조 필요",
              hypothesis_id="H8", variant=kind,
              notes="literal 문항은 같은 색채어의 문자 그대로 용법(과잉 관용 해석 탐지용)" if kind == "literal" else "관용 의미 문항")
    s4_ids.append(iid)
    s4_en[iid] = (sent, en)

# ------------------------------------------------------------------ S5 안색·정서
S5_ADJ = ["칙칙하다", "창백하다", "해쓱하다", "누렇다", "푸르뎅뎅하다", "거무튀튀하다", "까무잡잡하다", "검다",
          "발그레하다", "불그스름하다", "뽀얗다", "화사하다"]
S5_DIMS = [("healthy", "이 표현은 건강해 보이는 인상을 준다."),
           ("favorable", "이 표현은 호감을 주는 긍정적인 인상이다."),
           ("gloomy", "이 표현은 우울하거나 기운 없어 보이는 인상을 준다.")]
for adj in S5_ADJ:
    for dkey, dsent in S5_DIMS:
        add("S5", subset="S5_complexion", group_id=adj, task_type="rating", variant=dkey,
            prompt=(f"사람의 안색을 묘사하는 형용사: '{adj}'\n이 표현을 들었을 때 받는 인상에 대해 다음 문장이 얼마나 맞는지 "
                    f"1(전혀 그렇지 않다)부터 7(매우 그렇다)까지 점수로 평가하세요.\n문장: {dsent}"),
            option_order=FIXED, output_schema="rating", scale="1-7", gold="", gold_type="survey", gold_status="pending_survey",
            hypothesis_id="H9", notes="모어 화자 평정 평균과 상관 분석(어휘 단위)")

# ------------------------------------------------------------------ S6 어종×레지스터
S6_COLORS = [("black", "검은색", [("검은색", "native"), ("블랙", "loan"), ("흑색", "sino")]),
             ("white", "흰색", [("흰색", "native"), ("화이트", "loan"), ("백색", "sino")])]
S6_CTX = [("gov", "정부 부처가 배포하는 공식 문서(규정·공고문)"), ("ad", "프리미엄 제품을 홍보하는 광고 문구"),
          ("daily", "친구와 나누는 일상 대화"), ("poem", "시(詩)의 한 구절")]
for ckey, cko, words in S6_COLORS:
    for ctx_key, ctx in S6_CTX:
        for w, vtype in words:
            add("S6", subset="S6_register_rating", group_id=f"{ckey}-{ctx_key}", task_type="rating", variant=vtype,
                prompt=(f"색 이름 표현: '{w}' (대상: {cko})\n상황: {ctx}\n이 상황에서 위 표현을 쓰는 것이 얼마나 적절한지 "
                        "1(전혀 적절하지 않다)부터 7(매우 적절하다)까지 점수로 평가하세요."),
                option_order=FIXED, output_schema="rating", scale="1-7", gold="", gold_type="survey", gold_status="pending_survey",
                hypothesis_id="H10", notes="어종(variant)×상황(group 후반부) 상호작용 분석. '검은색/흰색'은 고유어+한자어 혼성이므로 '고유어계'로 표기")

S6O = [("white", "마케팅 용어로서 '프리미엄 화이트'와 '고급스러운 흰색'은 소비자에게 각각 어떤 사회적, 감성적 이미지를 전달합니까?"),
       ("red", "마케팅 용어로서 '클래식 레드'와 '전통적인 붉은색'은 소비자에게 각각 어떤 사회적, 감성적 이미지를 전달합니까?"),
       ("blue", "마케팅 용어로서 '딥 블루'와 '깊은 푸른빛'은 소비자에게 각각 어떤 사회적, 감성적 이미지를 전달합니까?")]
for k, p in S6O:
    add("S6", subset="S6_open_q5ext", group_id=f"open-{k}", task_type="open", prompt=p, option_order=FIXED,
        output_schema="free", gold="", gold_type="none", gold_status="n/a", hypothesis_id="H10",
        notes="Q5 확장. 코딩 범주: 계층/세대/글로벌성/전통성/가격·접근성 (코드북 참조)")

# ------------------------------------------------------------------ S7 영어 지시문
def add_en(base_id, prompt, hyp="H11"):
    b = BY_ID[base_id]
    add("S7", subset="S7_en_" + b["set_id"] + "_" + b["subset"], group_id=b["group_id"], task_type=b["task_type"], lang="en",
        prompt=prompt, options=b["options"], option_order=b["option_order"], output_schema=b["output_schema"],
        gold=b["gold"], gold_type=b["gold_type"], gold_confidence=b["gold_confidence"], gold_basis=b["gold_basis"],
        hypothesis_id=hyp, variant=b["variant"], pair_id=base_id,
        notes="지시문만 영어, 자극(한국어 단어·문장)은 원문 유지. 한국어 원 문항과 쌍으로 분석")


dim_en = {k: en for k, _, en in S1A_DIMS}
for (small, large, dkey), iid in s1a_ids.items():
    add_en(iid, f"Which of the following two Korean color words gives a stronger impression of being {dim_en[dkey]}? ({small}, {large})")
for gkey, iid in s1c_ids.items():
    add_en(iid, f"Sort the following Korean color adjectives (all describing {s1c_color_en[iid]}) from the one whose color feels deepest and most intense to the one that feels weakest.")
for iid, en in s3a_en.items():
    add_en(iid, f"Which Korean word most accurately refers to the color of {en}?")
for iid, (sent, en_opts) in s4_en.items():
    b = BY_ID[iid]
    row_id = add("S7", subset="S7_en_S4_" + b["subset"], group_id=b["group_id"], task_type="choice", lang="en",
                 prompt=("Read the following Korean sentence and choose the option that best describes the meaning of the color "
                         f"expression in it.\nSentence: {sent}"),
                 options=en_opts, option_order=b["option_order"], output_schema="choice", gold=en_opts[b["options"].index(b["gold"])],
                 gold_type=b["gold_type"], gold_confidence=b["gold_confidence"], gold_basis=b["gold_basis"], hypothesis_id="H11",
                 variant=b["variant"], pair_id=iid,
                 notes="영어 선택지는 한국어 선택지의 번역(순서 동일). 채점은 gold 문구 일치 또는 선택 인덱스 일치")

# ------------------------------------------------------------------ S8 RAG 충돌
S8_DOCS = {
    "chestnut": ("밤색은 잘 익은 밤의 겉껍질 빛깔에서 유래한 색 이름으로, 붉은 기가 도는 짙은 갈색을 가리킨다.",
                 ["밤의 겉껍질은 단단한 갈색이며, 잘 익을수록 갈색이 짙어지고 윤기가 난다.",
                  "수확한 밤은 갈색 껍질의 색과 크기를 기준으로 선별한다. 껍질이 연한 갈색이면 덜 익은 것이다.",
                  "밤 껍질을 벗기면 갈색 겉껍질 안쪽에 얇은 속껍질이 있는데, 속껍질도 옅은 갈색을 띤다."]),
    "mugwort": ("쑥색은 쑥잎의 빛깔처럼 누런 기가 도는 어두운 초록색을 가리키는 색 이름이다.",
                ["쑥은 봄에 돋아나는 초록색 잎을 먹는 대표적인 나물로, 어린잎일수록 연한 초록색이다.",
                 "쑥을 데치면 잎의 초록색이 선명해지고, 쑥국이나 쑥떡에는 부드러운 초록색 어린잎을 쓴다.",
                 "쑥잎 뒷면은 흰빛이 돌지만 앞면은 짙은 초록색이어서 다른 풀과 구별하기 쉽다."]),
    "apricot": ("살구색은 잘 익은 살구 열매의 빛깔 같은 연한 주황색을 뜻하는 색 이름이다.",
                ["살구는 초여름에 익는 주황색 열매로, 껍질과 과육이 모두 주황색을 띤다.",
                 "잘 익은 살구는 주황색이 짙고 향이 진하며, 덜 익은 살구는 연한 노란빛이 돈다.",
                 "살구 과육은 부드러운 주황색이어서 잼이나 청으로 만들어도 주황색이 곱게 난다."]),
    "tangerine": ("귤색은 잘 익은 귤 껍질처럼 밝은 주황빛을 띤 색을 말한다.",
                  ["귤은 껍질이 선명한 주황색일수록 잘 익은 것으로 본다.",
                   "겨울철 시장에 나오는 귤은 대체로 껍질이 얇고 주황색이 고르게 든다.",
                   "귤 껍질은 말려서 진피라는 약재로 쓰이며, 말리기 전에는 주황색을 띤다."]),
    "sky": ("하늘색은 맑게 갠 낮 하늘처럼 연한 푸른빛을 가리키는 색 이름이다.",
            ["맑은 날 낮 하늘이 파란색으로 보이는 것은 대기가 파란색 빛을 더 잘 산란시키기 때문이다.",
             "구름 한 점 없는 낮에는 하늘이 짙은 파란색을 띠고, 지평선 근처에서는 파란색이 옅어진다.",
             "아이들은 하늘을 그릴 때 흔히 파란색 크레파스를 고른다."]),
    "indigo": ("쪽빛은 쪽 풀로 물들인 천에서 나는 짙은 푸른빛을 가리키는 색 이름이다.",
               ["쪽 풀로 염색한 천은 파란색이 우러나며, 염색을 거듭할수록 파란색이 짙어진다.",
                "전통 염색에서 쪽은 대표적인 파란색 염료로 쓰였다.",
                "쪽 염색 천은 공기와 햇볕에 닿으면 파란색이 더욱 선명하게 발색한다."]),
    "blood": ("핏빛은 사람의 피처럼 붉고 짙은 빛깔을 뜻하는 색 이름이다.",
              ["사람의 피는 빨간색이며, 산소와 결합한 혈액일수록 선명한 빨간색을 띤다.",
               "상처에서 흐르는 피는 처음에는 선명한 빨간색이지만 굳으면 어두운 빨간색으로 변한다.",
               "혈액이 빨간색인 것은 적혈구 속 헤모글로빈 때문이다."]),
    "mouse": ("쥐색은 쥐털의 빛깔 같은 짙은 회색을 가리키는 색 이름이다.",
              ["쥐의 털은 대체로 회색을 띠는 경우가 많다.",
               "회색 털을 가진 쥐는 어두운 곳에서 눈에 잘 띄지 않아 천적을 피하기 쉽다.",
               "실험용 쥐 가운데 일부 품종은 회색 털을 가지고 있다."]),
}
IRRELEVANT = [
    "자전거 타이어 공기압은 계절과 노면 상태에 따라 조금씩 조절하는 것이 좋다. 공기압이 낮으면 펑크가 나기 쉽다.",
    "이 지역은 연평균 강수량이 많지 않아 봄가을에 건조주의보가 자주 발령된다.",
    "새로 나온 노트북은 배터리 지속 시간이 길고 무게가 가벼워 휴대성이 좋다는 평가를 받았다.",
    "도서관은 월요일을 제외하고 오전 9시부터 오후 6시까지 운영하며 대출 기간은 14일이다.",
]
for oi, (gid, phrase, _en, spec, hyper, ds, d3) in enumerate(S3A):
    base_id = s3a_ids[(gid, "hyper")]
    base = BY_ID[base_id]
    stem = base["prompt"]
    sup, biased = S8_DOCS[gid]
    conds = [
        ("support", [sup], "정답 지지: 하위어의 정의를 직접 제시"),
        ("biased_1", biased[:1], "상위어 편향 1건: 상위어만 반복 사용(사실과 모순되지 않음)"),
        ("biased_3", biased[:3], "상위어 편향 3건: 같은 상위어를 여러 문서에서 반복(빈도 효과)"),
        ("irrelevant", [IRRELEVANT[oi % 4]], "무관 문서: 길이 효과 통제"),
    ]
    for variant, docs, note in conds:
        doc_text = "\n".join(f"[문서 {i + 1}] {d}" for i, d in enumerate(docs))
        add("S8", subset="S8_rag_conflict", group_id=gid, task_type="choice",
            prompt=f"다음은 검색된 문서입니다.\n{doc_text}\n\n{stem}", options=base["options"], option_order=SHUF,
            output_schema="choice", gold=spec, gold_type="referent", gold_confidence="medium",
            gold_basis=base["gold_basis"], hypothesis_id="H12", variant=variant, base_item_id=base_id, context_doc=doc_text,
            notes=note + " / 문서는 연구용 합성 텍스트(실제 출처 없음). 기준선(문서 없음)은 base_item_id 문항")

# ------------------------------------------------------------------ S9 설명 충실성
S9_BASES = []
for small, large in S1A_PAIRS:
    if small != "가맣다":
        S9_BASES.append(s1a_ids[(small, large, "dark_turbid")])
for plain, fortis in S1B_PAIRS:
    if plain != "가맣다":
        S9_BASES.append(s1b_ids[(plain, fortis)])
for bid in S9_BASES:
    b = BY_ID[bid]
    gold = b["gold"]
    wrong = [o for o in b["options"] if o != gold][0]
    stem = b["prompt"]
    conds = [
        ("answer_first", f"{stem}\n먼저 '답: ○○' 형태로 한 줄로 답한 뒤, 그 이유를 설명하세요.", gold, ""),
        ("explain_first", f"{stem}\n먼저 이유를 단계적으로 설명한 뒤, 마지막 줄에 '답: ○○' 형태로 답하세요.", gold, ""),
        ("justify_gold", f"{stem}\n이 문제의 정답은 '{gold}'입니다. 그 이유를 언어학적 근거(음상징, 형태소, 음운 등)를 들어 설명하세요.", "", gold),
        ("justify_wrong", f"{stem}\n이 문제의 정답은 '{wrong}'입니다. 그 이유를 언어학적 근거(음상징, 형태소, 음운 등)를 들어 설명하세요.", "", wrong),
    ]
    for variant, prompt, g, premise in conds:
        add("S9", subset="S9_faithfulness", group_id=b["group_id"], task_type="open", prompt=prompt, prompt_includes_options=True,
            options=b["options"], option_order=FIXED, output_schema="free", gold=g,
            gold_type="linguistic" if g else "none", gold_confidence=b["gold_confidence"] if g else "",
            gold_basis=b["gold_basis"] if g else "", premise_option=premise, hypothesis_id="H13", variant=variant,
            base_item_id=bid,
            notes=("거짓 전제 조건: 모델이 전제를 반박하는지/그럴듯하게 합리화하는지 코딩" if variant == "justify_wrong"
                   else "설명 중 음상징·어원 주장은 추출해 사전·문헌과 대조(근거 환각률)"))

# ================================================================== v0.2 보강 문항 (기존 ID는 그대로, 새 문항은 각 세트 번호 뒤에 이어 붙임)
_V1_N = len(ITEMS)
_NEW_S7 = 0

# ------------------------------------------------------------------ S1e 가상어(비단어) 음상징: 암기·빈도 효과 제거
S1E_NONCE = [("소랗다", "수렇다"), ("조랗다", "주렇다"), ("모랗다", "무렇다"), ("호랗다", "후렇다"),
             ("타랗다", "터렇다"), ("라랗다", "러렇다"), ("차랗다", "처렇다"), ("가랗다", "거렇다")]
s1e_ids = {}
for small, large in S1E_NONCE:
    iid = add("S1", subset="S1e_nonce_vowel", group_id=f"nonce:{small}-{large}", task_type="choice",
              prompt=(f"다음은 실제로 쓰이지 않는 가상의 한국어 색채 형용사입니다. 두 단어 중, 더 어둡고 탁한 색을 가리킬 것 같은 "
                      f"단어는 무엇입니까? ({small}, {large} 중 택 1)"),
              options=[small, large], option_order=SHUF, output_schema="choice", gold=large,
              gold_type="linguistic", gold_confidence="medium",
              gold_basis="음상징 일반화 예측: 음성모음(큰말)이 더 어둡고 탁한 심상. 실제 단어가 아니므로 암기로 풀 수 없음",
              hypothesis_id="H1", variant="nonce",
              notes="가상어: 모델이 규칙을 비단어에도 일반화하는지 측정. 모어 화자 설문으로 gold 확인 필요(정답이라기보다 예측)")
    s1e_ids[(small, large)] = iid

# ------------------------------------------------------------------ S1f 단서 충돌: 모음 단서 vs 된소리 단서
S1F = [("빨갛다", "벌겋다", "strong_vivid", "강하고 선명한", "빨갛다", "벌겋다"),
       ("까맣다", "거멓다", "strong_vivid", "강하고 선명한", "까맣다", "거멓다"),
       ("빨갛다", "벌겋다", "dark_turbid", "어둡고 탁한", "빨갛다", "벌겋다"),
       ("까맣다", "거멓다", "dark_turbid", "어둡고 탁한", "까맣다", "거멓다")]
for a, b, dkey, dko, fortis_cue, vowel_cue in S1F:
    add("S1", subset="S1f_cue_conflict", group_id=f"conflict:{a}-{b}", task_type="choice",
        prompt=f"다음 두 단어 중, {dko} 느낌이 더 강한 단어는 무엇입니까? ({a}, {b} 중 택 1)",
        options=[a, b], option_order=SHUF, output_schema="choice", gold="", gold_type="survey",
        gold_status="pending_survey", gold_confidence="",
        gold_basis="양성모음+된소리 vs 음성모음+예사소리: 두 단서가 충돌. 정답 없이 어느 단서를 따르는지 분포를 본다",
        hypothesis_id="H1,H2", variant=dkey,
        notes=f"fortis_cue={fortis_cue};vowel_cue={vowel_cue}")

# ------------------------------------------------------------------ S1g 5단어 순위(변별력↑)
S1G = [("red_5", "붉은색", "red", ["새빨갛다", "빨갛다", "붉다", "불그스름하다", "불그레하다"]),
       ("black_5", "검은색", "black", ["새까맣다", "까맣다", "검다", "거무스름하다", "거뭇하다"])]
s1g_ids = {}
for gkey, cko, cen, words in S1G:
    iid = add("S1", subset="S1g_rank5", group_id=f"rank:{gkey}", task_type="rank",
              prompt=f"다음 한국어 색채 형용사들은 모두 {cko} 계열입니다. 해당 색이 더 짙고 강하게 느껴지는 순서(강한 것 → 약한 것)로 정렬하세요.",
              options=words, option_order=SHUF, output_schema="rank", gold=">".join(words),
              gold_type="linguistic", gold_confidence="medium",
              gold_basis="강조 접두사형 > 기본형 > 약화형. 마지막 두 단어의 순서는 불확실하므로 Kendall τ로 평가하고 설문으로 확정",
              hypothesis_id="H3", variant="affix5", notes="5개 항목: 전체 일치율보다 Kendall τ 사용 권장")
    s1g_ids[gkey] = iid
    s1c_color_en[iid] = cen

# ------------------------------------------------------------------ S3a 추가 지시 대상 8종 (hyper / nohyper)
S3A_NEW = [
    ("forsythia", "봄에 피는 개나리꽃", "a forsythia flower blooming in spring", "개나리색", "노란색", ["남색", "회색"], "분홍색"),
    ("azalea", "봄에 피는 진달래꽃", "an azalea flower blooming in spring", "진달래색", "분홍색", ["갈색", "남색"], "회색"),
    ("sprout", "봄에 갓 돋아난 어린 새싹", "a freshly sprouted young seedling in spring", "연두색", "초록색", ["보라색", "분홍색"], "회색"),
    ("mustard", "겨자씨를 간 겨자 가루", "ground mustard seed powder", "겨자색", "노란색", ["남색", "보라색"], "회색"),
    ("balloonflower", "도라지꽃", "a balloon flower (doraji)", "도라지색", "보라색", ["주황색", "갈색"], "연두색"),
    ("olive", "올리브나무의 잎", "the leaf of an olive tree", "올리브색", "초록색", ["분홍색", "남색"], "주황색"),
    ("eggplant", "가지(채소)의 껍질", "the skin of an eggplant", "가지색", "보라색", ["주황색", "회색"], "연두색"),
    ("chocolate", "녹여서 굳힌 다크 초콜릿", "dark chocolate", "초콜릿색", "갈색", ["하늘색", "분홍색"], "연두색"),
]
s3a_new_ids = {}
for gid, phrase, phrase_en, spec, hyper, ds, d3 in S3A_NEW:
    stem = f"{phrase} 색을 가장 정확하게 지칭하는 단어는 무엇입니까?"
    for variant, opts, conf in [("hyper", [spec, hyper] + ds, "medium"), ("nohyper", [spec] + ds + [d3], "high")]:
        iid = add("S3", subset="S3a_referent", group_id=gid, task_type="choice", prompt=stem, options=opts,
                  option_order=SHUF, output_schema="choice", gold=spec, gold_type="referent", gold_confidence=conf,
                  gold_basis=("대상 이름에서 유래한 하위 색이름이 상위어({})보다 정확하다고 본 연구자 정의. 사전 및 설문 확인 필요".format(hyper)
                              if variant == "hyper" else "오답 선택지가 모두 명백히 다른 색이라 정답이 하나로 정해짐(통제 조건)"),
                  hypothesis_id="H5", variant=variant,
                  notes=("v0.2 추가. 상위어 선택지 포함" if variant == "hyper" else "v0.2 추가. 상위어 제거"))
        s3a_new_ids[(gid, variant)] = iid
        s3a_en[iid] = phrase_en

# ------------------------------------------------------------------ S3b 추가 함정 4종 (trap / control)
S3B_NEW = [
    ("cloudysky", "비가 내리기 직전, 구름이 잔뜩 낀 낮 하늘의 색을 가장 정확하게 지칭하는 단어는 무엇입니까?",
     "the daytime sky just before rain, thick with clouds",
     ["하늘색", "회색", "주황색", "연두색"], ["회색", "주황색", "연두색", "분홍색"], "회색", "하늘색", "high",
     "'하늘'이 들어간 하늘색은 맑은 하늘의 색 이름. 잔뜩 흐린 하늘은 회색 계열"),
    ("sunsetsky", "해가 지는 저녁, 노을이 물든 서쪽 하늘의 색을 가장 정확하게 지칭하는 단어는 무엇입니까?",
     "the western sky at sunset, tinged by the afterglow",
     ["하늘색", "주황색", "연두색", "은색"], ["주황색", "연두색", "은색", "갈색"], "주황색", "하늘색", "medium",
     "노을은 주황·붉은 계열. 선택지에 붉은색이 없어 주황색을 정답으로 설정. 설문 확인 필요"),
    ("ginkgo", "늦가을 단풍이 든 은행나무 잎의 색을 가장 정확하게 지칭하는 단어는 무엇입니까?",
     "the leaves of a ginkgo tree in late-autumn foliage",
     ["은색", "노란색", "회색", "남색"], ["노란색", "회색", "남색", "분홍색"], "노란색", "은색", "high",
     "'은행'과 '은색'의 글자 겹침이 함정. 단풍 든 은행잎은 노란색"),
    ("cheongpomuk", "녹두 전분으로 쑤어 만든 청포묵의 색을 가장 정확하게 지칭하는 단어는 무엇입니까?",
     "cheongpomuk, a jelly made from mung bean starch",
     ["초록색", "흰색", "분홍색", "갈색"], ["흰색", "분홍색", "갈색", "보라색"], "흰색", "초록색", "medium",
     "녹두(초록)·'청'이라는 이름이 함정. 청포묵은 녹두 전분으로 만든 반투명한 흰색. 설문·조리 문헌 확인 필요"),
]
s3b_new_ids = {}
for gid, stem, en, trap_opts, ctrl_opts, gold, trap, conf, basis in S3B_NEW:
    for variant, opts in [("trap", trap_opts), ("control", ctrl_opts)]:
        iid = add("S3", subset="S3b_lexical_trap", group_id=gid, task_type="choice", prompt=stem, options=opts,
                  option_order=SHUF, output_schema="choice", gold=gold, gold_type="referent", gold_confidence=conf,
                  gold_basis=basis, hypothesis_id="H6", variant=variant,
                  notes=(f"v0.2 추가. 함정 선택지: {trap}" if variant == "trap" else "v0.2 추가. 통제: 함정 선택지 제거"))
        s3b_new_ids[(gid, variant)] = iid
        s3a_en[iid] = en

# ------------------------------------------------------------------ S4b 어려운 관용·다의 (거짓 친구, 외래어 관용, 반의 혼동)
# (expr, kind, group, sentence, ko, en, gold_idx, conf)
S4B = [
    ("파란만장", "false_friend", "wave-blue", "할머니는 파란만장한 삶을 살아오신 분이다.",
     ["푸른빛이 가득한 평화로운 삶이었다", "기복과 우여곡절이 많은 삶이었다", "젊은 시절을 푸른 산에서 보냈다", "판단할 수 없다"],
     ["It was a peaceful life full of blue", "It was a life full of ups, downs and twists", "She spent her youth in the green mountains", "Cannot be determined"], 1, "high"),
    ("홍일점", "idiom", "red-dot", "그 연구팀에서 그녀는 홍일점이었다.",
     ["팀에서 유일한 여성이었다", "얼굴에 붉은 점이 하나 있었다", "팀에서 가장 나이가 어렸다", "판단할 수 없다"],
     ["She was the only woman on the team", "She had a single red mole on her face", "She was the youngest on the team", "Cannot be determined"], 0, "high"),
    ("흑막", "idiom", "black-curtain", "이번 사건의 배후에 거대한 흑막이 있다는 소문이 돌았다.",
     ["사건 현장에 검은 막이 쳐져 있었다", "겉으로 드러나지 않은 음흉한 내막이 있다", "사건이 밤에 일어났다", "판단할 수 없다"],
     ["A black curtain had been hung at the scene", "There is a sinister hidden story behind it", "The incident happened at night", "Cannot be determined"], 1, "high"),
    ("흑막(literal)", "literal", "black-curtain", "무대 앞에 흑막이 내려와 객석에서 무대 위가 보이지 않았다.",
     ["검은 막이 내려와 무대가 가려졌다", "겉으로 드러나지 않은 음흉한 내막이 있다", "공연이 취소되었다", "판단할 수 없다"],
     ["A black curtain came down and hid the stage", "There is a sinister hidden story behind it", "The performance was cancelled", "Cannot be determined"], 0, "high"),
    ("청출어람", "idiom", "blue-from-indigo", "사람들은 그 젊은 연구자를 두고 청출어람이라고 평했다.",
     ["제자가 스승보다 더 뛰어나게 되었다", "쪽풀로 푸른 물감을 만드는 데 성공했다", "스승의 가르침을 그대로 따랐다", "판단할 수 없다"],
     ["The student has come to surpass the teacher", "He succeeded in making blue dye from indigo plants", "He followed his teacher's instructions exactly", "Cannot be determined"], 0, "high"),
    ("청사진", "idiom", "blueprint", "회사는 향후 10년의 성장 청사진을 발표했다.",
     ["푸른 종이에 인쇄한 설계도를 배포했다", "앞으로의 구체적인 계획과 구상을 밝혔다", "성장세가 둔화될 것이라고 전망했다", "판단할 수 없다"],
     ["It distributed a design drawing printed on blue paper", "It announced a concrete plan and vision for the future", "It forecast that growth would slow", "Cannot be determined"], 1, "medium"),
    ("적자", "idiom", "red-ink", "그 회사는 3년 연속 적자를 기록했다.",
     ["수입이 지출보다 많아 이익이 났다", "지출이 수입보다 많아 손실이 났다", "붉은 글씨로 장부를 정리했다", "판단할 수 없다"],
     ["Income exceeded expenses and it made a profit", "Expenses exceeded income and it made a loss", "It kept its ledger in red letters", "Cannot be determined"], 1, "high"),
    ("흑자", "idiom", "black-ink", "그 가게는 개업 2년 만에 처음으로 흑자로 돌아섰다.",
     ["수입이 지출보다 많아 이익이 났다", "지출이 수입보다 많아 손실이 났다", "검은 잉크로 장부를 정리했다", "판단할 수 없다"],
     ["Income exceeded expenses and it made a profit", "Expenses exceeded income and it made a loss", "It kept its ledger in black ink", "Cannot be determined"], 0, "high"),
    ("기분이 블루하다", "loan_idiom", "feel-blue", "비 오는 날이면 괜히 기분이 블루해진다.",
     ["우울하고 가라앉은 기분이 든다", "차가운 느낌의 파란 옷을 입고 싶어진다", "하늘이 파랗게 보인다", "판단할 수 없다"],
     ["He feels gloomy and down", "He wants to wear cool-looking blue clothes", "The sky looks blue to him", "Cannot be determined"], 0, "high"),
    ("블루(literal)", "literal", "feel-blue", "그녀는 블루 셔츠에 흰 바지를 입고 나왔다.",
     ["파란색 셔츠를 입고 나왔다", "우울한 기분으로 나왔다", "차가운 성격의 사람이다", "판단할 수 없다"],
     ["She came out wearing a blue shirt", "She came out feeling gloomy", "She is a cold-natured person", "Cannot be determined"], 0, "high"),
    ("그린라이트", "loan_idiom", "green-light", "그가 먼저 연락을 해 오자 친구들은 그린라이트가 켜진 것 같다고 했다.",
     ["상대가 호감이 있다는 신호", "교통 신호가 초록불로 바뀌었다", "그가 곧 연락을 끊을 것이라는 예고", "판단할 수 없다"],
     ["A sign that the other person is interested in him", "A traffic light turned green", "A warning that he would soon cut off contact", "Cannot be determined"], 0, "high"),
    ("블랙 컨슈머", "coinage", "black-consumer", "매장 직원들은 그 손님을 블랙 컨슈머로 분류해 두었다.",
     ["악의적으로 과도한 환불이나 보상을 요구하는 소비자", "검은 옷만 입는 손님", "물건을 많이 사 주는 우수 고객", "판단할 수 없다"],
     ["A consumer who maliciously demands excessive refunds or compensation", "A customer who only wears black", "An excellent customer who buys a lot", "Cannot be determined"], 0, "high"),
    ("붉은 악마", "idiom", "red-devils", "경기장은 붉은 악마의 응원 소리로 가득 찼다.",
     ["한국 축구 국가대표팀의 응원단", "붉은 옷을 입고 악마로 분장한 공연단", "상대 팀의 마스코트", "판단할 수 없다"],
     ["The supporters' group of the Korean national football team", "A troupe of performers dressed as devils in red", "The opposing team's mascot", "Cannot be determined"], 0, "high"),
    ("레드오션", "idiom", "red-ocean", "스마트폰 케이스 시장은 이미 레드오션이다.",
     ["경쟁이 치열해 이익을 내기 어려운 시장", "아직 경쟁자가 없는 새로운 시장", "붉은 바다에서 만드는 제품의 시장", "판단할 수 없다"],
     ["A market so fiercely competitive that profit is hard to make", "A new market with no competitors yet", "A market for products made in a red sea", "Cannot be determined"], 0, "high"),
    ("블루오션", "idiom", "blue-ocean", "그 스타트업은 아직 경쟁자가 없는 블루오션을 찾았다고 밝혔다.",
     ["경쟁이 거의 없는 새로운 시장", "푸른 바다에서 하는 해양 사업", "경쟁이 치열해 이익이 줄어드는 시장", "판단할 수 없다"],
     ["A new market with almost no competition", "A marine business conducted on the blue sea", "A market where fierce competition shrinks profits", "Cannot be determined"], 0, "high"),
    ("흑백논리", "idiom", "black-white-logic", "그는 모든 문제를 흑백논리로 판단해서 대화가 어렵다.",
     ["모든 것을 두 극단 중 하나로만 가르는 사고방식", "흑색과 백색 안료를 배합하는 원리", "타협을 중시하는 유연한 사고방식", "판단할 수 없다"],
     ["A way of thinking that splits everything into one of two extremes", "The principle of mixing black and white pigments", "A flexible way of thinking that values compromise", "Cannot be determined"], 0, "high"),
    ("회색 지대", "idiom", "gray-zone", "그 문제는 법적으로 회색 지대에 놓여 있다.",
     ["합법과 불법이 분명하지 않은 영역", "회색 건물이 모여 있는 구역", "불법임이 확실한 영역", "판단할 수 없다"],
     ["An area where legal and illegal are not clearly distinguished", "A district where gray buildings cluster", "An area that is clearly illegal", "Cannot be determined"], 0, "high"),
    ("회색(literal)", "literal", "gray-zone", "공장 지대에는 회색 콘크리트 건물만 줄지어 서 있었다.",
     ["회색 콘크리트 건물이 늘어서 있었다", "법적으로 애매한 상태였다", "도시가 활기로 가득했다", "판단할 수 없다"],
     ["Gray concrete buildings stood in rows", "It was in a legally ambiguous state", "The city was full of life", "Cannot be determined"], 0, "high"),
    ("핑크빛 기류", "idiom", "pink-air", "두 사람 사이에 핑크빛 기류가 흐른다는 소문이 돌았다.",
     ["두 사람이 서로 호감을 갖고 있는 듯한 분위기", "두 사람 사이에 분홍색 안개가 끼었다", "두 사람이 심하게 다투었다", "판단할 수 없다"],
     ["An atmosphere suggesting the two like each other", "A pink mist hung between the two", "The two had a fierce quarrel", "Cannot be determined"], 0, "high"),
    ("핑크빛(literal)", "literal", "pink-air", "아침 하늘에 핑크빛 노을이 번졌다.",
     ["노을이 분홍빛으로 물들었다", "연인 사이에 호감이 생겼다", "날씨가 곧 나빠질 것이다", "판단할 수 없다"],
     ["The sunrise glow spread in pink", "Affection developed between a couple", "The weather will soon turn bad", "Cannot be determined"], 0, "high"),
]
s4b_ids = []
for expr, kind, gid, sent, ko, en, gi, conf in S4B:
    iid = add("S4", subset="S4b_literal_control" if kind == "literal" else "S4b_hard_idiom", group_id=gid, task_type="choice",
              prompt=f"다음 문장에서 색채 표현이 어떤 의미로 쓰였는지 가장 알맞은 것을 고르세요.\n문장: {sent}",
              options=ko, option_order=SHUF_LAST, output_schema="choice", gold=ko[gi], gold_type="idiom",
              gold_confidence=conf, gold_basis=f"표현 '{expr}'의 관용/문자 의미. 사전·모어 화자 대조 필요",
              hypothesis_id="H8", variant=kind,
              notes=("v0.2 추가. literal 통제: 같은 색채어의 문자 그대로 용법" if kind == "literal"
                     else "v0.2 추가. 유형=" + kind + ("(색 글자처럼 보이지만 한자어 '波瀾'; 프롬프트의 '색채 표현' 전제가 거짓)" if kind == "false_friend" else "")))
    s4b_ids.append(iid)
    s4_en[iid] = (sent, en)

# ------------------------------------------------------------------ S8 추가 (신규 S3a 8종 x 4 조건)
S8_DOCS_NEW = {
    "forsythia": ("개나리색은 봄에 피는 개나리꽃의 빛깔처럼 선명하고 밝은 노란색을 가리키는 색 이름이다.",
                  ["개나리꽃은 이른 봄에 피는 노란색 꽃으로, 봄을 알리는 대표적인 꽃이다.",
                   "개나리는 가지마다 노란색 꽃이 가득 피어 먼 곳에서도 노란색이 눈에 띈다.",
                   "개나리꽃의 노란색 꽃잎은 네 갈래로 갈라져 있고, 꽃이 진 뒤에야 잎이 돋는다."]),
    "azalea": ("진달래색은 진달래꽃의 빛깔처럼 연하고 붉은 기가 도는 분홍색을 가리키는 색 이름이다.",
               ["진달래꽃은 이른 봄 산에 피는 분홍색 꽃으로, 화전을 부칠 때 쓰는 꽃이다.",
                "봄 산비탈이 분홍색 진달래꽃으로 물들면 사람들이 꽃구경을 나선다.",
                "진달래의 분홍색 꽃잎은 먹을 수 있어서 예로부터 화전과 두견주의 재료로 쓰였다."]),
    "sprout": ("연두색은 갓 돋아난 새싹이나 어린잎처럼 노란 기가 도는 연한 초록색을 가리키는 색 이름이다.",
               ["봄이 되면 갓 돋아난 새싹은 초록색을 띠며, 자라면서 초록색이 점점 짙어진다.",
                "새싹은 광합성을 시작하면서 잎이 초록색으로 변하고 줄기도 초록색을 띤다.",
                "봄비를 맞은 어린 새싹들이 초록색으로 온 들판을 덮기 시작했다."]),
    "mustard": ("겨자색은 겨자씨를 간 가루처럼 탁한 노란색에 갈색 기가 도는 색을 가리키는 색 이름이다.",
                ["겨자는 씨앗을 갈아 만든 노란색 양념으로, 고기나 냉채에 곁들여 먹는다.",
                 "겨자 가루에 물을 개면 노란색 겨자장이 되며 매운 향이 난다.",
                 "겨자 꽃은 노란색 십자 모양으로 피고 씨앗도 노란빛을 띤다."]),
    "balloonflower": ("도라지색은 도라지꽃의 빛깔처럼 푸른 기가 도는 연한 보라색을 가리키는 색 이름이다.",
                      ["도라지꽃은 여름에 피는 보라색 꽃으로, 종 모양으로 피어 있는 모습이 눈에 띈다.",
                       "도라지는 보라색 꽃과 흰색 꽃이 피는 품종이 있으며 보라색 꽃이 더 널리 알려져 있다.",
                       "산과 들에 핀 도라지의 보라색 꽃은 한여름 풍경을 대표한다."]),
    "olive": ("올리브색은 올리브나무 잎이나 열매처럼 누런 기가 도는 어두운 초록색을 가리키는 색 이름이다.",
              ["올리브나무의 잎은 앞면이 초록색이고 뒷면은 은빛이 돌아 바람에 흔들리면 반짝인다.",
               "올리브 열매는 덜 익었을 때 초록색을 띠고 익으면서 색이 변한다.",
               "올리브나무는 사계절 초록색 잎을 달고 있는 상록수다."]),
    "eggplant": ("가지색은 가지 열매의 껍질처럼 윤기 나는 짙은 보라색을 가리키는 색 이름이다.",
                 ["가지는 껍질이 보라색이고 속은 흰색인 여름 채소로, 볶음이나 무침으로 먹는다.",
                  "가지 꽃도 연한 보라색으로 피며, 열매가 자라면서 껍질의 보라색이 짙어진다.",
                  "가지를 오래 두면 껍질의 보라색이 윤기를 잃고 주름이 진다."]),
    "chocolate": ("초콜릿색은 다크 초콜릿처럼 어둡고 깊은 갈색을 가리키는 색 이름이다.",
                  ["초콜릿은 카카오 열매를 가공해 만든 갈색 과자로, 카카오 함량이 높을수록 갈색이 짙어진다.",
                   "녹인 초콜릿은 부드러운 갈색을 띠며 굳히면 윤기 나는 갈색 덩어리가 된다.",
                   "초콜릿 케이크는 반죽에 코코아를 넣어 짙은 갈색을 낸다."]),
}
for oi, (gid, phrase, _en, spec, hyper, ds, d3) in enumerate(S3A_NEW):
    base_id = s3a_new_ids[(gid, "hyper")]
    base = BY_ID[base_id]
    stem = base["prompt"]
    sup, biased = S8_DOCS_NEW[gid]
    conds = [
        ("support", [sup], "정답 지지: 하위어의 정의를 직접 제시"),
        ("biased_1", biased[:1], "상위어 편향 1건: 상위어만 반복 사용(사실과 모순되지 않음)"),
        ("biased_3", biased[:3], "상위어 편향 3건: 같은 상위어를 여러 문서에서 반복(빈도 효과)"),
        ("irrelevant", [IRRELEVANT[(oi + 2) % 4]], "무관 문서: 길이 효과 통제"),
    ]
    for variant, docs, note in conds:
        doc_text = "\n".join(f"[문서 {i + 1}] {d}" for i, d in enumerate(docs))
        add("S8", subset="S8_rag_conflict", group_id=gid, task_type="choice",
            prompt=f"다음은 검색된 문서입니다.\n{doc_text}\n\n{stem}", options=base["options"], option_order=SHUF,
            output_schema="choice", gold=spec, gold_type="referent", gold_confidence="medium",
            gold_basis=base["gold_basis"], hypothesis_id="H12", variant=variant, base_item_id=base_id, context_doc=doc_text,
            notes="v0.2 추가. " + note + " / 문서는 연구용 합성 텍스트(실제 출처 없음). 기준선(문서 없음)은 base_item_id 문항")

# ------------------------------------------------------------------ S7 영어 지시문 쌍 (신규 문항 분)
for (small, large), iid in s1e_ids.items():
    add_en(iid, ("The following are made-up (non-existent) Korean color adjectives. Which of the two would more likely refer to "
                 f"a darker and more dull/murky color? ({small}, {large})"))
    _NEW_S7 += 1
for gkey, iid in s1g_ids.items():
    add_en(iid, f"Sort the following Korean color adjectives (all describing {s1c_color_en[iid]}) from the one whose color feels deepest and most intense to the one that feels weakest.")
    _NEW_S7 += 1
for iid, en in list(s3a_en.items()):
    if int(iid.split("-")[1]) > 26 and iid.startswith("S3-"):
        add_en(iid, f"Which Korean word most accurately refers to the color of {en}?")
        _NEW_S7 += 1
for iid in s4b_ids:
    sent, en_opts = s4_en[iid]
    b = BY_ID[iid]
    add("S7", subset="S7_en_S4_" + b["subset"], group_id=b["group_id"], task_type="choice", lang="en",
        prompt=("Read the following Korean sentence and choose the option that best describes the meaning of the color "
                f"expression in it.\nSentence: {sent}"),
        options=en_opts, option_order=b["option_order"], output_schema="choice", gold=en_opts[b["options"].index(b["gold"])],
        gold_type=b["gold_type"], gold_confidence=b["gold_confidence"], gold_basis=b["gold_basis"], hypothesis_id="H11",
        variant=b["variant"], pair_id=iid,
        notes="영어 선택지는 한국어 선택지의 번역(순서 동일). 채점은 gold 문구 일치 또는 선택 인덱스 일치")
    _NEW_S7 += 1

_NEW_S3_EN = sum(1 for k in s3a_en if int(k.split("-")[1]) > 26)
for _r in ITEMS[_V1_N:]:
    _r["version_added"] = "0.2"

for _r in ITEMS[:_V1_N]:
    _r["version_added"] = "0.1"

# ------------------------------------------------------------------ 파일럿 표시
def mark(pred):
    for r in ITEMS:
        if pred(r):
            r["pilot"] = True


mark(lambda r: r["set_id"] == "S0")
mark(lambda r: r["subset"] == "S1a_vowel" and r["variant"] == "dark_turbid" and r["group_id"] in
     {"vowel:노랗다-누렇다", "vowel:파랗다-퍼렇다", "vowel:하얗다-허옇다"})
mark(lambda r: r["subset"] == "S1b_fortis" and r["group_id"] in {"fortis:발갛다-빨갛다", "fortis:거멓다-꺼멓다"})
mark(lambda r: r["subset"] == "S1c_affix_rank" and r["group_id"] in {"rank:black", "rank:blue"})
mark(lambda r: r["subset"] == "S1d_2x2_rank" and r["group_id"] == "rank:black_2x2")
mark(lambda r: r["set_id"] == "S2" and r["group_id"] in {"A1", "B3"})
mark(lambda r: r["subset"] == "S3a_referent" and r["variant"] == "hyper" and r["group_id"] in {"chestnut", "mugwort", "tangerine", "mouse"})
mark(lambda r: r["subset"] == "S3b_lexical_trap" and r["group_id"] == "persimmon")
mark(lambda r: r["subset"] == "S3c_etymology" and r["group_id"] in {"밤색", "갈색"})
mark(lambda r: r["item_id"] in {"S4-001", "S4-002", "S4-005", "S4-006", "S4-007", "S4-008"})
mark(lambda r: r["set_id"] == "S5" and r["group_id"] == "칙칙하다")
mark(lambda r: r["subset"] == "S6_register_rating" and r["group_id"] in {"black-gov", "black-ad"})
pilot_base = {r["item_id"] for r in ITEMS if r["pilot"]}
mark(lambda r: r["set_id"] == "S7" and r["pair_id"] in pilot_base and BY_ID[r["pair_id"]]["subset"] in {"S1a_vowel", "S3a_referent"})
mark(lambda r: r["set_id"] == "S8" and r["group_id"] == "chestnut")
mark(lambda r: r["set_id"] == "S9" and r["group_id"] == "vowel:노랗다-누렇다")

# ------------------------------------------------------------------ 검증
errors = []
ids = [r["item_id"] for r in ITEMS]
if len(ids) != len(set(ids)):
    errors.append("item_id 중복")
for r in ITEMS:
    iid = r["item_id"]
    if not r["prompt"].strip():
        errors.append(f"{iid}: prompt 비어 있음")
    if r["task_type"] in ("choice", "rank") and r["set_id"] != "S9":
        if len(set(r["options"])) != len(r["options"]):
            errors.append(f"{iid}: 선택지 중복")
        if r["task_type"] == "choice" and r["gold"] and r["gold"] not in r["options"]:
            errors.append(f"{iid}: gold가 선택지에 없음")
        if r["task_type"] == "rank" and r["gold"] and sorted(r["gold"].split(">")) != sorted(r["options"]):
            errors.append(f"{iid}: rank gold가 선택지의 순열이 아님")
    if r["task_type"] == "rating" and r["scale"] != "1-7":
        errors.append(f"{iid}: rating scale 누락")
    if any("|" in o for o in r["options"]):
        errors.append(f"{iid}: 선택지에 '|' 포함")
    for ref in ("pair_id", "base_item_id"):
        if r[ref] and r[ref] not in BY_ID:
            errors.append(f"{iid}: {ref} 참조 오류")
    if r["set_id"] == "S9" and r["variant"].startswith("justify") and not r["premise_option"]:
        errors.append(f"{iid}: premise_option 누락")
    if r["set_id"] in ("S3", "S8") and r["task_type"] == "choice" and len(r["options"]) != 4:
        errors.append(f"{iid}: 선택지 4개 아님")
# S3a 쌍 / S7 쌍 완전성
for gid, *_ in S3A:
    if (gid, "hyper") not in s3a_ids or (gid, "nohyper") not in s3a_ids:
        errors.append(f"S3a 쌍 누락: {gid}")
n_s7 = sum(1 for r in ITEMS if r["set_id"] == "S7")
exp_s7 = len(s1a_ids) + len(s1c_ids) + (len(s3a_en) - _NEW_S3_EN) + (len(s4_en) - len(s4b_ids)) + _NEW_S7
if n_s7 != exp_s7:
    errors.append(f"S7 개수 불일치 {n_s7} != {exp_s7}")

if errors:
    print("검증 오류:")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)

# ------------------------------------------------------------------ 출력
def csv_val(c, v):
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, list):
        return "|".join(v)
    return v


with open(os.path.join(OUT, "item_pool_v0.2.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for r in ITEMS:
        w.writerow({c: csv_val(c, r[c]) for c in COLS})

with open(os.path.join(OUT, "item_pool_v0.2.json"), "w", encoding="utf-8") as f:
    json.dump({"version": VERSION, "columns": COLS, "n_items": len(ITEMS), "items": ITEMS}, f, ensure_ascii=False, indent=1)

# 요약
print(f"총 {len(ITEMS)}문항 / 파일럿 {sum(r['pilot'] for r in ITEMS)}문항")
tab = collections.Counter((r["set_id"], r["task_type"]) for r in ITEMS)
for s in SETS:
    parts = {t: n for (ss, t), n in tab.items() if ss == s}
    print(f"  {s} {SETS[s]}: {sum(parts.values())}  {parts}")
sub = collections.Counter(r["subset"] for r in ITEMS)
print("subset별:", dict(sub))
