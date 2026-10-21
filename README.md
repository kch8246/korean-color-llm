# korean-color-llm

한국어 색채 어휘 처리에 나타나는 대규모 언어모델의 화용론적 숙달과 맥락 민감성 — 석사학위논문(동의대학교 대학원, 김찬호)의 실험 코드와 자료.

Code and data for a master's thesis on how LLMs handle Korean color terms and how their judgments shift with context (answer options, external documents).

## 구성
- `items/` 문항 풀. `item_pool_v0.2.json`(436문항, S0~S9), `item_pool_v0.3.json`(S3a 확장 32문항 추가), 코드북과 변경 기록.
- `code/` 실험 실행(`run_experiment.py`), 설문 생성·병합(`build_survey.py`, `merge_survey.py`), 문항 생성·분석 스크립트.
- `surveys/` 모어 화자 설문(단일 HTML)과 설문용 문항 JSON.
- `results/` 모델 응답 채점 결과(S3a 6회 반복, S3a 확장), 모어 화자 응답의 문항별 집계, 그림.

## 실행
1. 필요한 패키지: Python 3.10+, `requests`, `pandas`, `scipy`.
2. API 키는 코드에 넣지 말고 환경변수로 설정합니다 (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `PERPLEXITY_API_KEY`, `CLOVA_API_KEY`).
3. 저장소 루트에서 `python code/run_experiment.py --help`로 옵션을 확인합니다(`--pool --all --core --ext --items --reps --tag --models --out`). 문항 풀은 `items/item_pool_v0.3.json`을 자동으로 찾습니다. 이미 호출한 `model|temp|item|rep`는 건너뛰고 이어서 실행합니다.
   - 본 실험 436문항(문항당 1회): `python code/run_experiment.py --all --core --reps 1`
   - S3a 확장 32문항(6회): `python code/run_experiment.py --all --ext --reps 6`
   - S3a 전체 반복 호출 등 일부만: `--items S3-027,S3-028 --reps 6`
4. 분석 스크립트는 저장소 루트에서 실행하도록 상대 경로로 썼습니다. 모델 원 응답(`trials.jsonl`)과 응답자 개별 파일(`survey_in/`)이 필요한 스크립트(`analyze_final.py`, `analyze_human_*`, `make_figures.py`, `fig5.py`)는 해당 파일을 따로 두어야 실행됩니다.

## 주의
- 정답은 연구자가 정한 초안이며, 모어 화자 12명의 점검 결과는 논문 Ⅳ.10, Ⅳ.12에 있습니다.
- 모델 원 응답 파일(`trials.jsonl`)과 응답자 개별 파일은 용량과 개인정보 때문에 포함하지 않았습니다.
- 모델 버전과 서비스는 2026년 10월 기준입니다.
