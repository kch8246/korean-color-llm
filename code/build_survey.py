#!/usr/bin/env python3
"""
모어 화자 설문 페이지 생성기: item_pool_v0.3.json -> survey.html (단일 파일, 서버 불필요)

    python build_survey.py [item_pool_v0.3.json] [survey.html]

설문 구성
- 블록 B(전원 공통, 52문항): 정답 검증이 필요한 선택·순위 문항
    S3a 상위어 포함 16, S3b 함정 6, S1e 가상어 8, S1f 단서 충돌 4, S1b 4, S1a 저빈도 2,
    S1c/S1d/S1g 순위 10, S4 중 정답 확신도 medium 2
- 블록 R(평정, 응답자당 한 가지 폼, 30~36문항): 폼 1/2/3은 S2(자연스러움)·S5(안색)·S6(어종×상황)를 3등분
    같은 틀(프레임)의 변이형은 한 응답자에게 모두 제시해 쌍비교가 가능하게 함
- 주의 확인 문항 2개, 인구통계 문항
- 응답 완료 시 JSON 파일을 내려받거나(기본), SUBMIT_URL이 있으면 그 주소로 전송
"""
import json
import sys

POOL = sys.argv[1] if len(sys.argv) > 1 else "item_pool_v0.3.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "survey.html"
CORE = len(sys.argv) > 3 and sys.argv[3] == "core"  # 정답 검증 문항(블록 B)만, 평정 폼 없음

items = json.load(open(POOL, encoding="utf-8"))["items"]


def keep_b(i):
    if i["set_id"] in ("S0", "S7", "S8", "S9") or i["task_type"] not in ("choice", "rank"):
        return False
    st = i["subset"]
    if st == "S3a_referent":
        return i["variant"] == "hyper"
    if st == "S3b_lexical_trap":
        return i["variant"] == "trap"
    if st in ("S1e_nonce_vowel", "S1f_cue_conflict", "S1b_fortis", "S1c_affix_rank", "S1d_2x2_rank", "S1g_rank5"):
        return True
    if st == "S1a_vowel":
        return i["gold_confidence"] != "high"
    if st in ("S4_idiom", "S4b_hard_idiom"):
        return i["gold_confidence"] == "medium"
    return False


def slim(i):
    return {"id": i["item_id"], "set": i["set_id"], "subset": i["subset"], "group": i["group_id"], "type": i["task_type"],
            "variant": i["variant"], "prompt": i["prompt"], "options": i["options"]}


block_b = [slim(i) for i in items if keep_b(i)]
ratings = [slim(i) for i in items if i["task_type"] == "rating"]

# 평정 폼 3등분: 프레임/형용사/상황 단위로 나눔
forms = {1: [], 2: [], 3: []}
by_group = {}
for r in ratings:
    by_group.setdefault((r["set"], r["group"]), []).append(r)
for sid in ("S2", "S5", "S6"):
    groups = [g for (s, g) in by_group if s == sid]
    for k, g in enumerate(groups):
        forms[k % 3 + 1].extend(by_group[(sid, g)])
form_sizes = {k: len(v) for k, v in forms.items()}

ATTN_R = {"id": "ATTN-R", "set": "ATTN", "subset": "attention", "group": "ATTN", "type": "rating", "variant": "",
          "prompt": "이 문항은 응답에 집중하고 있는지 확인하는 문항입니다. 아래 1부터 7까지 점수 중 '2'를 선택해 주세요.", "options": []}
ATTN_C = {"id": "ATTN-C", "set": "ATTN", "subset": "attention", "group": "ATTN", "type": "choice", "variant": "",
          "prompt": "이 문항은 응답에 집중하고 있는지 확인하는 문항입니다. 아래 보기 중 '파란색'을 선택해 주세요.",
          "options": ["빨간색", "파란색", "노란색"]}

DATA = {"block_b": block_b, "forms": {str(k): v for k, v in forms.items()}, "attn_r": ATTN_R, "attn_c": ATTN_C,
        "core": CORE, "pool_version": json.load(open(POOL, encoding="utf-8")).get("version", "")}

HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>한국어 색깔 표현 설문</title>
<style>
:root{--bg:#f7f6f2;--fg:#1d1d1b;--mute:#6b6a64;--card:#fff;--line:#dcdad2;--acc:#2f5d8a;--acc-fg:#fff;--sel:#e3edf7}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#171717;--fg:#ececea;--mute:#a09f98;--card:#222221;--line:#3a3a37;--acc:#7aa7d4;--acc-fg:#10161c;--sel:#26394d;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#171717;--fg:#ececea;--mute:#a09f98;--card:#222221;--line:#3a3a37;--acc:#7aa7d4;--acc-fg:#10161c;--sel:#26394d;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.65 -apple-system,"Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",sans-serif}
main{max-width:640px;margin:0 auto;padding-block:20px 60px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1{font-size:1.35rem;margin:12px 0 4px}
.sub{color:var(--mute);font-size:.92rem;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 18px;margin:14px 0}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:8px 0 2px}
.bar>i{display:block;height:100%;background:var(--acc);width:0}
.prog{font-size:.85rem;color:var(--mute)}
.prompt{white-space:pre-line;font-size:1.05rem;margin:0 0 16px}
button{font:inherit;cursor:pointer}
.opt{display:block;width:100%;text-align:left;padding:13px 14px;margin:8px 0;border:1.5px solid var(--line);border-radius:10px;background:var(--card);color:var(--fg)}
.opt:hover{border-color:var(--acc)}
.opt.sel{border-color:var(--acc);background:var(--sel)}
.opt .n{display:inline-block;min-width:1.6em;font-weight:700;color:var(--acc)}
.scale{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;margin:6px 0}
.scale button{padding:14px 0;border:1.5px solid var(--line);border-radius:10px;background:var(--card);color:var(--fg);font-weight:600}
.scale button.sel{background:var(--acc);color:var(--acc-fg);border-color:var(--acc)}
.anch{display:flex;justify-content:space-between;color:var(--mute);font-size:.8rem}
.primary{background:var(--acc);color:var(--acc-fg);border:0;border-radius:10px;padding:13px 22px;font-weight:700}
.primary:disabled{opacity:.4;cursor:not-allowed}
.ghost{background:none;border:1px solid var(--line);color:var(--fg);border-radius:10px;padding:10px 14px}
.row{display:flex;gap:10px;align-items:center;justify-content:space-between;margin-top:16px;flex-wrap:wrap}
label.chk{display:flex;gap:8px;align-items:flex-start;margin:8px 0;color:var(--fg)}
input[type=text],select,textarea{width:100%;font:inherit;padding:10px;border:1.5px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
.small{font-size:.88rem;color:var(--mute)}
.unk{margin-top:10px;font-size:.9rem;color:var(--mute)}
.hide{display:none}
.steps{margin:8px 0 14px;padding-left:1.3em}
#out{font-size:.78rem;line-height:1.4;font-family:ui-monospace,Menlo,Consolas,monospace;word-break:break-all}
</style>
</head>
<body>
<main id="app"></main>
<script>
const DATA = __DATA__;
// 응답을 서버로 보내려면 주소를 넣으세요(예: Google Apps Script 웹 앱). 비워 두면 파일로 내려받습니다.
const SUBMIT_URL = "";
// 응답을 받을 연락처(카카오톡 아이디, 이메일 등). 비워 두면 안내 문구를 표시하지 않습니다.
const CONTACT = "";
const SEED_KEY = "kc_survey_v1";

function rng(seed){let s=seed>>>0;return()=>{s=(s+0x6D2B79F5)>>>0;let t=s;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};}
function shuffle(a,r){a=a.slice();for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function rid(){return Array.from(crypto.getRandomValues(new Uint8Array(8))).map(b=>b.toString(16).padStart(2,"0")).join("");}

const params = new URLSearchParams(location.search);
const S = {
  id: rid(), started: new Date().toISOString(), meta:{}, answers:[], idx:0, queue:[], form:null, seed:0, phase:"intro"
};
S.seed = parseInt(S.id.slice(0,8),16);
const R = rng(S.seed);
const f = params.get("form");
S.form = (f && DATA.forms[f]) ? f : String(1+Math.floor(R()*3));

function buildQueue(){
  const b = shuffle(DATA.block_b, R);
  b.splice(Math.floor(R()*b.length), 0, DATA.attn_c);
  let blocks;
  if(DATA.core){ blocks = [b]; }
  else {
    const rs = shuffle(DATA.forms[S.form], R);
    rs.splice(Math.floor(R()*rs.length), 0, DATA.attn_r);
    blocks = R() < 0.5 ? [b, rs] : [rs, b];
  }
  S.queue = blocks.flat().map(it => Object.assign({}, it, {
    shown: it.type==="rating" ? [] : shuffle(it.options, R)
  }));
}

const app = document.getElementById("app");
function el(h){const d=document.createElement("div");d.innerHTML=h;return d;}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}

function intro(){
  app.innerHTML = `
  <h1>한국어 색깔 표현 설문</h1>
  <div class="sub">소요 시간 약 ${DATA.core?"10":"15"}분 · 평소 쓰는 말 감각대로 답해 주세요</div>
  <div class="card">
    <p>이 설문은 한국어 색채 어휘에 대한 모어 화자의 판단을 수집해 언어 모델의 응답과 비교하는 연구에 사용됩니다.
    이름, 연락처 등 개인을 식별하는 정보는 수집하지 않으며, 응답은 통계 분석에만 사용됩니다. 중간에 그만두어도 불이익이 없습니다.</p>
    <label class="chk"><input type="checkbox" id="c1"> <span>위 내용을 읽었고 연구 참여에 동의합니다.</span></label>
    <label class="chk"><input type="checkbox" id="c2"> <span>저는 한국어를 모어(첫 번째로 배운 언어)로 사용합니다.</span></label>
    <label class="chk"><input type="checkbox" id="c3"> <span>만 18세 이상입니다.</span></label>
    <div class="row"><span class="small">세 항목에 모두 체크해야 시작할 수 있습니다.</span><button class="primary" id="go" disabled>시작하기</button></div>
  </div>`;
  const chk=["c1","c2","c3"].map(i=>document.getElementById(i));
  const go=document.getElementById("go");
  chk.forEach(c=>c.onchange=()=>go.disabled=!chk.every(x=>x.checked));
  go.onclick=()=>{S.meta.consent=true;S.meta.native_ko=true;S.meta.adult=true;buildQueue();S.idx=0;S.phase="items";render();};
}

function render(){
  if(S.phase==="intro")return intro();
  if(S.phase==="demo")return demo();
  if(S.phase==="done")return done();
  const it=S.queue[S.idx];
  const pct=Math.round(S.idx/S.queue.length*100);
  const t0=performance.now();
  let body="";
  if(it.type==="rating"){
    body=`<div class="scale">${[1,2,3,4,5,6,7].map(n=>`<button data-v="${n}">${n}</button>`).join("")}</div>
    <div class="anch"><span>1 (전혀/매우 아니다)</span><span>7 (매우 그렇다)</span></div>`;
  }else if(it.type==="choice"){
    body=it.shown.map((o,i)=>`<button class="opt" data-v="${esc(o)}">${esc(o)}</button>`).join("");
  }else{
    body=`<div class="small">강한 것부터 순서대로 눌러 주세요. 잘못 눌렀다면 '다시 정하기'를 누르세요.</div>`+
         it.shown.map(o=>`<button class="opt" data-v="${esc(o)}"><span class="n"></span>${esc(o)}</button>`).join("")+
         `<div class="row"><button class="ghost" id="reset">다시 정하기</button></div>`;
  }
  const unk = it.set==="ATTN" ? "" : `<label class="chk unk"><input type="checkbox" id="unk"> <span>이 문항에 처음 보거나 뜻을 모르는 표현이 있습니다.</span></label>`;
  app.innerHTML=`
    <div class="prog">${S.idx+1} / ${S.queue.length}</div><div class="bar"><i style="width:${pct}%"></i></div>
    <div class="card"><p class="prompt">${esc(it.prompt)}</p>${body}${unk}
    <div class="row"><button class="ghost" id="back" ${S.idx===0?"disabled":""}>이전</button><button class="primary" id="next" disabled>다음</button></div></div>`;
  const next=document.getElementById("next");
  let val=null; const order=[];
  const prev=S.answers.find(a=>a.item_id===it.id);
  function setNext(){
    if(it.type==="rank") next.disabled = order.length!==it.shown.length;
    else next.disabled = val===null;
  }
  if(it.type==="rating"){
    app.querySelectorAll(".scale button").forEach(b=>b.onclick=()=>{app.querySelectorAll(".scale button").forEach(x=>x.classList.remove("sel"));b.classList.add("sel");val=+b.dataset.v;setNext();});
  }else if(it.type==="choice"){
    app.querySelectorAll(".opt").forEach(b=>b.onclick=()=>{app.querySelectorAll(".opt").forEach(x=>x.classList.remove("sel"));b.classList.add("sel");val=b.dataset.v;setNext();});
  }else{
    const btns=[...app.querySelectorAll(".opt")];
    const paint=()=>btns.forEach(b=>{const k=order.indexOf(b.dataset.v);b.classList.toggle("sel",k>=0);b.querySelector(".n").textContent=k>=0?(k+1)+".":"";});
    btns.forEach(b=>b.onclick=()=>{if(order.includes(b.dataset.v))return;order.push(b.dataset.v);paint();setNext();});
    document.getElementById("reset").onclick=()=>{order.length=0;paint();setNext();};
  }
  document.getElementById("back").onclick=()=>{if(S.idx>0){S.idx--;render();}};
  next.onclick=()=>{
    const rec={item_id:it.id,set:it.set,subset:it.subset,group:it.group,variant:it.variant,type:it.type,
      answer: it.type==="rank"?order.slice():val, shown_order: it.shown, rt_ms: Math.round(performance.now()-t0),
      unfamiliar: !!(document.getElementById("unk")&&document.getElementById("unk").checked)};
    const k=S.answers.findIndex(a=>a.item_id===it.id); if(k>=0)S.answers[k]=rec; else S.answers.push(rec);
    try{localStorage.setItem(SEED_KEY,JSON.stringify({id:S.id,n:S.answers.length}));}catch(e){}
    S.idx++; if(S.idx>=S.queue.length){S.phase="demo";} render(); window.scrollTo(0,0);
  };
  // 이전 응답 복원
  if(prev){
    if(it.type==="rating"){const b=app.querySelector(`.scale button[data-v="${prev.answer}"]`);if(b)b.click();}
    else if(it.type==="choice"){[...app.querySelectorAll(".opt")].find(x=>x.dataset.v===prev.answer)?.click();}
    else{prev.answer.forEach(v=>[...app.querySelectorAll(".opt")].find(x=>x.dataset.v===v)?.click());}
    if(prev.unfamiliar&&document.getElementById("unk"))document.getElementById("unk").checked=true;
  }
}

function demo(){
  app.innerHTML=`<h1>마지막 질문</h1><div class="card">
  <p>연령대</p><select id="age"><option value="">선택</option><option>18~24</option><option>25~34</option><option>35~44</option><option>45~54</option><option>55~64</option><option>65 이상</option></select>
  <p>주로 자란 지역(선택)</p><select id="reg"><option value="">선택 안 함</option><option>서울·경기·인천</option><option>충청·강원</option><option>영남</option><option>호남·제주</option><option>해외</option></select>
  <p>색깔 표현에 대한 의견이나 이상했던 문항이 있다면 적어 주세요(선택).</p><textarea id="cm" rows="3"></textarea>
  <div class="row"><span class="small">연령대는 필수입니다.</span><button class="primary" id="fin" disabled>제출</button></div></div>`;
  const age=document.getElementById("age"),fin=document.getElementById("fin");
  age.onchange=()=>fin.disabled=!age.value;
  fin.onclick=()=>{S.meta.age=age.value;S.meta.region=document.getElementById("reg").value;S.meta.comment=document.getElementById("cm").value;S.phase="done";submit();render();};
}

function payload(){
  const slim = S.answers.map(a=>({item_id:a.item_id,type:a.type,answer:a.answer,rt_ms:a.rt_ms,unfamiliar:a.unfamiliar}));
  return {survey:"kc-color-v1",pool_version:DATA.pool_version,respondent_id:S.id,form:DATA.core?"core":S.form,started:S.started,finished:new Date().toISOString(),
    meta:S.meta,answers:slim};
}
async function submit(){
  S.sent=null;
  if(!SUBMIT_URL)return;
  try{await fetch(SUBMIT_URL,{method:"POST",mode:"no-cors",headers:{"Content-Type":"text/plain"},body:JSON.stringify(payload())});S.sent=true;}catch(e){S.sent=false;}
}
function done(){
  const text=JSON.stringify(payload());
  app.innerHTML=`<h1>감사합니다</h1>
  <div class="card"><p>응답이 모두 기록되었습니다. 마지막으로 아래 응답 내용을 연구자에게 보내 주세요.</p>
  <ol class="steps"><li>‘응답 복사하기’를 누릅니다.</li><li>카카오톡, 문자, 이메일 중 편한 곳에 붙여 넣어 보냅니다.</li></ol>
  ${CONTACT?`<p class="small">보낼 곳: <b>${esc(CONTACT)}</b></p>`:""}
  <div class="row" style="justify-content:flex-start"><button class="primary" id="cp">응답 복사하기</button><span id="cpmsg" class="small" aria-live="polite"></span></div>
  <p class="small" style="margin-top:16px">복사가 되지 않으면 아래 상자를 눌러 전체를 선택한 뒤 직접 복사해 주세요.</p>
  <textarea id="out" rows="6" readonly aria-label="응답 내용">${esc(text)}</textarea></div>`;
  const ta=document.getElementById("out"), msg=document.getElementById("cpmsg");
  const sel=()=>{ta.focus();ta.select();ta.setSelectionRange(0,ta.value.length);};
  ta.onclick=sel;
  document.getElementById("cp").onclick=async()=>{
    try{await navigator.clipboard.writeText(text);msg.textContent="복사되었습니다. 이제 붙여 넣어 보내 주세요.";}
    catch(e){sel();try{document.execCommand("copy");msg.textContent="복사되었습니다. 이제 붙여 넣어 보내 주세요.";}catch(_){msg.textContent="상자의 내용을 직접 복사해 주세요.";}}
  };
}
render();
</script>
</body>
</html>
"""

FULL = HTML.replace("__DATA__", json.dumps(DATA, ensure_ascii=False))
open(OUT, "w", encoding="utf-8").write(FULL)
# 아티팩트용: 문서 골격(doctype/html/head/body)을 빼고 title+meta+style+본문만
import re as _re
art = _re.sub(r"<!doctype html>\s*<html[^>]*>\s*<head>\s*<meta charset[^>]*>\s*<meta name=\"viewport\"[^>]*>\s*", "", FULL)
art = art.replace("</head>\n<body>", "").replace("</body>\n</html>", "").replace("</body></html>", "")
art = art.replace("<title>한국어 색깔 표현 설문</title>", '<title>한국어 색깔 표현 설문</title>\n<meta name="claude:viewport-fit-sides" content="cover">', 1)
open(OUT.replace(".html", "_artifact.html"), "w", encoding="utf-8").write(art)
print(f"{OUT} 생성: 블록 B {len(block_b)}문항, 평정 폼 {form_sizes} (+주의 확인 2)")
