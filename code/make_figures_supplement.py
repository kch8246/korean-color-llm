#!/usr/bin/env python3
"""보충 그림: 그림 4(문서 조건별 상위어 선택 비율, 첫 호출), 그림 5·6(검색 유무, 첫 호출과 6회 평균).

저장소 루트에서 실행:  python code/make_figures_supplement.py
입력: results/all_trials_scored_compact.csv   출력: results/figures/fig4_*.png, fig5_*.png, fig6_*.png
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from decimal import Decimal, ROUND_HALF_UP
for p in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in p or "NotoSansCJK-Bold" in p: fm.fontManager.addfont(p)
plt.rcParams.update({"font.family":"Noto Sans CJK KR","font.size":7.5,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"savefig.dpi":300})
OUT="results/figures/"
def r(v,n=3): return format(Decimal(repr(round(float(v),10))).quantize(Decimal("0."+"0"*n),rounding=ROUND_HALF_UP),"f")
def r2(v): return r(v,2)
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/d
    return max(0,c-h),min(1,c+h)
W=130/25.4; INK,MUT,GRID,BLUE,ORANGE,GR="#1a1a1a","#666666","#dddddd","#2a78d6","#eb6834","#555555"
SHORT={"claude-haiku-5-5":"Claude Haiku 5.5","claude-sonnet-5-5":"Claude Sonnet 5.5","gemini-3.8-flash":"Gemini 3.8 Flash","deepseek-v4-pro":"DeepSeek V4 Pro","perplexity-sonar":"Perplexity Agent","gemma-4-31b-it":"Gemma 4 31B","gpt-5.4":"GPT-5.4","HCX-007":"HCX-007","gpt-4o":"GPT-4o","HCX-005":"HCX-005"}
d=pd.read_csv("results/all_trials_scored_compact.csv",encoding="utf-8-sig",low_memory=False)
d["search"]=d.search.astype(str).str.lower().eq("true"); d["c"]=d.correct
d.loc[(d.task_type=="choice")&d.gold.notna()&d.c.isna(),"c"]=0.0
d=d[d.version_added.astype(str)!="0.3"]
# ---- 그림 4 자료: 첫 호출의 문서 조건별 상위어 선택 비율
b1=d[d.model_label.isin(SHORT)&~d.search&(d.rep==1)]
R=b1[b1.set_id=="S8"].pivot_table(index="model_label",columns="variant",values="chose_hyper")
R["no_doc"]=b1[(b1.subset=="S3a_referent")&(b1.variant=="hyper")].groupby("model_label").chose_hyper.mean()
conds=["support","irrelevant","no_doc","biased_1","biased_3"]
clab=["지지\n문서","무관\n문서","문서\n없음","편향 문서\n1건","편향 문서\n3건"]
R=R[conds].loc[list(SHORT)]
# ---- 그림 5·6 자료
rng=np.random.default_rng(1); D={}
for m in ["gpt-4o","claude-sonnet-5-5"]:
    o={}
    for lab,key in [(m,"off"),(m+"+websearch","on")]:
        x=d[(d.model_label==lab)&d.set_id.isin(["S3","S4"])&d.c.notna()]
        r1=x[x.rep==1]; k=int(r1.c.sum()); n=len(r1); lo,hi=wilson(k,n)
        im=x[x.rep.between(1,6)].groupby("item_id").c.mean().values
        bs=[im[rng.integers(0,len(im),len(im))].mean() for _ in range(5000)]
        o[key]=dict(first=k/n,flo=lo,fhi=hi,six=float(im.mean()),slo=float(np.percentile(bs,2.5)),shi=float(np.percentile(bs,97.5)))
        s8=d[(d.model_label==lab)&(d.set_id=="S8")]
        o[key+"_s8"]={v:dict(first=float(s8[(s8.rep==1)&(s8.variant==v)].chose_hyper.mean()),six=float(s8[s8.rep.between(1,6)&(s8.variant==v)].groupby("item_id").chose_hyper.mean().mean())) for v in ["support","irrelevant","biased_1","biased_3"]}
    D[m]=o
MS=[("gpt-4o","GPT-4o"),("claude-sonnet-5-5","Claude Sonnet 5.5")]
# ---- 그림 4
fig,ax=plt.subplots(figsize=(130/25.4,3.6)); x=np.arange(5)
mk=["o","s","^","D","v","P","X","<",">","h"]
for (m,row),k in zip(R.iterrows(),mk):
    ax.plot(x,row.values,color="#8c8c8c",lw=0.9,marker="o",ms=2,mfc="#8c8c8c",mec="#8c8c8c",zorder=2)
mu=R.mean(); ax.plot(x,mu.values,color=BLUE,lw=2.2,marker="o",ms=4.5,zorder=4)
for xi,v in zip(x,mu.values):
    if xi in (0,1,2,3): ax.text(xi+0.07,v+(0.05 if xi==0 else -0.075),r2(v),ha="left",fontsize=6.6,color=BLUE,fontweight="bold",bbox=dict(fc="white",ec="none",pad=0.5,alpha=0.85),zorder=6)
# 오른쪽 라벨(겹침 완화)
end=R["biased_3"].sort_values(); ys=[]; gap=0.058
pos={}
for i,(m,v) in enumerate(end.items()):
    pos[m]=v if not ys else max(v,ys[-1]+gap); ys.append(pos[m])
over=ys[-1]-end.iloc[-1]
if over>0:
    for m in pos: pos[m]-=over/2
for m,v in end.items():
    ax.plot([4.04,4.2],[v,pos[m]],color="#aaaaaa",lw=0.5,zorder=1)
    ax.text(4.23,pos[m],f"{SHORT[m]}  "+r2(v),va="center",fontsize=6.2,color=INK)
ax.set_xticks(x); ax.set_xticklabels(clab,fontsize=6.8)
ax.set_ylim(-0.03,1.12); ax.set_xlim(-0.3,5.35)
ax.set_ylabel("상위어(편향 답) 선택 비율",fontsize=7)
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); ax.axvline(2,color=GRID,lw=0.8,ls="--")
ax.text(-0.25,1.11,"굵은 파란선 = 10개 모델 평균, 회색선 = 개별 모델",fontsize=6.3,color=MUT,va="top")
fig.savefig(OUT+"fig4_s8_doc_conditions.png",bbox_inches="tight",facecolor="white"); plt.close(fig)
# ---- 그림 5
fig,ax=plt.subplots(figsize=(W,3.0))
xt=[];xl=[]
for gi,(m,lab) in enumerate(MS):
    for si,(kind,nm) in enumerate([("first","첫 호출"),("six","6회 평균")]):
        cx=gi*2.4+si*1.0; xt.append(cx); xl.append(nm)
        for key,dx,col,mk in [("off",-0.17,GR,"o"),("on",0.17,BLUE,"D")]:
            o=D[m][key]; v=o[kind]; lo,hi=(o["flo"],o["fhi"]) if kind=="first" else (o["slo"],o["shi"])
            ax.errorbar([cx+dx],[v],yerr=[[v-lo],[hi-v]],fmt=mk,color=col,ms=5,capsize=2.5,lw=1.0,mfc=("white" if kind=="first" else col),mew=1.0)
            ax.text(cx+dx+(-0.09 if key=="off" else 0.09),v,r(v),ha=("right" if key=="off" else "left"),va="center",fontsize=6.3,color=INK)
    ax.text(gi*2.4+0.5,1.028,lab,ha="center",va="bottom",fontsize=7.6,fontweight="bold")
ax.set_xticks(xt); ax.set_xticklabels(xl,fontsize=7); ax.set_xlim(-0.75,4.15); ax.set_ylim(0.80,1.045); ax.set_yticks([0.80,0.85,0.90,0.95,1.00])
ax.set_ylabel("정답률 (95% 신뢰구간)",fontsize=7); ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); ax.axvline(1.7,color=GRID,lw=0.8,ls="--")
h=[Line2D([],[],marker="o",color=GR,ls="",ms=5,label="검색 없음"),Line2D([],[],marker="D",color=BLUE,ls="",ms=5,label="웹 검색 사용")]
ax.legend(handles=h,loc="lower right",fontsize=6.6,frameon=False,ncol=2,bbox_to_anchor=(1.0,-0.005))
fig.savefig(OUT+"fig5_search_accuracy.png",bbox_inches="tight",facecolor="white"); plt.close(fig)
# ---- 그림 6
conds=["support","irrelevant","biased_1","biased_3"]; cl=["지지","무관","편향 1건","편향 3건"]
fig,axs=plt.subplots(1,2,figsize=(W,2.7),sharey=True)
for ax,(m,lab),col in zip(axs,MS,[ORANGE,BLUE]):
    x=np.arange(4)
    for key,ls,mk,mfc in [("off_s8","--","o","white"),("on_s8","-","s",None)]:
        f=[D[m][key][c]["first"] for c in conds]; s=[D[m][key][c]["six"] for c in conds]
        ax.plot(x,f,ls=ls,color="#b0b0b0",lw=0.9,marker=mk,ms=3.2,mfc=("white" if mfc else "#b0b0b0"),mec="#b0b0b0",zorder=2)
        ax.plot(x,s,ls=ls,color=col,lw=1.5,marker=mk,ms=5,mfc=(mfc or col),mec=col,zorder=3)
    for i in (2,3):
        a=D[m]["off_s8"][conds[i]]["six"]; b=D[m]["on_s8"][conds[i]]["six"]
        ax.text(i+0.1,a+0.045,r(a,2),fontsize=6.2,color=INK,ha="left"); ax.text(i+0.1,b-0.085,r(b,2),fontsize=6.2,color=INK,ha="left")
    ax.set_title(lab,fontsize=7.8,pad=4); ax.set_xticks(x); ax.set_xticklabels(cl,fontsize=6.8); ax.set_xlim(-0.3,3.6); ax.set_ylim(-0.03,1.08)
    ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
    h=[Line2D([],[],ls="--",marker="o",color=col,mfc="white",ms=5,lw=1.5,label="검색 없음 (6회 평균)"),Line2D([],[],ls="-",marker="s",color=col,ms=5,lw=1.5,label="검색 있음 (6회 평균)"),Line2D([],[],ls="-",marker="s",color="#b0b0b0",ms=3.2,lw=0.9,label="회색: 첫 호출")]
    ax.legend(handles=h,loc="upper left",fontsize=6.0,frameon=False,handlelength=2.6)
axs[0].set_ylabel("상위어 선택 비율",fontsize=7)
fig.tight_layout(w_pad=1.2); fig.savefig(OUT+"fig6_s8_search.png",bbox_inches="tight",facecolor="white"); plt.close(fig)
