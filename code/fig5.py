import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for p in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in p: fm.fontManager.addfont(p)
plt.rcParams.update({"font.family":"Noto Sans CJK KR","font.size":7.5,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,"savefig.dpi":300})
t=pd.read_csv('results/table_s8_search_rates.csv')
V=['support','irrelevant','biased_1','biased_3']; L=['지지','무관','편향 1건','편향 3건']
fig,axs=plt.subplots(1,2,figsize=(130/25.4,58/25.4),sharey=True)
for ax,(m,name,c) in zip(axs,[('gpt-4o','GPT-4o','#eb6834'),('claude-sonnet-5-5','Claude Sonnet 5.5','#2a78d6')]):
    for lab,ls,mk,fc in [(m,'--','o','white'),(m+'+websearch','-','s',c)]:
        d=t[t.model_label==lab].set_index('var').loc[V,'mean']
        ax.plot(range(4),d.values,ls,color=c,marker=mk,mfc=fc,ms=4.5,lw=1.1,label='검색 있음' if 'web' in lab else '검색 없음')
    ax.set_xticks(range(4)); ax.set_xticklabels(L); ax.set_title(name,fontsize=8); ax.set_ylim(-0.03,1.05)
    ax.grid(axis='y',color='#ddd',lw=.5)
    ax.legend(frameon=False,loc='upper left',fontsize=7)
axs[0].set_ylabel('상위어 선택 비율')
fig.tight_layout()
fig.savefig('figures/fig5_s8_search.png',bbox_inches='tight',facecolor='white'); fig.savefig('figures/fig5_s8_search.pdf',bbox_inches='tight',facecolor='white')
