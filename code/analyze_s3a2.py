import pandas as pd, json, ast, numpy as np, sys
from scipy.stats import binomtest, spearmanr
f=sys.argv[1]; pool_p=sys.argv[2]; old_csv=sys.argv[3]
core=['gpt-4o','gpt-5.4','claude-sonnet-5-5','claude-haiku-5-5','gemini-3.8-flash','gemma-4-31b-it','deepseek-v4-pro','HCX-005','HCX-007','perplexity-sonar']
P=json.load(open(pool_p,encoding='utf-8'))['items']; G={i['item_id']:i for i in P}
new_ids={i['item_id'] for i in P if i['version_added']=='0.3'}
df=pd.read_json(f,lines=True)
s=df[df.item_id.isin(new_ids)&df.model_label.isin(core)].copy()
s['ok']=s.error.isna()&s.parse_ok.astype(bool); nbad=int((~s.ok).sum()); s=s[s.ok]
s=s.sort_values('ts').drop_duplicates(['model_label','item_id','rep'],keep='last')
def ans(p):
    if isinstance(p,dict): return p.get('answer')
    try: return ast.literal_eval(p).get('answer')
    except Exception: return None
s['answer']=s.parsed.map(ans).astype(str).str.strip()
s['gold']=s.item_id.map(lambda x:G[x]['gold'])
s['hyp']=s.item_id.map(lambda x:G[x]['options'][1] if G[x]['variant']=='hyper' else None)
s['correct']=(s.answer==s.gold).astype(float)
print('제외 호출',nbad,'분석',len(s))
def summ(x,label):
    m=x.groupby(['model_label','variant']).correct.mean().unstack()
    print(label,'hyper %.3f nohyper %.3f'%(x[x.variant=='hyper'].correct.mean(),x[x.variant=='nohyper'].correct.mean()),'하락 모델',int((m.hyper<m.nohyper).sum()),'상승',int((m.hyper>m.nohyper).sum()))
    return m
m=summ(s,'새 16쌍(6회)'); print(m.round(3))
# 상위어 선택률(hyper 조건)
h=s[s.variant=='hyper'].copy(); h['pickhyp']=(h.answer==h.hyp).astype(float)
print('상위어 선택률 %.3f'%h.pickhyp.mean())
it=h.groupby('group_id').agg(acc=('correct','mean'),hyp=('pickhyp','mean')).round(3).sort_values('acc'); print(it)
cons=h.groupby(['model_label','item_id']).answer.nunique(); print('6회 답 동일 비율 %.3f'%((cons==1).mean()))
ami=h.groupby(['model_label','item_id']).correct.mean(); print('항상정답 %.3f 항상오답 %.3f 혼합 %.3f'%((ami==1).mean(),(ami==0).mean(),((ami>0)&(ami<1)).mean()))
rng=np.random.default_rng(1)
g=s.groupby(['group_id','variant']).correct.mean().unstack(); d=(g.nohyper-g.hyper).values
bs=[rng.choice(d,len(d)).mean() for _ in range(5000)]
print('하락폭 %.3f [%.3f, %.3f]'%(d.mean(),*np.percentile(bs,[2.5,97.5])))
for r,gg in s.groupby('rep'):
    mm=gg.groupby(['model_label','variant']).correct.mean().unstack(); print('rep',r,'hyper %.3f'%gg[gg.variant=='hyper'].correct.mean(),'하락',int((mm.hyper<mm.nohyper).sum()),'상승',int((mm.hyper>mm.nohyper).sum()))
s.to_csv('results/s3a2_scored.csv',index=False)
# 통합 32쌍
old=pd.read_csv(old_csv)
c=pd.concat([old[['model_label','item_id','group_id','variant','rep','correct']],s[['model_label','item_id','group_id','variant','rep','correct']]])
mc=summ(c,'통합 32쌍(6회)'); print(mc.round(3))
g2=c.groupby(['group_id','variant']).correct.mean().unstack(); d2=(g2.nohyper-g2.hyper).values
bs=[rng.choice(d2,len(d2)).mean() for _ in range(5000)]
print('통합 하락폭 %.3f [%.3f, %.3f], 문항(쌍) 중 하락>0: %d/%d, 0: %d, <0: %d'%(d2.mean(),*np.percentile(bs,[2.5,97.5]),(d2>0).sum(),len(d2),(d2==0).sum(),(d2<0).sum()))
