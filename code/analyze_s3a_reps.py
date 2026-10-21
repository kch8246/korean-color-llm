import pandas as pd, json, ast, re, numpy as np, sys
from scipy.stats import binomtest
f=sys.argv[1]; pool_p=sys.argv[2]
df=pd.read_json(f,lines=True)
pool=json.load(open(pool_p)); pool=pool if isinstance(pool,list) else pool['items']
gold={i['item_id']:i['gold'] for i in pool}
core=['gpt-4o','gpt-5.4','claude-sonnet-5-5','claude-haiku-5-5','gemini-3.8-flash','gemma-4-31b-it','deepseek-v4-pro','HCX-005','HCX-007','perplexity-sonar']
s=df[(df.subset=='S3a_referent')&df.model_label.isin(core)].copy()
s['ok']=s.error.isna()&s.parse_ok.astype(bool)
n_bad=int((~s.ok).sum()); s=s[s.ok]
s=s.sort_values('ts').drop_duplicates(['model_label','item_id','rep'],keep='last')
def ans(p):
    if isinstance(p,dict): return p.get('answer')
    try: return ast.literal_eval(p).get('answer')
    except Exception: return None
s['answer']=s.parsed.map(ans)
s['gold']=s.item_id.map(gold)
s['correct']=(s.answer.astype(str).str.strip()==s.gold.astype(str).str.strip()).astype(float)
print('제외(오류·파싱실패) 호출',n_bad,'/ 분석 호출',len(s))
bad_groups={'azalea','eggplant','forsythia','balloonflower'}
hyp={'azalea':'분홍색','eggplant':'보라색','forsythia':'노란색','balloonflower':'보라색'}
def accs(x,label):
    m=x.groupby(['model_label','variant']).correct.mean().unstack()
    print(label,'hyper %.3f nohyper %.3f'%(x[x.variant=='hyper'].correct.mean(),x[x.variant=='nohyper'].correct.mean()),
          '하락 모델',int((m.hyper<m.nohyper).sum()),'상승',int((m.hyper>m.nohyper).sum()))
accs(s,'전체(rep1-6)')
# 반복별 정답률
r=s.groupby(['rep','variant']).correct.mean().unstack(); print(r.round(3))
# 반복 내 hyper 하락 모델 수
for k,g in s.groupby('rep'):
    m=g.groupby(['model_label','variant']).correct.mean().unstack(); print('rep',k,'하락 모델',int((m.hyper<m.nohyper).sum()),'상승',int((m.hyper>m.nohyper).sum()))
# 모델별 6회 평균
m=s.groupby(['model_label','variant']).correct.agg(['mean','sum','count']).unstack()
print(m.round(3))
# 문항수준 짝지은 비교 (모델×group×rep)
p=s.pivot_table(index=['model_label','group_id','rep'],columns='variant',values='correct').dropna()
b=int(((p.nohyper==1)&(p.hyper==0)).sum());c=int(((p.nohyper==0)&(p.hyper==1)).sum())
print('짝 불일치',b,c,'p=%.2g'%binomtest(b,b+c).pvalue)
# 응답 변동: (모델×문항)에서 6회 응답이 일관(모두 같은 답)한 비율, hyper조건
h=s[s.variant=='hyper']
cons=h.groupby(['model_label','item_id']).answer.nunique()
print('hyper 조건 모델×문항 중 6회 답이 모두 같은 비율 %.3f'%((cons==1).mean()), '(n=%d)'%len(cons))
acc_mi=h.groupby(['model_label','item_id']).correct.mean()
print('모델×문항 정답률 분포: 항상 정답 %.3f, 항상 오답 %.3f, 혼합 %.3f'%((acc_mi==1).mean(),(acc_mi==0).mean(),((acc_mi>0)&(acc_mi<1)).mean()))
# 문항별 hyper 정답률 (60회)
it=h.groupby('group_id').correct.agg(['mean','count']).round(3); print(it.sort_values('mean'))
# 제외 강건성
accs(s[~s.group_id.isin(bad_groups)],'4문항 제외')
t=s.copy(); mm=(t.variant=='hyper')&t.group_id.isin(bad_groups)&(t.answer==t.group_id.map(hyp)); t.loc[mm,'correct']=1.0
accs(t,'4문항 상위어 인정')
# bootstrap CI over items for hyper acc and difference
rng=np.random.default_rng(1)
g=s.groupby(['group_id','variant']).correct.mean().unstack()
d=(g.nohyper-g.hyper).values; bs=[rng.choice(d,len(d)).mean() for _ in range(5000)]
print('하락폭(nohyper-hyper) 문항 단위 평균 %.3f, 95%% bootstrap [%.3f, %.3f]'%(d.mean(),*np.percentile(bs,[2.5,97.5])))
s.to_csv('results/s3a_scored_reps.csv',index=False)
