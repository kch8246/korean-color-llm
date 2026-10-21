import json,glob,collections,sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, binomtest
POOL='items/item_pool_v0.3.json'
P={i['item_id']:i for i in json.load(open(POOL,encoding='utf-8'))['items']}
files=sorted(glob.glob('survey_in/*.json'))
R=[json.load(open(f,encoding='utf-8')) for f in files]
rows=[];resp=[]
for r in R:
    a={x['item_id']:x for x in r['answers']}
    attn=a.get('ATTN-C',{}).get('answer')
    rts=[x['rt_ms'] for x in r['answers'] if x['item_id']!='ATTN-C']
    sec=(pd.Timestamp(r['finished'])-pd.Timestamp(r['started'])).total_seconds()
    resp.append(dict(id=r['respondent_id'],age=r['meta']['age'],region=r['meta']['region'],attn_ok=attn=='파란색',n_items=len(rts),fast_lt1s=sum(t<1000 for t in rts),median_rt=np.median(rts)/1000,total_s=sec))
    for x in r['answers']:
        if x['item_id'].startswith('S3-'): rows.append(dict(id=r['respondent_id'],item=x['item_id'],ans=x['answer'],unf=x.get('unfamiliar',False)))
resp=pd.DataFrame(resp); print(resp.to_string()); print('주의확인 통과',resp.attn_ok.sum(),'/',len(resp))
H=pd.DataFrame(rows)
def build(H):
    out=[]
    for it,g in H.groupby('item'):
        P_=P[it]; gold=P_['gold']; hyp=P_['options'][1]; n=len(g)
        c=g.ans.value_counts()
        out.append(dict(item=it,group=P_['group_id'],gold=gold,hyper=hyp,n=n,gold_n=int(c.get(gold,0)),hyp_n=int(c.get(hyp,0)),other_n=n-int(c.get(gold,0))-int(c.get(hyp,0)),unf_n=int(g.unf.sum()),dist=dict(c)))
    return pd.DataFrame(out)
T=build(H); T['gold_rate']=T.gold_n/T.n; T['hyp_rate']=T.hyp_n/T.n
pd.set_option('display.width',250); print(T.sort_values('item')[['item','gold','hyper','n','gold_n','hyp_n','other_n','unf_n','dist']].to_string())
print('전체 gold 선택률 %.3f, 상위어 선택률 %.3f'%(T.gold_n.sum()/T.n.sum(),T.hyp_n.sum()/T.n.sum()))
# 민감도: 1초 미만 다수 응답자 제외
fast=resp[resp.fast_lt1s>=3].id.tolist(); print('빠른 응답자',fast)
T2=build(H[~H.id.isin(fast)]); print('제외 후 gold %.3f hyp %.3f'%(T2.gold_n.sum()/T2.n.sum(),T2.hyp_n.sum()/T2.n.sum()))
# 모델
S=pd.read_csv('results/s3a2_scored.csv')
mh=S[S.variant=='hyper'].groupby('group_id').correct.mean(); mn=S[S.variant=='nohyper'].groupby('group_id').correct.mean()
T['model_hyper_acc']=T.group.map(mh); T['model_nohyper_acc']=T.group.map(mn)
rho,p=spearmanr(T.gold_rate,T.model_hyper_acc); print('문항단위 Spearman(사람 gold 선택률, 모델 hyper 정답률) %.2f p=%.3f'%(rho,p))
cons=T[T.gold_rate>=0.75].group.tolist(); hypmaj=T[T.hyp_n>=7].group.tolist(); print('합의(>=75%) 문항',cons); print('상위어 다수(>=7) 문항',hypmaj)
def subset(groups,label):
    x=S[S.group_id.isin(groups)]
    m=x.groupby(['model_label','variant']).correct.mean().unstack()
    print(label,len(groups),'hyper %.3f nohyper %.3f'%(x[x.variant=='hyper'].correct.mean(),x[x.variant=='nohyper'].correct.mean()),'하락 모델',int((m.hyper<m.nohyper).sum()),'상승',int((m.hyper>m.nohyper).sum()))
subset(T.group.tolist(),'전체16')
subset(cons,'합의 문항')
subset([g for g in T.group if g not in hypmaj],'상위어 다수 문항 제외')
x=S.copy(); mm=(x.variant=='hyper')&x.group_id.isin(hypmaj)&(x.answer==x.group_id.map(dict(zip(T.group,T.hyper)))); x.loc[mm,'correct']=1.0
m=x.groupby(['model_label','variant']).correct.mean().unstack(); print('상위어 다수 문항 상위어 인정: hyper %.3f nohyper %.3f 하락 모델 %d 상승 %d'%(x[x.variant=='hyper'].correct.mean(),x[x.variant=='nohyper'].correct.mean(),int((m.hyper<m.nohyper).sum()),int((m.hyper>m.nohyper).sum())))
# 문항 단위 하락폭 (합의 문항) 부트스트랩
rng=np.random.default_rng(1)
for lab,gs in (('전체',T.group.tolist()),('합의',cons),('상위어다수제외',[g for g in T.group if g not in hypmaj])):
    gg=S[S.group_id.isin(gs)].groupby(['group_id','variant']).correct.mean().unstack(); d=(gg.nohyper-gg.hyper).values
    bs=[rng.choice(d,len(d)).mean() for _ in range(5000)]; print(lab,'하락폭 %.3f [%.3f, %.3f] (n=%d)'%(d.mean(),*np.percentile(bs,[2.5,97.5]),len(d)))
T.drop(columns=['dist']).to_csv('results/human_s3a2_items.csv',index=False,encoding='utf-8-sig'); resp.to_csv('survey_in/human_s3a2_respondents.csv',index=False,encoding='utf-8-sig')
print(T[['item','group','gold_rate','hyp_rate','model_hyper_acc']].round(2).sort_values('gold_rate').to_string())
