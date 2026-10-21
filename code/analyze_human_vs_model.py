import pandas as pd, numpy as np, json, re
from scipy.stats import spearmanr, binomtest
S=pd.read_csv('survey_out/human_item_summary.csv').set_index('item_id')
sc=pd.read_csv('results/scored_rep0.csv')
pool=json.load(open('items/item_pool_v0.2.json'))['items']
by={i['item_id']:i for i in pool}
core=[m for m in sc.model_label.unique() if '+' not in m]
x=sc[sc.model_label.isin(core)&(sc.subset=='S3a_referent')]
hyp=x[x.variant=='hyper'].copy()
def hyper_of(i):
    m=re.search(r'상위어\(([^)]+)\)',by[i]['gold_basis']); return m.group(1) if m else None
hm={i:hyper_of(i) for i in hyp.item_id.unique()}
print('no hypernym parsed:',[k for k,v in hm.items() if v is None])
hyp['hyper']=hyp.item_id.map(hm); hyp['is_h']=(hyp.answer==hyp.hyper)
mh=hyp.groupby('item_id').is_h.mean()
hr={}
for i in mh.index:
    dist=json.loads(S.loc[i,'dist']); n=S.loc[i,'n']; hr[i]=dist.get(hm[i],0)/n
hr=pd.Series(hr)
df=pd.DataFrame({'gold':[by[i]['gold'] for i in mh.index],'hyper':[hm[i] for i in mh.index],'model_hyper_rate':mh,'human_hyper_rate':hr,'human_gold_rate':[json.loads(S.loc[i,'dist']).get(by[i]['gold'],0)/S.loc[i,'n'] for i in mh.index],'model_acc':hyp.groupby('item_id').correct.mean()})
print(df.round(2).to_string())
print('mean model hyper rate',mh.mean().round(3),' human',hr.mean().round(3))
print('Spearman item-level',spearmanr(df.model_hyper_rate,df.human_hyper_rate))
print('human acc (draft gold)',df.human_gold_rate.mean().round(3),'model acc',hyp.correct.mean().round(3))
cons=[i for i in mh.index if S.loc[i,'agree']>=0.75]
hyp['gid']=hyp.item_id.map(lambda i:by[i]['group_id'])
pl=x[x.variant=='nohyper'].copy(); pl['gid']=pl.item_id.map(lambda i:by[i]['group_id'])
def paired(h):
    m=h.merge(pl[['model_label','gid','correct']],on=['model_label','gid'],suffixes=('_h','_p'))
    b=int(((m.correct_p==1)&(m.correct_h==0)).sum()); c=int(((m.correct_p==0)&(m.correct_h==1)).sum())
    return len(m),m.correct_p.mean(),m.correct_h.mean(),b,c,binomtest(min(b,c),b+c,0.5).pvalue if b+c else None
print('consensus items',cons); print('consensus (pairs,plain,hyper,b,c,p):',paired(hyp[hyp.item_id.isin(cons)]))
print('all 16:',paired(hyp))
print('non-consensus 10:',paired(hyp[~hyp.item_id.isin(cons)]))
