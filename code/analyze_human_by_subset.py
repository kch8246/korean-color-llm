import pandas as pd, numpy as np, json, glob
from itertools import combinations
pool=json.load(open('items/item_pool_v0.2.json'))['items']
by={i['item_id']:i for i in pool}
def tau(pred,gold):
    pos={x:i for i,x in enumerate(pred)}; c=d=0
    for a,b in combinations(gold,2):
        if pos[a]<pos[b]: c+=1
        else: d+=1
    return (c-d)/(c+d)
resp=[]
for p in sorted(glob.glob('survey_in/*.json')):
    r=json.loads(open(p,encoding='utf-8').read()); resp.append(r)
rows=[]
for r in resp:
    for a in r['answers']:
        if a['item_id'].startswith('ATTN'): continue
        it=by[a['item_id']]; typ=it['task_type']
        if typ=='choice': ok=float(a['answer']==it['gold'])
        elif typ=='rank':
            if not it['gold']: continue
            ok=tau(a['answer'],it['gold'].split('>'))
        else: continue
        rows.append(dict(rid=r['respondent_id'][:6],item=a['item_id'],subset=it['subset'],typ=typ,ok=ok,rt=a['rt_ms'],unf=a.get('unfamiliar',False)))
D=pd.DataFrame(rows)
print('items',D.item.nunique())
print('--- human acc/tau by subset (all 12)')
print(D.groupby(['subset','typ']).ok.agg(['mean','count']).round(3).to_string())
fast=['892975','f2b26a','35ce7d']
print('--- excluding 3 speed-flagged respondents (n=9)')
print(D[~D.rid.isin(fast)].groupby(['subset','typ']).ok.mean().round(3).to_string())
print('--- per respondent acc (choice items)')
print(D[D.typ=='choice'].groupby('rid').ok.mean().round(3).sort_values().to_string())
# model acc on same items (core models)
sc=pd.read_csv('results/scored_rep0.csv')
core=[m for m in sc.model_label.unique() if '+' not in m]
x=sc[sc.model_label.isin(core)&sc.item_id.isin(D.item.unique())]
print('--- model mean by subset (choice acc / rank tau)')
print(x.groupby('subset').agg(acc=('correct','mean'),tau=('tau','mean')).round(3).to_string())
