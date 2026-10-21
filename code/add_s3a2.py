import json, copy
P=json.load(open('items/item_pool_v0.2.json',encoding='utf-8'))
items=P['items']
tpl_h=next(i for i in items if i['item_id']=='S3-027'); tpl_n=next(i for i in items if i['item_id']=='S3-028')
# (group, 지시 대상 문장, 하위 색이름, 상위어, hyper 방해 2개, nohyper 방해 3개)
S=[
('lemon','잘 익은 레몬 껍질의 색을','레몬색','노란색',['남색','분홍색'],['남색','분홍색','회색']),
('peach','잘 익은 복숭아 겉면의 색을','복숭아색','분홍색',['남색','회색'],['남색','회색','초록색']),
('grape','잘 익은 포도(거봉) 껍질의 색을','포도색','보라색',['주황색','회색'],['주황색','회색','노란색']),
('ivory','코끼리 상아의 색을','상아색','흰색',['남색','빨간색'],['남색','빨간색','초록색']),
('jade','옥으로 만든 가락지의 색을','옥색','초록색',['분홍색','주황색'],['분홍색','주황색','회색']),
('pomegranate','잘 익어 벌어진 석류 알의 색을','석류색','빨간색',['남색','회색'],['남색','회색','초록색']),
('redbean','삶은 팥의 색을','팥색','갈색',['연두색','남색'],['연두색','남색','분홍색']),
('soil','밭의 마른 흙의 색을','흙색','갈색',['분홍색','남색'],['분홍색','남색','연두색']),
('pea','삶은 완두콩의 색을','완두색','초록색',['분홍색','회색'],['분홍색','회색','주황색']),
('cherry','잘 익은 앵두 열매의 색을','앵두색','빨간색',['남색','회색'],['남색','회색','연두색']),
('walnut','호두 껍데기의 색을','호두색','갈색',['남색','연두색'],['남색','연두색','분홍색']),
('inkstick','먹물의 색을','먹색','검은색',['분홍색','주황색'],['분홍색','주황색','연두색']),
('greengrape','청포도 알의 색을','청포도색','초록색',['분홍색','남색'],['분홍색','남색','주황색']),
('lilac','라일락꽃의 색을','라일락색','보라색',['주황색','회색'],['주황색','회색','초록색']),
('carrot','껍질을 벗긴 당근의 색을','당근색','주황색',['남색','회색'],['남색','회색','보라색']),
('amber','송진이 굳어서 만들어진 보석 호박(琥珀)의 색을','호박색','노란색',['남색','분홍색'],['남색','분홍색','회색']),
]
n=51; new=[]
for g,obj,low,hyp,dh,dn in S:
    stem=f"{obj} 가장 정확하게 지칭하는 단어는 무엇입니까?"
    for var,opts in (('hyper',[low,hyp]+dh),('nohyper',[low]+dn)):
        t=copy.deepcopy(tpl_h if var=='hyper' else tpl_n)
        t.update(item_id=f"S3-{n:03d}",group_id=g,variant=var,prompt=stem,options=opts,gold=low,
            version_added='0.3',pilot=False)
        t['gold_basis']=(f"대상 이름에서 유래한 하위 색이름이 상위어({hyp})보다 정확하다고 본 연구자 정의. 사전 및 모어 화자 확인 필요" if var=='hyper'
                         else "오답 선택지가 모두 명백히 다른 색이라 정답이 하나로 정해짐(통제 조건)")
        t['gold_confidence']='medium' if var=='hyper' else 'high'
        t['notes']='v0.3 추가(S3a 확장). '+('상위어 선택지 포함' if var=='hyper' else '상위어 제거')
        new.append(t); n+=1
P['items']=items+new; P['version']='0.3'
json.dump(P,open('items/item_pool_v0.3.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
# 설문용: 새 hyper 16 + 기존 대조 4(연두색 합의, 쥐색 합의, 개나리색·도라지색 논쟁)
keep=[i for i in new if i['variant']=='hyper']+[i for i in items if i['item_id'] in ('S3-031','S3-015','S3-027','S3-035')]
Q=dict(P); Q['items']=keep
json.dump(Q,open('surveys/pool_survey_s3a2.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
ids=[i['item_id'] for i in new]
print(len(new),ids[0],ids[-1]); open('items/new_item_ids.txt','w').write(','.join(ids))
import csv
with open('items/s3a2_items.csv','w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f); w.writerow(['item_id','group','variant','prompt','options','gold']);
    for i in new: w.writerow([i['item_id'],i['group_id'],i['variant'],i['prompt'],' / '.join(i['options']),i['gold']])
