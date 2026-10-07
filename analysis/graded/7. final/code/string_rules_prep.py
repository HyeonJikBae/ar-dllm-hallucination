import json,re,collections,unicodedata
H='/home/hyeonjik/hallucination/'
V2='/tmp/claude-1018/-home-hyeonjik-hallucination/01013139-0ac8-4fd2-9061-9f903a6ac0b6/scratchpad/v2/'
TYPE={**{r:'A' for r in['director','producer','screenwriter','composer','author']},**{r:'B' for r in['father','mother']},
 **{r:'C' for r in['genre','religion','sport','occupation','color']},**{r:'D' for r in['place of birth','country','capital','capital of']}}
def norm(s):
    s=unicodedata.normalize('NFKD',s.lower()); s=''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',s)).strip()
TITLES={'sir','dr','mr','mrs','ms','prof','professor','lord','lady','dame','the','a','an','saint','st','jr','sr','ii','iii','iv'}
ABST=re.compile(r"\b(not specified|no information|not provided|not mentioned|don t know|do not know|unknown|cannot|can t|unclear|no answer|not available|not given|not known|not sure|there is no|n a)\b")
def loopy(t):
    w=t.split()
    if len(w)>=4:
        c=collections.Counter(w)
        if c.most_common(1)[0][1]>=3 and len(w)>=5: return True
        bg=collections.Counter(zip(w,w[1:]))
        if bg.most_common(1)[0][1]>=3: return True
    return False
prop={};
for l in open(H+'data/popqa/4. popqa_paraphrased.jsonl'):
    r=json.loads(l); prop[str(r['id'])]=r['prop']
rows=[json.loads(l) for l in open(H+'analysis/graded/1. reviewed 1/qwen2.5-7b_popqa_graded.jsonl')]
r5={}
for l in open(H+'analysis/graded/5. reviewed 5/qwen2.5-7b_popqa_reclassified.jsonl'):
    x=json.loads(l); r5[(x['sample_id'],x['version'])]=x['reason']
pairs=collections.OrderedDict(); det={}
for i,r in enumerate(rows):
    if r['verdict']!='incorrect': continue
    rel=prop[r['sample_id']]; a=r['model_answer']; na=norm(a); ng=norm(r['gold_value'])
    nal={norm(x) for x in r['gold_aliases']}|{ng}; nq=norm(r['question'])
    code=None
    if not na: code='6-empty'
    elif 'question' in na.split() and ('answer' in na.split() or True) and re.search(r'\b(question|answer)\b ?:?',a.lower()) and re.search(r'(?i)question:|answer:|direct answer',a): code='6-leak'
    elif loopy(na): code='6-loop'
    elif len(na.split())<=14 and ABST.search(na): code='N-abstain'
    elif na in nal: code='C-alias'
    else:
        wa=[w for w in na.split() if w not in TITLES]; wg=[w for w in ng.split() if w not in TITLES]
        if wa and wg and set(wa)==set(wg): code='C-alias'
        elif wa and wg and TYPE.get(rel) in('A','B') and len(wa)>=2 and set(wa)<set(wg): code='C-short'
        elif wa and wg and len(wg)>=2 and wa==[wg[-1]] and TYPE.get(rel) in('A','B'): code='C-short'
        elif wa and wg and set(wg)<set(wa) and len(wa)-len(wg)<=2 and len(wa)<=6: code='C-alias'
        else:
            ws=set(nq.split())-TITLES
            if wa and len(' '.join(wa))>=3 and re.search(r'(?<![a-z0-9])'+re.escape(' '.join(wa))+r'(?![a-z0-9])',nq) : code='3-copy'
    if code: det[i]=code; continue
    t=TYPE[rel]
    key=(rel,na) if t in 'AB' and len(na.split())<=5 else (rel,ng,na)
    d=pairs.setdefault(key,{'rel':rel,'gold':r['gold_value'],'ans':a,'q':r['question'],'n':0,'idx':[]})
    d['n']+=1; d['idx'].append(i)
print('incorrect',sum(1 for r in rows if r['verdict']=='incorrect'),'| deterministic',len(det),collections.Counter(det.values()).most_common())
print('pair keys',len(pairs),'rows covered',sum(d['n'] for d in pairs.values()))
# 배치: 관계별 정렬 후 120개씩
items=sorted(pairs.items(),key=lambda kv:(kv[1]['rel'],kv[1]['gold'].lower() if len(kv[0])==3 else '',kv[1]['ans'].lower()))
B=120; nb=0
with open(V2+'pairs.json','w') as f: json.dump([[list(k),v['rel'],v['gold'],v['ans'],v['q'],v['n']] for k,v in items],f)
for b in range(0,len(items),B):
    with open(V2+f'batches/b{nb:03d}.txt','w') as f:
        for j,(k,v) in enumerate(items[b:b+B]):
            f.write(f"{b+j}\t{v['rel']}\tgold={v['gold']}\tans={v['ans'][:140]}\tq={v['q'][:90]}\tn={v['n']}\n")
    nb+=1
json.dump({str(k):v for k,v in det.items()},open(V2+'det.json','w'))
print('batches',nb)
print(collections.Counter(TYPE[v['rel']] for v in pairs.values()))
