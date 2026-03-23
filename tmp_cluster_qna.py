import json, pathlib, re, csv
from difflib import SequenceMatcher
from collections import defaultdict, Counter
p=pathlib.Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\crewai_agentic_rag_sec.ipynb')
nb=json.loads(p.read_text(encoding='utf-8'))
cell=nb['cells'][32]
arr=[]
for o in cell.get('outputs',[]):
    d=o.get('data',{})
    if isinstance(d,dict) and 'text/plain' in d:
        v=d['text/plain']
        arr.append(''.join(v) if isinstance(v,list) else str(v))
plain='\n'.join(arr)
plain=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',plain)
# extract raw pairs
pat=re.compile(r'Question:\s*(?P<q>.+?)\s*"final_answer"\s*:\s*"(?P<a>(?:\\.|[^"\\])*)"',re.S)
pairs=[]
for m in pat.finditer(plain):
    q=m.group('q')
    a=bytes(m.group('a'),'utf-8').decode('unicode_escape')
    # normalize garbage and layout artifacts
    q=q.replace('│',' ').replace('â',' ').replace('`',' ').replace('\u2502',' ')
    q=re.sub(r'\s+',' ',q).strip(' -:|')
    # keep only leading part up to question mark if present
    if '?' in q:
        q=q.split('?',1)[0].strip()+'?'
    a=a.replace('â',' ').replace('│',' ')
    a=re.sub(r'\s+',' ',a).strip(' -:|')
    if q and a:
        pairs.append((q,a))
print('pairs',len(pairs))
# initial unique by exact question
byq=defaultdict(list)
for q,a in pairs: byq[q].append(a)
questions=list(byq.keys())
print('exact unique q',len(questions))
# cluster similar questions
clusters=[]
for q in questions:
    placed=False
    for cl in clusters:
        rep=cl[0]
        sim=SequenceMatcher(None,q.lower(),rep.lower()).ratio()
        if sim>=0.93 or (q.lower().startswith(rep.lower()[:40]) or rep.lower().startswith(q.lower()[:40])):
            cl.append(q); placed=True; break
    if not placed:
        clusters.append([q])
print('clusters',len(clusters))
final_rows=[]
for cl in clusters:
    # longest question as canonical
    cq=max(cl,key=len)
    answers=[]
    for q in cl: answers.extend(byq[q])
    ca=Counter(answers).most_common(1)[0][0]
    final_rows.append((cq,ca,len(cl),len(answers)))
# sort by question text for determinism
final_rows.sort(key=lambda x:x[0])
print('final rows',len(final_rows))
for i,(q,a,c,n) in enumerate(final_rows[:12],1):
    print(f'[{i}] c={c} n={n} Q={q[:95]}')
# write
out=pathlib.Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\results\eval_results_from_output_qna.csv')
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['question','final_answer'])
    for q,a,_,_ in final_rows:
        w.writerow([q,a])
print('wrote',out)
