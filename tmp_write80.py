import json, pathlib, re, csv
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
pat=re.compile(r'Question:\s*(?P<q>.+?)\s*"final_answer"\s*:\s*"(?P<a>(?:\\.|[^"\\])*)"',re.S)
byq=defaultdict(list)
for m in pat.finditer(plain):
    q=m.group('q')
    a=bytes(m.group('a'),'utf-8').decode('unicode_escape')
    q=q.replace('│',' ').replace('â',' ').replace('`',' ').replace('\u2502',' ')
    q=re.sub(r'\s+',' ',q).strip(' -:|')
    if '?' in q:
        q=q.split('?',1)[0].strip()+'?'
    a=a.replace('â',' ').replace('│',' ')
    a=re.sub(r'\s+',' ',a).strip(' -:|')
    if q and a:
        byq[q].append(a)
rows=[]
for q,ans in byq.items():
    rows.append((q, Counter(ans).most_common(1)[0][0], len(ans)))
rows.sort(key=lambda x:x[0])
out=pathlib.Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\results\eval_results_from_output_qna.csv')
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['question','final_answer'])
    for q,a,_ in rows:
        w.writerow([q,a])
print('rows',len(rows))
print('wrote',out)
print('sample1',rows[0][0][:120],'|',rows[0][1][:120])
print('sample2',rows[-1][0][:120],'|',rows[-1][1][:120])
