import json, pathlib, re, csv
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
# remove ANSI escapes if any
plain = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', plain)
# Extract Q -> final_answer by nearest following answer in same local block
pat = re.compile(r'Question:\s*(?P<q>[^\n\r]+).*?"final_answer"\s*:\s*"(?P<a>(?:\\.|[^"\\])*)"', re.S)
pairs=[]
for m in pat.finditer(plain):
    q=m.group('q').strip()
    a=bytes(m.group('a'),'utf-8').decode('unicode_escape').strip()
    # cleanup trailing box chars
    q=q.strip(' │')
    a=a.strip(' │')
    if q and a:
        pairs.append((q,a))
print('raw pairs',len(pairs))
# dedupe by exact pair order-preserving
seen=set(); uniq=[]
for q,a in pairs:
    key=(q,a)
    if key in seen: continue
    seen.add(key); uniq.append((q,a))
print('unique pairs',len(uniq))
# also dedupe by question keeping most frequent answer
from collections import Counter, defaultdict
byq=defaultdict(list)
for q,a in pairs: byq[q].append(a)
qa=[]
for q,ans in byq.items():
    a=Counter(ans).most_common(1)[0][0]
    qa.append((q,a))
qa.sort(key=lambda x: x[0])
print('unique questions',len(qa))
for i,(q,a) in enumerate(qa[:8],1):
    print(f'[{i}] Q: {q[:120]}')
    print(f'[{i}] A: {a[:120]}')
# choose question-unique set
out=pathlib.Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\results\eval_results_from_output_qna.csv')
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['question','final_answer'])
    w.writerows(qa)
print('wrote',out)
