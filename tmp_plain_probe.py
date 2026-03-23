import json, pathlib, re
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
print('text/plain outputs',len(arr),'chars',len(plain))
for pat in ['Question:','"question"','"final_answer"','final_answer','Final Answer:','This run processed']:
    print(pat, plain.count(pat))
print('head',plain[:1200].replace('\n','\\n'))
print('tail',plain[-1200:].replace('\n','\\n'))
for i,m in enumerate(re.finditer(r'Question:\s*[^\n\r]+', plain),1):
    print('Q',i,m.group(0)[:220])
    if i>=8: break
for i,m in enumerate(re.finditer(r'"final_answer"\s*:\s*"', plain),1):
    s=max(0,m.start()-150); e=min(len(plain),m.start()+260)
    print('Actx',i,plain[s:e].replace('\n',' '))
    if i>=6: break
