import json, pathlib, re, html
p = pathlib.Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\crewai_agentic_rag_sec.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
cell=nb['cells'][32]
texts=[]
for o in cell.get('outputs',[]):
    d=o.get('data',{})
    if isinstance(d,dict):
        for v in d.values():
            if isinstance(v,list): texts.append(''.join(str(x) for x in v))
            elif isinstance(v,str): texts.append(v)
    t=o.get('text',None)
    if t is not None: texts.append(''.join(t) if isinstance(t,list) else str(t))
alltxt='\n'.join(texts)
clean=re.sub(r'<[^>]+>',' ',alltxt)
clean=html.unescape(clean)
clean=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',clean)
for i,m in enumerate(re.finditer(r'"final_answer"\s*:', clean),1):
    s=max(0,m.start()-1200); e=min(len(clean),m.start()+800)
    sn=clean[s:e]
    keys=sorted(set(re.findall(r'"([A-Za-z0-9_]+)"\s*:', sn)))
    print(f'--- {i} keys ---', keys)
    print(sn[:900].replace('\n',' '))
    if i>=6: break
print('search plain Question:', len(re.findall(r'Question\s*[:\-]', clean)))
print('search question text lines sample:')
for j,mm in enumerate(re.finditer(r'Question\s*[:\-][^\n\r]{10,200}', clean),1):
    print(mm.group(0)[:220])
    if j>=10: break
