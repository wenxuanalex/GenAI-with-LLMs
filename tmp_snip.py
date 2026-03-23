import json, pathlib, re
p = pathlib.Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\crewai_agentic_rag_sec.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
cell = nb['cells'][32]
texts=[]
for o in cell.get('outputs',[]):
    d=o.get('data',{})
    if isinstance(d,dict):
        for v in d.values():
            texts.append(''.join(v) if isinstance(v,list) else (v if isinstance(v,str) else ''))
    t=o.get('text',None)
    if t is not None:
        texts.append(''.join(t) if isinstance(t,list) else str(t))
alltxt='\n'.join(texts)
# print snippets around first several occurrences
idxs=[m.start() for m in re.finditer(r'"final_answer"\s*:', alltxt)]
print('final_answer_occurrences',len(idxs))
for i,pos in enumerate(idxs[:8],1):
    s=max(0,pos-260); e=min(len(alltxt),pos+420)
    sn=alltxt[s:e].replace('\n',' ')
    print(f'--- occ {i} ---')
    print(sn[:680])
# Try strict pair extraction in same object
pair_pat=re.compile(r'"question"\s*:\s*"(?P<q>(?:\\.|[^"\\])*)"[^{}]{0,2500}?"final_answer"\s*:\s*"(?P<a>(?:\\.|[^"\\])*)"',re.S)
pairs=[]
for m in pair_pat.finditer(alltxt):
    q=bytes(m.group('q'),'utf-8').decode('unicode_escape')
    a=bytes(m.group('a'),'utf-8').decode('unicode_escape')
    pairs.append((q,a))
print('pairs_found',len(pairs))
if pairs:
    print('sample_q',pairs[0][0][:220])
    print('sample_a',pairs[0][1][:220])
