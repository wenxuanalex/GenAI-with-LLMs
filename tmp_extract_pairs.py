import json, pathlib, re, html, csv
p = pathlib.Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\crewai_agentic_rag_sec.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
cell = nb['cells'][32]
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
# strip html tags and decode entities
clean = re.sub(r'<[^>]+>', ' ', alltxt)
clean = html.unescape(clean)
# strip ANSI escapes
clean = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', clean)
# collapse repeated spaces only where excessive
clean = re.sub(r'[ \t]{2,}', ' ', clean)
print('clean chars', len(clean))
print('question key count', clean.count('"question"'))
print('final_answer key count', clean.count('"final_answer"'))
# try extract compact json object snippets that include both keys
obj_pat = re.compile(r'\{[^{}]{0,8000}?"question"\s*:\s*"(?:\\.|[^"\\])*"[^{}]{0,12000}?"final_answer"\s*:\s*"(?:\\.|[^"\\])*"[^{}]{0,6000}?\}', re.S)
objs = obj_pat.findall(clean)
print('candidate objects', len(objs))
pair_pat = re.compile(r'"question"\s*:\s*"(?P<q>(?:\\.|[^"\\])*)".*?"final_answer"\s*:\s*"(?P<a>(?:\\.|[^"\\])*)"', re.S)
pairs=[]
for obj in objs:
    m = pair_pat.search(obj)
    if not m:
        continue
    q = bytes(m.group('q'),'utf-8').decode('unicode_escape')
    a = bytes(m.group('a'),'utf-8').decode('unicode_escape')
    if q.strip() and a.strip():
        pairs.append((q.strip(), a.strip()))
# dedupe keep order
seen=set(); uniq=[]
for q,a in pairs:
    key=(q,a)
    if key in seen: continue
    seen.add(key); uniq.append((q,a))
print('unique pairs', len(uniq))
for i,(q,a) in enumerate(uniq[:5],1):
    print(f'[{i}] Q:', q[:120])
    print(f'[{i}] A:', a[:120])
# write csv regardless for inspection
out = pathlib.Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\results\eval_results_from_output_qna.csv')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='', encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['question','final_answer'])
    w.writerows(uniq)
print('wrote', out)
