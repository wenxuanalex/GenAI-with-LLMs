import json, pathlib, re
p = pathlib.Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\crewai_agentic_rag_sec.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
cell = nb['cells'][32]
outs = cell.get('outputs', [])
texts = []
for o in outs:
    d = o.get('data', {})
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, list):
                texts.append(''.join(str(x) for x in v))
            elif isinstance(v, str):
                texts.append(v)
    t = o.get('text', None)
    if t is not None:
        if isinstance(t, list):
            texts.append(''.join(str(x) for x in t))
        else:
            texts.append(str(t))
alltxt = '\n'.join(texts)
print('outputs:', len(outs))
print('chars:', len(alltxt))
for pat in ['This run processed', 'Question', 'Final Answer', 'final_answer', 'question_id', '"question"', '"final_answer"']:
    print(pat, alltxt.count(pat))
m = re.search(r'This run processed[^\n\r]*', alltxt)
print('processed_line:', m.group(0) if m else 'NONE')
print('head:', alltxt[:1200].replace('\n','\\n'))
print('tail:', alltxt[-1200:].replace('\n','\\n'))
