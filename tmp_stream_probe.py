import json, pathlib, re
p=pathlib.Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\crewai_agentic_rag_sec.ipynb')
nb=json.loads(p.read_text(encoding='utf-8'))
cell=nb['cells'][32]
streams=[]
for o in cell.get('outputs',[]):
    if o.get('output_type')=='stream':
        t=o.get('text','')
        streams.append(''.join(t) if isinstance(t,list) else str(t))
stream_txt='\n'.join(streams)
print('stream outputs',len(streams),'chars',len(stream_txt))
print('Question:',stream_txt.count('Question:'))
print('final_answer:',stream_txt.count('"final_answer"'))
print('This run processed:',stream_txt.count('This run processed'))
print('sample first 1500:')
print(stream_txt[:1500].replace('\n','\\n'))
print('sample tail 1000:')
print(stream_txt[-1000:].replace('\n','\\n'))
# show first few occurrences context
for i,m in enumerate(re.finditer(r'Question:\s*(.+)', stream_txt),1):
    print('Q',i,m.group(1)[:180])
    if i>=5: break
for i,m in enumerate(re.finditer(r'"final_answer"\s*:\s*"', stream_txt),1):
    s=max(0,m.start()-180); e=min(len(stream_txt),m.start()+260)
    print('Actx',i,stream_txt[s:e].replace('\n',' '))
    if i>=5: break
