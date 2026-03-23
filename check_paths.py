import sys
sys.path.insert(0, r'c:\Users\wenxu\GenAI-with-LLMs')
from config import CONFIG
print(f'sec_chunks_path: {CONFIG["sec_chunks_path"]}')
print(f'chroma_db_path: {CONFIG["chroma_db_path"]}')
print(f'sec_eval_csv_path: {CONFIG["sec_eval_csv_path"]}')
from pathlib import Path
print(f'\nResolved locations:')
print(f'  sec_chunks_path exists: {Path(CONFIG["sec_chunks_path"]).exists()}')
print(f'  chroma_db_path exists: {Path(CONFIG["chroma_db_path"]).exists()}')
