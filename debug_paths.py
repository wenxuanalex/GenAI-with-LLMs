from pathlib import Path
import sys
sys.path.insert(0, r'c:\Users\wenxu\GenAI-with-LLMs')

# Check what exists where
print('Checking directory structure:')
proj_root = Path(r'c:\Users\wenxu\GenAI-with-LLMs')
user_root = Path(r'c:\Users\wenxu')

print(f'  {proj_root} exists: {proj_root.exists()}')
print(f'  {proj_root / "sec_rag_team_share"} exists: {(proj_root / "sec_rag_team_share").exists()}')
print(f'  {user_root / "sec_rag_team_share"} exists: {(user_root / "sec_rag_team_share").exists()}')

# Now check the config paths
from config import CONFIG
print()
print('SHARED_CONFIG paths:')
print(f'  sec_chunks_path: {CONFIG["sec_chunks_path"]}')
print(f'  chroma_db_path: {CONFIG["chroma_db_path"]}')
print()
print('Checking if paths exist:')
print(f'  {CONFIG["sec_chunks_path"]} exists: {Path(CONFIG["sec_chunks_path"]).exists()}')
print(f'  {CONFIG["chroma_db_path"]} exists: {Path(CONFIG["chroma_db_path"]).exists()}')
