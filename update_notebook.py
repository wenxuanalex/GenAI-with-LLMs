#!/usr/bin/env python3
"""Update CrewAI notebook to use shared retriever."""
import json
from pathlib import Path

nb_path = Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\crewai_agentic_rag_sec.ipynb')

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

updated_count = 0

# Find and update the imports cell (cell 6) to add project root detection
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    
    source_str = ''.join(cell['source'])
    
    # Update imports cell (contains "from crewai import")
    if 'from crewai import' in source_str and 'from pathlib import Path' not in source_str:
        print(f"Found imports cell at index {i}, adding shared_retriever imports...")
        # Replace the entire imports section
        cell['source'] = [
            'import os\n',
            'import re\n',
            'import json\n',
            'import sys\n',
            'import time\n',
            'import warnings\n',
            'from pathlib import Path\n',
            'from dataclasses import dataclass\n',
            'from typing import Any, Dict, List, Optional, Literal, Type\n',
            '\n',
            'import chromadb\n',
            'import numpy as np\n',
            'import pandas as pd\n',
            'from tqdm.auto import tqdm\n',
            'from dotenv import load_dotenv\n',
            '\n',
            'from rank_bm25 import BM25Okapi\n',
            'from sentence_transformers import SentenceTransformer, CrossEncoder\n',
            'from pydantic import BaseModel, Field, field_validator\n',
            '\n',
            'from crewai import Agent, Task, Crew, Process, LLM\n',
            'from crewai.tools import BaseTool\n',
            '\n',
            'warnings.filterwarnings(\'ignore\')\n',
            'load_dotenv()\n',
            '\n',
            '# Resolve project root robustly when notebook runs from subfolder\n',
            'def detect_project_root(start: Path) -> Path:\n',
            '    for p in [start, *start.parents]:\n',
            '        if (p / "config.py").exists() and (p / "shared_retriever.py").exists():\n',
            '            return p\n',
            '    return start\n',
            '\n',
            'PROJECT_ROOT = detect_project_root(Path.cwd())\n',
            'if str(PROJECT_ROOT) not in sys.path:\n',
            '    sys.path.insert(0, str(PROJECT_ROOT))\n',
            '\n',
            'from config import CONFIG as SHARED_CONFIG\n',
            'from shared_retriever import initialize_corpus\n',
            '\n',
            'print(\'Libraries and shared modules loaded.\')',
        ]
        updated_count += 1
        print("✓ Updated imports cell")
    
    # Update CONFIG cell to use SHARED_CONFIG paths
    elif 'sec_chunks_path' in source_str and 'CONFIG: Dict' in source_str:
        print(f"Found CONFIG cell at index {i}, will add override cell...")
        # This is OK - will add override cell below
    
    # Update cell 13 (data loading) - contains "CorpusIndex(chunk_dicts"
    elif 'CorpusIndex(chunk_dicts' in source_str and 'load_sec_chunks' in source_str:
        print(f"Found data loading cell at index {i}")
        cell['source'] = [
            '# ── SEC Corpus Loading ────────────────────────────────────────────────────────────────────────\n',
            '\n',
            'print(\'Initializing CorpusIndex from shared_retriever...\')\n',
            'global_index = initialize_corpus(\n',
            '    chunks_jsonl=CONFIG[\'sec_chunks_path\'],\n',
            '    chroma_db_path=CONFIG[\'chroma_db_path\'],\n',
            ')\n',
            'print(f\'CorpusIndex ready: {len(global_index.df):,} chunks with hybrid retrieval\')',
        ]
        updated_count += 1
        print("✓ Updated data loading cell")
    
    # Update CorpusIndex class definition cell - replace the entire class
    elif 'class CorpusIndex:' in source_str and 'def __init__' in source_str:
        print(f"Found CorpusIndex class at index {i}")
        cell['source'] = [
            '# CorpusIndex is now imported from shared_retriever\n',
            '# (no duplicate needed here)\n',
            'print(\'CorpusIndex imported from shared_retriever.\')',
        ]
        updated_count += 1
        print("✓ Removed duplicate CorpusIndex class")
    
    # Add override cell after CONFIG (find the cell after CONFIG definition)
    if 'CONFIG[\'judge_sample_n\']' in source_str and 'Provider fallback order' not in source_str:
        # This is the old CONFIG ending - need to check if override is needed
        print(f"Found CONFIG ending cell at index {i}")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"✓ Updated {nb_path} ({updated_count} cells modified)")
