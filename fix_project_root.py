#!/usr/bin/env python3
"""Fix the PROJECT_ROOT detection in the notebook."""
import json
from pathlib import Path

nb_path = Path(r'c:\Users\wenxu\GenAI-with-LLMs\CrewAI\crewai_agentic_rag_sec.ipynb')

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the imports cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    
    source_str = ''.join(cell['source'])
    
    # Find the imports cell (contains "detect_project_root")
    if 'def detect_project_root' in source_str:
        print(f"Found imports cell at index {i}")
        # Replace the entire cell with the fixed version
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
            '# Look for unique project markers to find the correct root\n',
            'def detect_project_root(start: Path) -> Path:\n',
            '    for p in [start, *start.parents]:\n',
            '        has_config = (p / "config.py").exists()\n',
            '        has_retriever = (p / "shared_retriever.py").exists()\n',
            '        has_sec_data = (p / "sec_rag_team_share").exists()\n',
            '        if has_config and has_retriever and has_sec_data:\n',
            '            return p\n',
            '    return start\n',
            '\n',
            'PROJECT_ROOT = detect_project_root(Path.cwd())\n',
            'print(f"PROJECT_ROOT detected as: {PROJECT_ROOT}")\n',
            '\n',
            'if str(PROJECT_ROOT) not in sys.path:\n',
            '    sys.path.insert(0, str(PROJECT_ROOT))\n',
            '\n',
            'from config import CONFIG as SHARED_CONFIG\n',
            'from shared_retriever import initialize_corpus\n',
            '\n',
            'print("Libraries and shared modules loaded.")',
        ]
        print("✓ Updated imports cell with improved PROJECT_ROOT detection")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"✓ Updated {nb_path}")
