"""
Shared Retrieval Module for SEC Agentic RAG
=============================================
Framework-agnostic hybrid retrieval pipeline:
1. BM25 (sparse lexical search)
2. Dense embedding search (Chroma vector store)
3. Reciprocal Rank Fusion (RRF) merge
4. Adjacent chunk expansion (±1 neighbors within same filing)
5. CrossEncoder reranking

This module is designed to be used identically across CrewAI, LlamaIndex, and LangGraph.
"""

import json
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# Import centralized config
from config import CONFIG

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_existing_path(path_str: str) -> Path:
    """Resolve path against direct, cwd-relative, and project-root-relative locations."""
    p = Path(path_str)
    if p.exists():
        return p
    if not p.is_absolute():
        cwd_candidate = (Path.cwd() / p).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        root_candidate = (PROJECT_ROOT / p).resolve()
        if root_candidate.exists():
            return root_candidate
    return p


def _sqlite_header_ok(sqlite_path: Path) -> bool:
    """Return True when the file begins with the standard SQLite header."""
    try:
        with sqlite_path.open('rb') as f:
            return f.read(16) == b'SQLite format 3\x00'
    except OSError:
        return False


def _looks_like_lfs_pointer(file_path: Path) -> bool:
    """Detect common Git LFS pointer files."""
    try:
        with file_path.open('rb') as f:
            head = f.read(64)
        return head.startswith(b'version https://git-lfs.github.com/spec/')
    except OSError:
        return False


# ────────────────────────────────────────────────────────────────────────────
# Data Structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """Standardized retrieved chunk representation across all frameworks."""
    doc_name:         str    # e.g., "AAPL_10-K_2024-11-01"
    company:          str    # e.g., "Apple Inc."
    ticker:           str    # e.g., "AAPL"
    form_type:        str    # e.g., "10-K" or "10-Q"
    filing_year:      int    # e.g., 2024
    page_num:         int    # chunk index within filing
    chunk_id:         str    # unique chunk identifier
    raw_chunk:        str    # original unformatted text
    contextual_chunk: str    # formatted with metadata headers
    score:            float  # retrieval score (0.0-1.0)
    source:           str    # 'bm25' | 'dense' | 'adjacent_expanded' | 'hybrid_reranked'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for framework-agnostic output."""
        return {
            'chunk_id':        self.chunk_id,
            'text':            self.contextual_chunk,
            'raw_text':        self.raw_chunk,
            'score':           self.score,
            'source':          self.source,
            'metadata': {
                'doc_name':    self.doc_name,
                'company':     self.company,
                'ticker':      self.ticker,
                'form_type':   self.form_type,
                'filing_year': self.filing_year,
                'page_num':    self.page_num,
            }
        }


# ────────────────────────────────────────────────────────────────────────────
# Global Model Instances (loaded once at module import)
# ────────────────────────────────────────────────────────────────────────────

def _resolve_trust_remote_code(model_name: str) -> bool:
    """
    Resolve trust_remote_code with a conservative policy.

    - true/false/1/0 in config are honored explicitly.
    - auto enables trust only for allowlisted models known to require it.
    """
    raw_value = str(CONFIG.get('dense_trust_remote_code', 'auto')).strip().lower()
    if raw_value in {'1', 'true', 'yes', 'on'}:
        return True
    if raw_value in {'0', 'false', 'no', 'off'}:
        return False

    allowlist = {
        'nomic-ai/nomic-embed-text-v1.5',
    }
    return model_name in allowlist


def _load_models():
    """Load dense embedding and reranker models."""
    model_name = CONFIG['dense_model_name']
    trust_remote_code = _resolve_trust_remote_code(model_name)
    print(f"Loading embedding model: {model_name}...")
    dense_model = SentenceTransformer(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if trust_remote_code:
        warnings.warn(
            f"trust_remote_code=True for dense model '{model_name}'. "
            "Only enable this for trusted model repositories.",
            RuntimeWarning,
        )
    
    print(f"Loading reranker model: {CONFIG['reranker_model_name']}...")
    reranker = CrossEncoder(CONFIG['reranker_model_name'])
    
    return dense_model, reranker


# Load models globally once
_dense_model, _reranker = _load_models()


# ────────────────────────────────────────────────────────────────────────────
# CorpusIndex: Unified Retrieval Engine
# ────────────────────────────────────────────────────────────────────────────

class CorpusIndex:
    """
    Corpus index combining BM25 and dense retrieval with RRF fusion and reranking.
    Implements adjacent chunk expansion for context enrichment.
    
    This class is framework-agnostic and designed to work with:
    - CrewAI: as a tool backend
    - LlamaIndex: as part of a custom retriever
    - LangGraph: as a node utility
    """

    def __init__(
        self,
        chunks_jsonl: str,
        chroma_db_path: str,
        dense_model_name: str = None,
        reranker_model_name: str = None,
    ):
        """
        Initialize the corpus index.
        
        Args:
            chunks_jsonl: Path to sec_chunks.jsonl file
            chroma_db_path: Path to persisted Chroma DB
            dense_model_name: Override embed model from config
            reranker_model_name: Override reranker model from config
        """
        self.chunks_path = _resolve_existing_path(chunks_jsonl)
        self.chroma_db_path = _resolve_existing_path(chroma_db_path)
        self.chroma_sqlite_path = self.chroma_db_path / 'chroma.sqlite3'
        self.chroma_runtime_path = self.chroma_db_path
        
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks JSONL not found: {self.chunks_path}")
        if not self.chroma_db_path.exists():
            raise FileNotFoundError(f"Chroma DB not found: {self.chroma_db_path}")

        # Load chunks into DataFrame for rapid O(1) lookup
        print(f"Loading chunks from {self.chunks_path}...")
        chunks_data = [json.loads(line) for line in open(self.chunks_path, encoding='utf-8')]
        raw_df = pd.DataFrame(chunks_data)

        # Normalize to canonical schema.
        # Supports both:
        # - {chunk_id, text, metadata}
        # - sec_rag_team_share schema with metadata at top level
        if {'chunk_id', 'text', 'metadata'}.issubset(raw_df.columns):
            self.df = raw_df.copy()
        else:
            if 'chunk_id' not in raw_df.columns or 'text' not in raw_df.columns:
                raise ValueError("Chunks JSONL missing required columns: chunk_id/text")

            def _build_meta(row: pd.Series) -> Dict[str, Any]:
                return {
                    'company_name': row.get('company_name', row.get('company', '')),
                    'ticker': str(row.get('ticker', '')).upper(),
                    'form_type': str(row.get('form_type', '')).upper(),
                    'filing_year': int(row.get('filing_year', 0) or 0),
                    'filing_date': str(row.get('filing_date', ''))[:10],
                    'chunk_index': int(row.get('chunk_index', 0) or 0),
                    'section_title': row.get('section_title', ''),
                }

            self.df = pd.DataFrame({
                'chunk_id': raw_df['chunk_id'].astype(str),
                'text': raw_df['text'].fillna('').astype(str),
                'metadata': raw_df.apply(_build_meta, axis=1),
            })
        
        print(f"Loaded {len(self.df)} chunks")

        # Build contextual chunk column to mirror advanced notebook retrieval behavior.
        self.df['contextual_chunk'] = self.df.apply(
            lambda row: self._contextual_from_meta(
                str(row.get('text', '')),
                row.get('metadata', {}) if isinstance(row.get('metadata', {}), dict) else {},
            ),
            axis=1,
        )

        # Build BM25 index
        print("Building BM25 index...")
        if 'chunk_id_str' not in self.df.columns:
            self.df['chunk_id_str'] = self.df['chunk_id'].astype(str)
        if 'bm25_tokens' not in self.df.columns:
            self.df['bm25_tokens'] = self.df['contextual_chunk'].fillna('').astype(str).str.lower().str.split()
        self.bm25 = BM25Okapi(
            self.df['bm25_tokens'].tolist()
        )

        # Use global models or accept overrides
        self.dense_model = _dense_model
        self.reranker = _reranker

        # Connect to Chroma DB, rebuilding locally if the persisted store is invalid.
        self.chroma_client, self.chroma_collection = self._initialize_chroma()
        # Backward-compatible alias used by older notebook code.
        self.collection = self.chroma_collection
        print(f"Chroma collection: {self.chroma_collection.count()} vectors")

        # Build internal lookup maps for adjacent chunk expansion
        self._build_lookup_maps()
        print("CorpusIndex ready.")

    def _initialize_chroma(self):
        """Open the persisted Chroma collection or rebuild it from chunks if needed."""
        print(f"Connecting to Chroma DB at {self.chroma_db_path}...")
        needs_rebuild = False

        if not self.chroma_sqlite_path.exists():
            print(f"[Chroma] Missing {self.chroma_sqlite_path.name}; rebuilding local vector store.")
            needs_rebuild = True
        elif _looks_like_lfs_pointer(self.chroma_sqlite_path):
            print(
                "[Chroma] Detected a Git LFS pointer instead of a real SQLite database. "
                "Rebuilding local Chroma store from sec_chunks.jsonl."
            )
            needs_rebuild = True
        elif not _sqlite_header_ok(self.chroma_sqlite_path):
            print(
                f"[Chroma] Invalid SQLite header in {self.chroma_sqlite_path.name}. "
                "Rebuilding local Chroma store from sec_chunks.jsonl."
            )
            needs_rebuild = True

        if needs_rebuild:
            self.chroma_runtime_path = self._local_rebuild_path()
            return self._rebuild_chroma_store()

        try:
            client = chromadb.PersistentClient(path=str(self.chroma_runtime_path))
            collections = client.list_collections()
            if collections:
                collection = client.get_collection(collections[0].name)
            else:
                print("[Chroma] No collections found; rebuilding local vector store.")
                self.chroma_runtime_path = self._local_rebuild_path()
                return self._rebuild_chroma_store()
            if collection.count() == 0:
                print("[Chroma] Collection is empty; rebuilding local vector store.")
                self.chroma_runtime_path = self._local_rebuild_path()
                return self._rebuild_chroma_store()
            return client, collection
        except Exception as exc:
            print(f"[Chroma] Failed to open persisted store ({exc}). Rebuilding local vector store.")
            self.chroma_runtime_path = self._local_rebuild_path()
            return self._rebuild_chroma_store()

    def _local_rebuild_path(self) -> Path:
        """Return a clean local path for rebuilt Chroma artifacts."""
        return (PROJECT_ROOT / ".cache" / "rebuilt_chroma_db").resolve()

    def _rebuild_chroma_store(self):
        """Recreate the Chroma DB from chunk text and locally computed embeddings."""
        rebuild_path = self.chroma_runtime_path
        print(f"[Chroma] Rebuild target: {rebuild_path}")
        if rebuild_path.exists():
            shutil.rmtree(rebuild_path, ignore_errors=True)
        rebuild_path.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(rebuild_path))
        collection = client.get_or_create_collection(name="sec_filings")

        batch_size = 128
        total = len(self.df)
        print(f"[Chroma] Rebuilding vector store with {total:,} chunks...")

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = self.df.iloc[start:end]
            texts = batch['contextual_chunk'].fillna('').astype(str).tolist()
            embeddings = self.dense_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            metadatas = [self._normalize_chroma_metadata(meta) for meta in batch['metadata'].tolist()]
            ids = batch['chunk_id'].astype(str).tolist()
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            if start == 0 or end == total or (start // batch_size) % 10 == 0:
                print(f"[Chroma] Indexed {end:,}/{total:,} chunks")

        return client, collection

    def _normalize_chroma_metadata(self, meta: Any) -> Dict[str, Any]:
        """Convert metadata to Chroma-safe primitive values."""
        if not isinstance(meta, dict):
            return {}

        normalized = {}
        for key, value in meta.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                normalized[key] = value
            else:
                normalized[key] = str(value)
        return normalized

    @staticmethod
    def _normalize_section_label(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")

    def _parse_section_style_id(self, doc_id: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Parse Chroma ids like:
        NVDA_10-K_2024-02-21_Item_1A_-_Risk_Factors
        NVDA_10-Q_2023-11-21_Item_1A_-_Risk_Factors_(10-Q)
        """
        parts = str(doc_id or "").split("_")
        if len(parts) < 4:
            return None
        ticker = parts[0].upper()
        form_type = parts[1].upper()
        filing_date = parts[2]
        section_raw = "_".join(parts[3:])
        section_norm = self._normalize_section_label(section_raw)
        if not ticker or not form_type or not filing_date or not section_norm:
            return None
        return (ticker, form_type, filing_date, section_norm)

    def _build_lookup_maps(self):
        """Build internal maps for O(1) adjacent chunk lookup."""
        # Map: chunk_id -> row index
        self._str_to_row = {str(cid): idx for idx, cid in enumerate(self.df['chunk_id'])}
        # Fallback map for dense hits when Chroma ids do not match chunk_id exactly.
        self._contextual_to_rows = {}
        for idx, text in enumerate(self.df['contextual_chunk'].fillna('').astype(str)):
            self._contextual_to_rows.setdefault(text, []).append(idx)
        # Map section-style ids back to chunk rows using filing metadata.
        self._sectionkey_to_rows = {}
        
        # Map: (ticker, form_type, filing_date) -> chunk_index -> row_index
        # Allows fast adjacent chunk lookup within same filing
        self._filing_chunk_lookup = {}
        for idx, row in self.df.iterrows():
            meta = row.get('metadata', {})
            if not isinstance(meta, dict):
                try:
                    meta = json.loads(meta) if isinstance(meta, str) else {}
                except:
                    meta = {}
            
            ticker = meta.get('ticker', '')
            form_type = meta.get('form_type', '')
            filing_date = meta.get('filing_date', '')
            chunk_index = int(meta.get('chunk_index', 0))
            section_norm = self._normalize_section_label(meta.get('section_title', ''))
            
            filing_key = (ticker, form_type, filing_date)
            if filing_key not in self._filing_chunk_lookup:
                self._filing_chunk_lookup[filing_key] = {}
            self._filing_chunk_lookup[filing_key][chunk_index] = idx
            if ticker and form_type and filing_date and section_norm:
                section_key = (str(ticker).upper(), str(form_type).upper(), str(filing_date), section_norm)
                self._sectionkey_to_rows.setdefault(section_key, []).append(idx)

    def _chunk_from_row(
        self, row_idx: int, score: float, source: str
    ) -> RetrievedChunk:
        """Construct a RetrievedChunk from a dataframe row."""
        row = self.df.iloc[row_idx]
        meta = row.get('metadata', {})
        if not isinstance(meta, dict):
            try:
                meta = json.loads(meta) if isinstance(meta, str) else {}
            except:
                meta = {}
        
        raw_text = str(row.get('text', ''))
        contextual_text = self._contextual_from_meta(raw_text, meta)
        
        doc_name = f"{meta.get('ticker', '')}_{meta.get('form_type', '')}_{str(meta.get('filing_date', ''))[:10]}"
        
        return RetrievedChunk(
            doc_name=doc_name,
            company=meta.get('company_name', ''),
            ticker=meta.get('ticker', ''),
            form_type=meta.get('form_type', ''),
            filing_year=int(meta.get('filing_year', 0)),
            page_num=int(meta.get('chunk_index', 0)),
            chunk_id=str(row.get('chunk_id', '')),
            raw_chunk=raw_text,
            contextual_chunk=contextual_text,
            score=float(score),
            source=source,
        )

    @staticmethod
    def _contextual_from_meta(text: str, meta: dict) -> str:
        """Format chunk with metadata headers for context."""
        return (
            f"Company: {meta.get('company_name', '')} ({meta.get('ticker', '')})\n"
            f"Filing: {meta.get('form_type', '')} | Date: {str(meta.get('filing_date', ''))[:10]} "
            f"| Year: {meta.get('filing_year', '')}\n"
            f"Section: {meta.get('section_title', '')}\n"
            f"Content: {text}"
        )

    def _chroma_where(
        self, ticker: str = None, filing_year: int = None, form_type: str = None
    ) -> Optional[Dict]:
        """Build Chroma metadata filter."""
        conditions = []
        if ticker:
            conditions.append({"ticker": {"$eq": str(ticker).upper()}})
        if filing_year:
            conditions.append({"filing_year": {"$eq": int(filing_year)}})
        if form_type:
            conditions.append({"form_type": {"$eq": str(form_type).upper()}})
        
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _meta_matches_filters(
        self, meta: Dict[str, Any], ticker: str = None, filing_year: int = None, form_type: str = None
    ) -> bool:
        """Apply the same filter semantics used by BM25 to Chroma metadata."""
        if not isinstance(meta, dict):
            return False
        if ticker and str(meta.get('ticker', '')).upper() != str(ticker).upper():
            return False
        if filing_year and int(meta.get('filing_year', 0) or 0) != int(filing_year):
            return False
        if form_type and str(meta.get('form_type', '')).upper() != str(form_type).upper():
            return False
        return True

    def _bm25_mask(
        self, ticker: str = None, filing_year: int = None, form_type: str = None
    ) -> Optional[np.ndarray]:
        """Build a BM25 row mask compatible with legacy notebook call sites."""
        if not ticker and not filing_year and not form_type:
            return None

        mask = np.ones(len(self.df), dtype=float)
        for idx, row in self.df.iterrows():
            meta = row.get('metadata', {})
            if not isinstance(meta, dict):
                try:
                    meta = json.loads(meta) if isinstance(meta, str) else {}
                except Exception:
                    meta = {}

            if ticker and str(meta.get('ticker', '')).upper() != str(ticker).upper():
                mask[idx] = 0.0
                continue
            if filing_year and int(meta.get('filing_year', 0) or 0) != int(filing_year):
                mask[idx] = 0.0
                continue
            if form_type and str(meta.get('form_type', '')).upper() != str(form_type).upper():
                mask[idx] = 0.0

        return mask if mask.sum() > 0 else None

    def bm25_search(
        self, query: str, top_k: int, mask: Optional[np.ndarray] = None
    ) -> List[RetrievedChunk]:
        """BM25 sparse lexical search."""
        scores = np.array(self.bm25.get_scores(query.lower().split()))
        if mask is not None:
            scores = scores * mask
        top_idx = np.argsort(scores)[::-1]
        top_idx = [i for i in top_idx if scores[i] > 0][:top_k]
        return [self._chunk_from_row(i, scores[i], 'bm25') for i in top_idx]

    def dense_search(
        self,
        query: str,
        top_k: int,
        ticker: str = None,
        filing_year: int = None,
        form_type: str = None,
    ) -> List[RetrievedChunk]:
        """Dense embedding search via Chroma."""
        # Use query prefix matching Nomic embedding expectations
        query_prefix = str(CONFIG.get('dense_query_prefix', 'search_query: '))
        query_with_prefix = f"{query_prefix}{query}" if query_prefix else query
        q_emb = self.dense_model.encode([query_with_prefix], normalize_embeddings=True)[0].tolist()
        
        where = self._chroma_where(ticker, filing_year, form_type)
        results = self.chroma_collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where,
            include=['distances', 'metadatas', 'documents']
        )
        raw_filtered_hits = len(results['ids'][0]) if results.get('ids') and results['ids'] else 0
        broad_hits = 0
        locally_filtered_hits = raw_filtered_hits

        fallback_mode = 'strict'
        if not results['ids'] or not results['ids'][0]:
            # If strict Chroma metadata filtering misses, retry broadly and apply local filter logic.
            retry_n = max(top_k * 4, top_k)
            broad_results = self.chroma_collection.query(
                query_embeddings=[q_emb],
                n_results=retry_n,
                include=['distances', 'metadatas', 'documents']
            )
            broad_hits = len(broad_results['ids'][0]) if broad_results.get('ids') and broad_results['ids'] else 0
            filtered_ids = []
            filtered_docs = []
            filtered_dists = []
            filtered_metas = []
            fallback_filters = [
                ('strict_local', dict(ticker=ticker, filing_year=filing_year, form_type=form_type)),
                # Fiscal-year questions often need filing-year relaxation while keeping company/form constraints.
                ('relax_year', dict(ticker=ticker, filing_year=None, form_type=form_type)),
                ('ticker_only', dict(ticker=ticker, filing_year=None, form_type=None)),
                ('unfiltered', dict(ticker=None, filing_year=None, form_type=None)),
            ]
            for mode_name, mode_filters in fallback_filters:
                filtered_ids = []
                filtered_docs = []
                filtered_dists = []
                filtered_metas = []
                for doc_id, text, dist, meta in zip(
                    broad_results['ids'][0],
                    broad_results['documents'][0],
                    broad_results['distances'][0],
                    broad_results['metadatas'][0],
                ):
                    if self._meta_matches_filters(meta, **mode_filters):
                        filtered_ids.append(doc_id)
                        filtered_docs.append(text)
                        filtered_dists.append(dist)
                        filtered_metas.append(meta)
                    if len(filtered_ids) >= top_k:
                        break
                if filtered_ids:
                    fallback_mode = mode_name
                    break
            locally_filtered_hits = len(filtered_ids)
            results = {
                'ids': [filtered_ids],
                'documents': [filtered_docs],
                'distances': [filtered_dists],
                'metadatas': [filtered_metas],
            }

        chunks = []
        unresolved_ids = []
        for doc_id, text, dist, meta in zip(
            results['ids'][0],
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0],
        ):
            # Find row index from chunk_id
            doc_id_str = str(doc_id)
            row_idx = self._str_to_row.get(doc_id_str)
            if row_idx is None and isinstance(meta, dict):
                for key in ('chunk_id', 'id', 'doc_id'):
                    meta_id = meta.get(key)
                    if meta_id is not None:
                        row_idx = self._str_to_row.get(str(meta_id))
                        if row_idx is not None:
                            break
            if row_idx is None and text is not None:
                candidate_rows = self._contextual_to_rows.get(str(text), [])
                if len(candidate_rows) == 1:
                    row_idx = candidate_rows[0]
                elif candidate_rows:
                    for candidate_idx in candidate_rows:
                        candidate_meta = self.df.iloc[candidate_idx].get('metadata', {})
                        if self._meta_matches_filters(
                            candidate_meta,
                            ticker=ticker,
                            filing_year=filing_year,
                            form_type=form_type,
                        ):
                            row_idx = candidate_idx
                            break
            if row_idx is None:
                section_key = self._parse_section_style_id(doc_id_str)
                if section_key is not None:
                    candidate_rows = self._sectionkey_to_rows.get(section_key, [])
                    if len(candidate_rows) == 1:
                        row_idx = candidate_rows[0]
                    elif candidate_rows:
                        # Prefer the earliest chunk within the matched section.
                        row_idx = min(candidate_rows, key=lambda i: int(self.df.iloc[i].get('metadata', {}).get('chunk_index', 0) or 0))
            if row_idx is not None:
                chunk = self._chunk_from_row(row_idx, 1.0 - float(dist), 'dense')
                chunks.append(chunk)
            else:
                unresolved_ids.append(doc_id_str)

        debug_enabled = bool(CONFIG.get('dense_debug_logging', True))
        if debug_enabled:
            print(
                "    [DenseDebug] "
                f"query={query[:80]!r} where={where} "
                f"filtered_hits={raw_filtered_hits} broad_hits={broad_hits} "
                f"fallback_mode={fallback_mode} local_filtered_hits={locally_filtered_hits} returned={len(chunks)} "
                f"unresolved_sample={unresolved_ids[:3]}"
            )
        
        return chunks

    def _expand_adjacent(
        self, pool: Dict[str, RetrievedChunk], expand_n: int = 1
    ) -> List[RetrievedChunk]:
        """
        Expand pool with adjacent chunks (±expand_n) within same filing.
        Returns new chunks not already in pool.
        """
        extra = {}
        existing_ids = set(pool.keys())
        
        for chunk in pool.values():
            # Find filing key
            ticker = chunk.ticker
            form_type = chunk.form_type
            filing_date_str = chunk.doc_name.split('_')[-1]  # e.g., "2024-11-01"
            
            filing_key = (ticker, form_type, filing_date_str)
            if filing_key not in self._filing_chunk_lookup:
                continue
            
            ci_map = self._filing_chunk_lookup[filing_key]
            base_ci = chunk.page_num
            
            # Look for neighbors
            for delta in range(-expand_n, expand_n + 1):
                if delta == 0:
                    continue
                adj_row_idx = ci_map.get(base_ci + delta)
                if adj_row_idx is None:
                    continue
                adj_id = str(self.df.iloc[adj_row_idx]['chunk_id'])
                if adj_id in existing_ids or adj_id in extra:
                    continue
                extra[adj_id] = self._chunk_from_row(adj_row_idx, 0.0, 'adjacent_expanded')
        
        return list(extra.values())

    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        bm25_top_k: int = None,
        dense_top_k: int = None,
        rerank_top_k: int = None,
        embed_model: Any = None,
        reranker: Any = None,
        ticker: str = None,
        filing_year: int = None,
        form_type: str = None,
        expand_n: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute full hybrid retrieval pipeline:
        1. BM25 sparse search
        2. Dense embedding search
        3. Reciprocal Rank Fusion merge
        4. Adjacent chunk expansion (contextual enrichment)
        5. CrossEncoder reranking
        
        Returns standardized list of dicts with text, metadata, score, source.
        """
        # Use config defaults if not provided
        legacy_mode = any(
            x is not None for x in [bm25_top_k, dense_top_k, rerank_top_k, embed_model, reranker]
        )

        bm25_top_k = bm25_top_k or CONFIG['bm25_top_k']
        dense_top_k = dense_top_k or CONFIG['dense_top_k']
        rerank_top_k = rerank_top_k or top_k or CONFIG['rerank_top_k']
        expand_n = expand_n if expand_n is not None else CONFIG.get('adjacent_chunk_expansion_n', 1)

        if embed_model is not None:
            self.dense_model = embed_model
        if reranker is not None:
            self.reranker = reranker

        # 1. BM25 Search
        bm25_mask = self._bm25_mask(ticker=ticker, filing_year=filing_year, form_type=form_type)
        bm25_results = self.bm25_search(query, top_k=bm25_top_k, mask=bm25_mask)
        
        # 2. Dense Search
        dense_results = self.dense_search(
            query, top_k=dense_top_k, 
            ticker=ticker, filing_year=filing_year, form_type=form_type
        )
        
        # 3. RRF Merge
        rrf_k = int(CONFIG.get('rrf_k', 60))
        def _rrf_score(rank: int, k: int = rrf_k) -> float:
            return 1.0 / (k + rank)
        
        pool: Dict[str, RetrievedChunk] = {}
        rrf: Dict[str, float] = {}
        
        for rank, chunk in enumerate(bm25_results):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + _rrf_score(rank)
            if chunk.chunk_id not in pool:
                pool[chunk.chunk_id] = chunk
        
        for rank, chunk in enumerate(dense_results):
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + _rrf_score(rank)
            if chunk.chunk_id not in pool:
                pool[chunk.chunk_id] = chunk
        
        # 4. Adjacent Chunk Expansion
        for adj_chunk in self._expand_adjacent(pool, expand_n=expand_n):
            pool[adj_chunk.chunk_id] = adj_chunk
            rrf[adj_chunk.chunk_id] = 0.0
        
        if not pool:
            return []
        
        # 5. CrossEncoder Reranking
        candidates = [pool[k] for k in sorted(rrf, key=rrf.__getitem__, reverse=True)]
        scores = self.reranker.predict(
            [(query, chunk.contextual_chunk) for chunk in candidates],
            show_progress_bar=False,
        )
        
        for chunk, score in zip(candidates, scores):
            chunk.score = float(score)
            chunk.source = 'hybrid_reranked'
        
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:rerank_top_k]
        
        # Log retrieval stats
        n_unique_from_bm25_dense = len(
            set(c.chunk_id for c in bm25_results) | set(c.chunk_id for c in dense_results)
        )
        n_overlap = len(
            set(c.chunk_id for c in bm25_results) & set(c.chunk_id for c in dense_results)
        )
        n_adjacent = len(candidates) - n_unique_from_bm25_dense + n_overlap
        
        print(
            f"    [Retrieval] pool={len(candidates)} (bm25={len(bm25_results)} "
            f"dense={len(dense_results)} adj={n_adjacent}) → rerank top-{rerank_top_k}"
        )
        
        if legacy_mode:
            return ranked

        # Convert to standardized output format
        return [chunk.to_dict() for chunk in ranked]


# ────────────────────────────────────────────────────────────────────────────
# Module-level convenience functions
# ────────────────────────────────────────────────────────────────────────────

_corpus_index: Optional[CorpusIndex] = None


def initialize_corpus(
    chunks_jsonl: str = None,
    chroma_db_path: str = None,
) -> CorpusIndex:
    """Initialize the global corpus index."""
    global _corpus_index
    
    chunks_path = chunks_jsonl or CONFIG['sec_chunks_path']
    chroma_path = chroma_db_path or CONFIG['chroma_db_path']
    
    _corpus_index = CorpusIndex(
        chunks_jsonl=chunks_path,
        chroma_db_path=chroma_path,
    )
    return _corpus_index


def get_corpus() -> CorpusIndex:
    """Get the global corpus index (lazily initialize if needed)."""
    global _corpus_index
    if _corpus_index is None:
        initialize_corpus()
    return _corpus_index


def hybrid_search(
    query: str,
    top_k: int = None,
    ticker: str = None,
    filing_year: int = None,
    form_type: str = None,
) -> List[Dict[str, Any]]:
    """
    Public API for hybrid search. Initializes corpus on first call.
    
    Returns list of dicts with standardized schema:
    {
        'chunk_id': str,
        'text': str,  # contextual chunk with headers
        'raw_text': str,
        'score': float,
        'source': str,
        'metadata': dict  # {ticker, company, form_type, filing_year, page_num, ...}
    }
    """
    corpus = get_corpus()
    return corpus.hybrid_search(query, top_k=top_k, ticker=ticker, filing_year=filing_year, form_type=form_type)


if __name__ == '__main__':
    # Test initialization
    print("Testing shared_retriever module...")
    corpus = initialize_corpus()
    
    # Example query
    sample_query = "What were Apple's net revenues?"
    print(f"\nTest query: {sample_query}")
    results = corpus.hybrid_search(sample_query, top_k=3, ticker="AAPL")
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['metadata']['doc_name']}")
        print(f"   Score: {result['score']:.3f} | Source: {result['source']}")
        print(f"   Text: {result['text'][:200]}...")
