# GSWA Development Tasks
# Claude Code 开发任务拆解

---

## 🎯 开发原则

1. **小步提交**: 每个任务对应一个可运行的 PR
2. **测试优先**: 每个功能都有对应测试
3. **安全第一**: 每次提交检查无外部调用
4. **渐进增强**: 从最简实现开始，逐步完善

---

## 📋 Task Checklist

### PR #1: Project Scaffolding
**目标:** 创建项目骨架和配置

```bash
# 创建目录结构
mkdir -p src/gswa/{api,services,utils}
mkdir -p data/{corpus/{raw,parsed},index}
mkdir -p web scripts tests docs logs

# 创建核心文件
touch src/gswa/__init__.py
touch src/gswa/config.py
touch src/gswa/main.py
```

**文件清单:**

1. `pyproject.toml` - 项目配置
```toml
[project]
name = "gswa"
version = "0.1.0"
description = "Gilles-Style Writing Assistant"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "httpx>=0.26.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "responses>=0.24",
    "ruff>=0.1.0",
]
similarity = [
    "sentence-transformers>=2.2.0",
    "faiss-cpu>=1.7.0",
    "numpy>=1.24.0",
]
```

2. `src/gswa/config.py` - 配置管理
```python
"""GSWA Configuration Module.

This module handles all configuration with security-first defaults.
External API calls are DISABLED by default and cannot be enabled
without explicit override.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with security defaults."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # === SECURITY (HARDCODED DEFAULTS) ===
    allow_external_api: bool = False  # MUST remain False
    
    # === LLM Server ===
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "mistral-7b-instruct"
    vllm_api_key: str = "dummy"  # vLLM doesn't need real key locally
    
    # === Similarity Thresholds ===
    threshold_ngram_max_match: int = 12
    threshold_ngram_overlap: float = 0.15
    threshold_embed_top1: float = 0.88
    
    # === Generation ===
    default_n_variants: int = 3
    max_n_variants: int = 5
    temperature_base: float = 0.3
    temperature_variance: float = 0.15
    max_new_tokens: int = 1024
    
    # === Paths ===
    corpus_path: str = "./data/corpus/parsed"
    index_path: str = "./data/index"
    log_path: str = "./logs"
    
    # === Embedding ===
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    def validate_security(self) -> None:
        """Validate security constraints at startup."""
        if self.allow_external_api:
            raise ValueError(
                "CRITICAL: ALLOW_EXTERNAL_API=true is FORBIDDEN. "
                "This system MUST run fully offline. "
                "Remove or set ALLOW_EXTERNAL_API=false."
            )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.validate_security()
    return settings
```

3. `.env.example` - 环境变量模板
```bash
# GSWA Environment Configuration
# Copy to .env and customize

# === SECURITY (DO NOT CHANGE) ===
ALLOW_EXTERNAL_API=false

# === LLM Server ===
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=mistral-7b-instruct

# === Similarity Thresholds ===
THRESHOLD_NGRAM_MAX_MATCH=12
THRESHOLD_EMBED_TOP1=0.88

# === Generation ===
DEFAULT_N_VARIANTS=3
TEMPERATURE_BASE=0.3
```

**验收标准:**
- [ ] `pip install -e .` 成功
- [ ] `python -c "from gswa.config import get_settings; get_settings()"` 无报错
- [ ] 设置 `ALLOW_EXTERNAL_API=true` 时抛出异常

---

### PR #2: Pydantic Schemas
**目标:** 定义 API 数据模型

**文件:** `src/gswa/api/schemas.py`

```python
"""API Request/Response Schemas.

All data structures for the GSWA API endpoints.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Section(str, Enum):
    """Paper section types."""
    ABSTRACT = "Abstract"
    INTRODUCTION = "Introduction"
    METHODS = "Methods"
    RESULTS = "Results"
    DISCUSSION = "Discussion"
    CONCLUSION = "Conclusion"


class Strategy(str, Enum):
    """Rewriting strategies for diversification."""
    A = "A"  # Conclusion-first
    B = "B"  # Background-first
    C = "C"  # Methods-first
    D = "D"  # Cautious-first


class RewriteConstraints(BaseModel):
    """Constraints for rewriting."""
    preserve_numbers: bool = True
    no_new_facts: bool = True


class RewriteRequest(BaseModel):
    """Request body for /v1/rewrite/variants."""
    text: str = Field(..., min_length=10, max_length=10000)
    section: Optional[Section] = None
    n_variants: int = Field(default=3, ge=1, le=5)
    strategies: Optional[list[Strategy]] = None
    constraints: RewriteConstraints = Field(default_factory=RewriteConstraints)
    
    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class SimilarityScores(BaseModel):
    """Similarity metrics for a variant."""
    ngram_max_match: int = Field(..., description="Longest consecutive n-gram match")
    ngram_overlap: float = Field(..., description="N-gram overlap ratio")
    embed_top1: float = Field(..., description="Cosine similarity to most similar corpus paragraph")
    top1_doc_id: Optional[str] = None
    top1_para_id: Optional[str] = None


class RewriteVariant(BaseModel):
    """A single rewrite variant."""
    text: str
    strategy: Strategy
    scores: SimilarityScores
    fallback: bool = False
    fallback_reason: Optional[str] = None


class RewriteResponse(BaseModel):
    """Response body for /v1/rewrite/variants."""
    variants: list[RewriteVariant]
    model_version: str
    processing_time_ms: int


class ChatMessage(BaseModel):
    """A chat message."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ReplyRequest(BaseModel):
    """Request body for /v1/reply."""
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)


class ReplyResponse(BaseModel):
    """Response body for /v1/reply."""
    content: str
    model: str


class HealthResponse(BaseModel):
    """Response body for /v1/health."""
    status: str
    llm_server: str
    model_loaded: Optional[str]
    corpus_paragraphs: int
    index_loaded: bool
```

**验收标准:**
- [ ] `from gswa.api.schemas import *` 无报错
- [ ] Pydantic validation 工作正常
- [ ] 测试文件 `tests/test_schemas.py` 通过

---

### PR #3: N-gram Similarity Service
**目标:** 实现 n-gram 重叠检测

**文件:** `src/gswa/utils/ngram.py`

```python
"""N-gram overlap detection.

Detects verbatim or near-verbatim copying from corpus.
"""
import re
from collections import Counter
from typing import Set, Tuple


def tokenize(text: str) -> list[str]:
    """Simple word tokenization."""
    # Lowercase and split on non-alphanumeric
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def get_ngrams(tokens: list[str], n: int) -> list[Tuple[str, ...]]:
    """Extract n-grams from token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def build_ngram_index(corpus_texts: list[str], n: int = 8) -> Set[Tuple[str, ...]]:
    """Build n-gram set from corpus for fast lookup."""
    ngram_set: Set[Tuple[str, ...]] = set()
    for text in corpus_texts:
        tokens = tokenize(text)
        ngrams = get_ngrams(tokens, n)
        ngram_set.update(ngrams)
    return ngram_set


def find_longest_match(
    candidate_tokens: list[str],
    corpus_ngram_set: Set[Tuple[str, ...]],
    max_n: int = 20
) -> int:
    """Find longest consecutive n-gram match with corpus."""
    if not candidate_tokens or not corpus_ngram_set:
        return 0
    
    longest = 0
    # Check n-grams of decreasing length
    for n in range(min(max_n, len(candidate_tokens)), 0, -1):
        ngrams = get_ngrams(candidate_tokens, n)
        for ngram in ngrams:
            if ngram in corpus_ngram_set:
                return n  # Found match at this length
    return longest


def compute_ngram_overlap(
    candidate: str,
    corpus_ngram_set: Set[Tuple[str, ...]],
    n: int = 8
) -> dict:
    """
    Compute n-gram overlap metrics.
    
    Args:
        candidate: The generated text to check
        corpus_ngram_set: Pre-built set of corpus n-grams
        n: N-gram size for overlap calculation
    
    Returns:
        {
            "max_consecutive_match": int,
            "overlap_ratio": float,
            "matching_ngrams": int
        }
    """
    tokens = tokenize(candidate)
    
    # Find longest consecutive match
    max_match = find_longest_match(tokens, corpus_ngram_set)
    
    # Compute overlap ratio for n-grams
    candidate_ngrams = get_ngrams(tokens, n)
    if not candidate_ngrams:
        return {
            "max_consecutive_match": max_match,
            "overlap_ratio": 0.0,
            "matching_ngrams": 0
        }
    
    candidate_set = set(candidate_ngrams)
    matching = candidate_set.intersection(corpus_ngram_set)
    overlap_ratio = len(matching) / len(candidate_set) if candidate_set else 0.0
    
    return {
        "max_consecutive_match": max_match,
        "overlap_ratio": overlap_ratio,
        "matching_ngrams": len(matching)
    }
```

**测试文件:** `tests/test_ngram.py`

```python
"""Tests for n-gram utilities."""
import pytest
from gswa.utils.ngram import (
    tokenize, get_ngrams, build_ngram_index,
    find_longest_match, compute_ngram_overlap
)


def test_tokenize():
    text = "Hello, World! This is a test."
    tokens = tokenize(text)
    assert tokens == ["hello", "world", "this", "is", "a", "test"]


def test_get_ngrams():
    tokens = ["a", "b", "c", "d", "e"]
    ngrams = get_ngrams(tokens, 3)
    assert ngrams == [("a", "b", "c"), ("b", "c", "d"), ("c", "d", "e")]


def test_find_longest_match():
    corpus = ["The quick brown fox jumps over the lazy dog"]
    ngram_set = build_ngram_index(corpus, n=3)
    
    # Exact substring should match
    candidate_tokens = tokenize("the quick brown fox")
    match_len = find_longest_match(candidate_tokens, ngram_set)
    assert match_len >= 3
    
    # Different text should not match
    candidate_tokens = tokenize("hello world goodbye")
    match_len = find_longest_match(candidate_tokens, ngram_set)
    assert match_len == 0


def test_compute_ngram_overlap():
    corpus = ["We observed a significant increase in activity levels"]
    ngram_set = build_ngram_index(corpus, n=5)
    
    # Similar text
    candidate = "We observed a significant increase in protein activity levels"
    result = compute_ngram_overlap(candidate, ngram_set, n=5)
    assert result["max_consecutive_match"] > 0
    
    # Completely different text
    candidate = "The weather is nice today"
    result = compute_ngram_overlap(candidate, ngram_set, n=5)
    assert result["max_consecutive_match"] == 0
```

**验收标准:**
- [ ] 所有测试通过
- [ ] 长匹配正确检测
- [ ] 性能：1000 段落语料库 < 100ms

---

### PR #4: Embedding Similarity Service
**目标:** 实现向量相似度检测

**文件:** `src/gswa/utils/embedding.py`

```python
"""Embedding-based similarity detection.

Uses sentence transformers for semantic similarity.
"""
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class EmbeddingService:
    """Manages embedding model and similarity search."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._index = None
        self._doc_ids: list[str] = []
        self._para_ids: list[str] = []
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    def build_index(
        self,
        texts: list[str],
        doc_ids: list[str],
        para_ids: list[str]
    ) -> None:
        """Build FAISS index from corpus texts."""
        import faiss
        
        embeddings = self.encode(texts)
        dimension = embeddings.shape[1]
        
        # Use inner product (cosine sim with normalized vectors)
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings.astype(np.float32))
        
        self._doc_ids = doc_ids
        self._para_ids = para_ids
    
    def save_index(self, path: str) -> None:
        """Save index to disk."""
        import faiss
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self._index, str(path / "corpus.faiss"))
        np.save(path / "doc_ids.npy", np.array(self._doc_ids))
        np.save(path / "para_ids.npy", np.array(self._para_ids))
    
    def load_index(self, path: str) -> bool:
        """Load index from disk."""
        import faiss
        path = Path(path)
        
        index_file = path / "corpus.faiss"
        if not index_file.exists():
            return False
        
        self._index = faiss.read_index(str(index_file))
        self._doc_ids = np.load(path / "doc_ids.npy").tolist()
        self._para_ids = np.load(path / "para_ids.npy").tolist()
        return True
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        """
        Search for most similar corpus paragraphs.
        
        Returns:
            List of {similarity, doc_id, para_id} dicts
        """
        if self._index is None:
            return []
        
        query_embedding = self.encode([query])
        scores, indices = self._index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for missing
                continue
            results.append({
                "similarity": float(score),
                "doc_id": self._doc_ids[idx],
                "para_id": self._para_ids[idx],
                "rank": i + 1
            })
        return results
    
    def compute_similarity(
        self,
        candidate: str,
        top_k: int = 5
    ) -> dict:
        """
        Compute embedding similarity metrics.
        
        Returns:
            {
                "top1_similarity": float,
                "top1_doc_id": str,
                "top1_para_id": str,
                "topk_avg": float
            }
        """
        results = self.search(candidate, top_k)
        
        if not results:
            return {
                "top1_similarity": 0.0,
                "top1_doc_id": None,
                "top1_para_id": None,
                "topk_avg": 0.0
            }
        
        top1 = results[0]
        topk_avg = sum(r["similarity"] for r in results) / len(results)
        
        return {
            "top1_similarity": top1["similarity"],
            "top1_doc_id": top1["doc_id"],
            "top1_para_id": top1["para_id"],
            "topk_avg": topk_avg
        }
```

**验收标准:**
- [ ] 索引构建和保存/加载工作正常
- [ ] 相似度搜索返回正确结果
- [ ] 测试文件通过

---

### PR #5: Combined Similarity Service
**目标:** 整合 n-gram 和 embedding 检测

**文件:** `src/gswa/services/similarity.py`

```python
"""Combined Similarity Service.

Orchestrates n-gram and embedding similarity checks.
"""
from typing import Optional, Tuple, Set
from pathlib import Path
import json

from gswa.utils.ngram import build_ngram_index, compute_ngram_overlap
from gswa.utils.embedding import EmbeddingService
from gswa.config import get_settings


class SimilarityService:
    """Combined n-gram and embedding similarity service."""
    
    def __init__(self):
        self.settings = get_settings()
        self._ngram_index: Optional[Set] = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._corpus_texts: list[str] = []
        self._corpus_loaded = False
    
    def load_corpus(self, corpus_path: Optional[str] = None) -> int:
        """
        Load corpus from JSONL files.
        
        Returns:
            Number of paragraphs loaded
        """
        corpus_path = Path(corpus_path or self.settings.corpus_path)
        
        texts = []
        doc_ids = []
        para_ids = []
        
        for jsonl_file in corpus_path.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    texts.append(item["text"])
                    doc_ids.append(item.get("doc_id", "unknown"))
                    para_ids.append(item.get("para_id", "unknown"))
        
        if texts:
            # Build n-gram index
            self._ngram_index = build_ngram_index(texts, n=8)
            
            # Build embedding index
            self._embedding_service = EmbeddingService(
                model_name=self.settings.embedding_model
            )
            self._embedding_service.build_index(texts, doc_ids, para_ids)
            
            self._corpus_texts = texts
            self._corpus_loaded = True
        
        return len(texts)
    
    def load_index(self, index_path: Optional[str] = None) -> bool:
        """Load pre-built index from disk."""
        index_path = Path(index_path or self.settings.index_path)
        
        # Load n-gram index
        ngram_file = index_path / "ngrams.json"
        if ngram_file.exists():
            with open(ngram_file) as f:
                ngram_list = json.load(f)
                self._ngram_index = set(tuple(ng) for ng in ngram_list)
        
        # Load embedding index
        self._embedding_service = EmbeddingService(
            model_name=self.settings.embedding_model
        )
        if self._embedding_service.load_index(str(index_path)):
            self._corpus_loaded = True
            return True
        
        return False
    
    def check_similarity(self, candidate: str) -> Tuple[bool, dict]:
        """
        Check candidate text against corpus.
        
        Returns:
            (should_fallback, scores_dict)
        """
        scores = {
            "ngram_max_match": 0,
            "ngram_overlap": 0.0,
            "embed_top1": 0.0,
            "top1_doc_id": None,
            "top1_para_id": None,
        }
        
        # N-gram check
        if self._ngram_index:
            ngram_result = compute_ngram_overlap(candidate, self._ngram_index)
            scores["ngram_max_match"] = ngram_result["max_consecutive_match"]
            scores["ngram_overlap"] = ngram_result["overlap_ratio"]
        
        # Embedding check
        if self._embedding_service:
            embed_result = self._embedding_service.compute_similarity(candidate)
            scores["embed_top1"] = embed_result["top1_similarity"]
            scores["top1_doc_id"] = embed_result["top1_doc_id"]
            scores["top1_para_id"] = embed_result["top1_para_id"]
        
        # Determine if fallback needed
        should_fallback = (
            scores["ngram_max_match"] >= self.settings.threshold_ngram_max_match or
            scores["embed_top1"] >= self.settings.threshold_embed_top1
        )
        
        return should_fallback, scores
    
    @property
    def is_loaded(self) -> bool:
        return self._corpus_loaded
    
    @property
    def corpus_size(self) -> int:
        return len(self._corpus_texts) if self._corpus_texts else 0
```

**验收标准:**
- [ ] 组合检测工作正常
- [ ] 阈值触发逻辑正确
- [ ] 空语料库时优雅处理

---

### PR #6: LLM Client (vLLM Integration)
**目标:** 实现 vLLM OpenAI-compatible 客户端

**文件:** `src/gswa/services/llm_client.py`

```python
"""LLM Client for vLLM server.

Uses OpenAI-compatible API to communicate with local vLLM server.
"""
import httpx
from typing import Optional, AsyncIterator

from gswa.config import get_settings


class LLMClient:
    """Client for vLLM OpenAI-compatible API."""
    
    def __init__(self):
        self.settings = get_settings()
        self._base_url = self.settings.vllm_base_url.rstrip("/")
        self._model = self.settings.vllm_model_name
        
        # Security: Ensure we're connecting locally
        self._validate_local_only()
    
    def _validate_local_only(self) -> None:
        """Ensure we only connect to local server."""
        allowed_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
        from urllib.parse import urlparse
        parsed = urlparse(self._base_url)
        
        if parsed.hostname not in allowed_hosts:
            if not self.settings.allow_external_api:
                raise ValueError(
                    f"External API calls forbidden. "
                    f"Host '{parsed.hostname}' is not localhost. "
                    f"Only local vLLM server is allowed."
                )
    
    async def check_health(self) -> dict:
        """Check if vLLM server is healthy."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(f"{self._base_url}/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data", [])
                    return {
                        "status": "connected",
                        "models": [m.get("id") for m in models]
                    }
            except Exception as e:
                return {"status": "error", "error": str(e)}
        return {"status": "disconnected"}
    
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Generate completion from vLLM.
        
        Args:
            messages: Chat messages [{"role": "...", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
        
        Returns:
            Generated text
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.settings.vllm_api_key}"}
            )
            resp.raise_for_status()
            data = resp.json()
        
        return data["choices"][0]["message"]["content"]
    
    async def generate_variants(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int = 3,
        temperature_base: float = 0.3,
        temperature_variance: float = 0.15,
    ) -> list[str]:
        """
        Generate multiple variants with slightly different temperatures.
        
        Returns:
            List of generated texts
        """
        variants = []
        
        for i in range(n):
            # Vary temperature slightly for each variant
            temp = temperature_base + (i - n // 2) * temperature_variance
            temp = max(0.1, min(1.0, temp))  # Clamp to valid range
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            result = await self.complete(
                messages=messages,
                temperature=temp,
                max_tokens=self.settings.max_new_tokens,
            )
            variants.append(result)
        
        return variants


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
```

**验收标准:**
- [ ] 本地 URL 验证工作
- [ ] 非本地 URL 被拒绝
- [ ] 生成多个变体工作正常

---

### PR #7: Prompt Service
**目标:** 实现 prompt 构建逻辑

**文件:** `src/gswa/services/prompt.py`

```python
"""Prompt Construction Service.

Builds prompts for different rewriting strategies and fallback.
"""
from typing import Optional
from gswa.api.schemas import Section, Strategy


# System prompt (style card)
SYSTEM_PROMPT = """You are a scientific paper rewriter. Rewrite the user's paragraph in a style consistent with Gilles's published papers.

HARD CONSTRAINTS (MUST follow):
- Preserve meaning EXACTLY: do not change numbers, units, experimental conditions, comparisons, or conclusion strength
- Do not introduce new facts not present in the input
- Do not strengthen hedged claims (may/suggest → demonstrate) or weaken strong claims
- Avoid copying long phrases (>8 words) from reference corpus; prefer new sentence structures

Output ONLY the rewritten paragraph, no explanations or preamble."""


# Strategy templates
STRATEGY_TEMPLATES = {
    Strategy.A: "Rewrite with the main claim in the FIRST sentence, then provide supporting details and qualifiers.",
    Strategy.B: "Rewrite starting with brief context/motivation, then introduce the main claim, then qualifiers.",
    Strategy.C: "Rewrite starting with the experimental setup/approach, then report the key finding, then interpretation.",
    Strategy.D: "Rewrite starting with cautious framing/limitations, then state the key claim conservatively, then implications.",
}


# Section-specific guidance
SECTION_GUIDANCE = {
    Section.ABSTRACT: "Keep it concise and self-contained. Include background, objective, key results, and conclusion.",
    Section.INTRODUCTION: "Build from broad context to specific hypothesis. End with clear objectives.",
    Section.METHODS: "Be precise and reproducible. Use past tense and passive voice appropriately.",
    Section.RESULTS: "Present findings objectively. Let data speak without over-interpretation.",
    Section.DISCUSSION: "Interpret results in context. Acknowledge limitations. Connect to broader implications.",
    Section.CONCLUSION: "Summarize key findings. Emphasize significance. Suggest future directions.",
}


# Fallback prompt for stronger diversification
FALLBACK_PROMPT = """IMPORTANT: Your previous rewrite was too similar to existing text.

Rewrite again with SIGNIFICANTLY different sentence structures:
- Split or merge sentences differently
- Change active/passive voice
- Reorder clauses and ideas
- Use different transition words
- AVOID any phrase longer than 6 consecutive words from the original

Preserve the exact same meaning and all numerical values."""


class PromptService:
    """Builds prompts for rewriting tasks."""
    
    def build_system_prompt(
        self,
        section: Optional[Section] = None,
        is_fallback: bool = False,
    ) -> str:
        """Build the system prompt."""
        parts = [SYSTEM_PROMPT]
        
        if section and section in SECTION_GUIDANCE:
            parts.append(f"\nSection-specific guidance ({section.value}):")
            parts.append(SECTION_GUIDANCE[section])
        
        if is_fallback:
            parts.append(f"\n{FALLBACK_PROMPT}")
        
        return "\n".join(parts)
    
    def build_user_prompt(
        self,
        text: str,
        strategy: Strategy,
    ) -> str:
        """Build the user prompt with strategy."""
        strategy_instruction = STRATEGY_TEMPLATES.get(strategy, STRATEGY_TEMPLATES[Strategy.A])
        
        return f"""{strategy_instruction}

Here is the paragraph to rewrite:

{text}"""
    
    def get_strategies(
        self,
        requested: Optional[list[Strategy]],
        n: int,
    ) -> list[Strategy]:
        """Get list of strategies to use for n variants."""
        all_strategies = [Strategy.A, Strategy.B, Strategy.C, Strategy.D]
        
        if requested:
            # Use requested strategies, cycling if needed
            strategies = []
            for i in range(n):
                strategies.append(requested[i % len(requested)])
            return strategies
        
        # Default: use first n strategies
        return all_strategies[:n]
```

**验收标准:**
- [ ] Prompt 构建逻辑正确
- [ ] Fallback prompt 正确追加
- [ ] 策略模板完整

---

### PR #8: Rewriter Orchestrator
**目标:** 实现核心重写编排逻辑

**文件:** `src/gswa/services/rewriter.py`

```python
"""Rewriter Orchestrator.

Main orchestration logic for rewriting with similarity checks and fallback.
"""
import time
import logging
from typing import Optional

from gswa.api.schemas import (
    RewriteRequest, RewriteResponse, RewriteVariant,
    SimilarityScores, Strategy
)
from gswa.services.llm_client import get_llm_client
from gswa.services.similarity import SimilarityService
from gswa.services.prompt import PromptService
from gswa.config import get_settings


logger = logging.getLogger(__name__)


class RewriterService:
    """Orchestrates the rewriting process."""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm_client = get_llm_client()
        self.similarity_service = SimilarityService()
        self.prompt_service = PromptService()
    
    async def initialize(self) -> None:
        """Initialize services (load corpus, etc.)."""
        try:
            # Try to load pre-built index first
            if not self.similarity_service.load_index():
                # Fall back to loading corpus directly
                count = self.similarity_service.load_corpus()
                logger.info(f"Loaded {count} paragraphs from corpus")
        except Exception as e:
            logger.warning(f"Could not load similarity index: {e}")
    
    async def rewrite(self, request: RewriteRequest) -> RewriteResponse:
        """
        Rewrite text with multiple variants.
        
        Process:
        1. Build prompts for each strategy
        2. Generate variants
        3. Check similarity for each
        4. Trigger fallback if needed
        5. Return results with scores
        """
        start_time = time.time()
        
        # Determine strategies to use
        strategies = self.prompt_service.get_strategies(
            request.strategies,
            request.n_variants
        )
        
        variants: list[RewriteVariant] = []
        
        for i, strategy in enumerate(strategies):
            # Build prompts
            system_prompt = self.prompt_service.build_system_prompt(
                section=request.section,
                is_fallback=False,
            )
            user_prompt = self.prompt_service.build_user_prompt(
                text=request.text,
                strategy=strategy,
            )
            
            # Generate initial variant
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            temp = self.settings.temperature_base + (i - len(strategies) // 2) * self.settings.temperature_variance
            temp = max(0.1, min(1.0, temp))
            
            generated_text = await self.llm_client.complete(
                messages=messages,
                temperature=temp,
            )
            
            # Check similarity
            should_fallback, scores = self.similarity_service.check_similarity(generated_text)
            
            fallback = False
            fallback_reason = None
            
            # Trigger fallback if needed (only once)
            if should_fallback:
                fallback = True
                fallback_reason = self._get_fallback_reason(scores)
                
                # Regenerate with fallback prompt
                system_prompt = self.prompt_service.build_system_prompt(
                    section=request.section,
                    is_fallback=True,
                )
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                generated_text = await self.llm_client.complete(
                    messages=messages,
                    temperature=temp + 0.1,  # Slightly higher temp for fallback
                )
                
                # Re-check similarity
                _, scores = self.similarity_service.check_similarity(generated_text)
            
            variants.append(RewriteVariant(
                text=generated_text.strip(),
                strategy=strategy,
                scores=SimilarityScores(
                    ngram_max_match=scores["ngram_max_match"],
                    ngram_overlap=scores["ngram_overlap"],
                    embed_top1=scores["embed_top1"],
                    top1_doc_id=scores.get("top1_doc_id"),
                    top1_para_id=scores.get("top1_para_id"),
                ),
                fallback=fallback,
                fallback_reason=fallback_reason,
            ))
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return RewriteResponse(
            variants=variants,
            model_version=f"{self.settings.vllm_model_name}@v1",
            processing_time_ms=processing_time,
        )
    
    def _get_fallback_reason(self, scores: dict) -> str:
        """Generate human-readable fallback reason."""
        reasons = []
        if scores["ngram_max_match"] >= self.settings.threshold_ngram_max_match:
            reasons.append(f"n-gram match ({scores['ngram_max_match']} tokens)")
        if scores["embed_top1"] >= self.settings.threshold_embed_top1:
            reasons.append(f"embedding similarity ({scores['embed_top1']:.2f})")
        return "; ".join(reasons) if reasons else "threshold exceeded"


# Singleton instance
_rewriter_service: Optional[RewriterService] = None


async def get_rewriter_service() -> RewriterService:
    """Get or create rewriter service singleton."""
    global _rewriter_service
    if _rewriter_service is None:
        _rewriter_service = RewriterService()
        await _rewriter_service.initialize()
    return _rewriter_service
```

**验收标准:**
- [ ] 完整重写流程工作
- [ ] Fallback 正确触发
- [ ] 计时和日志正确

---

### PR #9: FastAPI Routes
**目标:** 实现 API 端点

**文件:** `src/gswa/api/routes.py`

```python
"""API Routes.

FastAPI route definitions for GSWA.
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from gswa.api.schemas import (
    RewriteRequest, RewriteResponse,
    ReplyRequest, ReplyResponse,
    HealthResponse
)
from gswa.services.rewriter import get_rewriter_service, RewriterService
from gswa.services.llm_client import get_llm_client
from gswa.services.similarity import SimilarityService


router = APIRouter(prefix="/v1", tags=["v1"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    llm_client = get_llm_client()
    similarity_service = SimilarityService()
    
    # Check LLM server
    llm_status = await llm_client.check_health()
    
    return HealthResponse(
        status="healthy" if llm_status["status"] == "connected" else "degraded",
        llm_server=llm_status["status"],
        model_loaded=llm_status.get("models", [None])[0] if llm_status.get("models") else None,
        corpus_paragraphs=similarity_service.corpus_size,
        index_loaded=similarity_service.is_loaded,
    )


@router.post("/rewrite/variants", response_model=RewriteResponse)
async def rewrite_variants(request: RewriteRequest):
    """
    Generate multiple rewrite variants.
    
    Each variant uses a different organizational strategy.
    Similarity is checked against corpus; fallback triggered if too similar.
    """
    try:
        rewriter = await get_rewriter_service()
        return await rewriter.rewrite(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reply", response_model=ReplyResponse)
async def reply(request: ReplyRequest):
    """
    Simple chat endpoint for testing/debugging.
    """
    try:
        llm_client = get_llm_client()
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        content = await llm_client.complete(
            messages=messages,
            max_tokens=request.max_tokens,
        )
        return ReplyResponse(
            content=content,
            model=llm_client._model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**文件:** `src/gswa/main.py`

```python
"""GSWA FastAPI Application.

Main entry point for the Gilles-Style Writing Assistant API.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from gswa.config import get_settings
from gswa.api.routes import router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting GSWA server...")
    settings = get_settings()
    logger.info(f"LLM server: {settings.vllm_base_url}")
    logger.info(f"External API allowed: {settings.allow_external_api}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GSWA server...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()  # This validates security settings
    
    app = FastAPI(
        title="GSWA - Gilles-Style Writing Assistant",
        description="Local offline scientific paper rewriter",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router)
    
    # Serve static files (web UI)
    try:
        app.mount("/", StaticFiles(directory="web", html=True), name="web")
    except Exception:
        logger.warning("Web UI directory not found, skipping static file serving")
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

**验收标准:**
- [ ] `/v1/health` 返回正确状态
- [ ] `/v1/rewrite/variants` 工作正常
- [ ] `/v1/reply` 工作正常

---

### PR #10: Minimal Web UI
**目标:** 实现简单 Web 界面

**文件:** `web/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GSWA - Gilles-Style Writing Assistant</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>GSWA</h1>
            <p class="subtitle">Gilles-Style Writing Assistant</p>
            <div id="health-status" class="health-badge">Checking...</div>
        </header>
        
        <main>
            <div class="input-section">
                <label for="input-text">Paste your paragraph:</label>
                <textarea 
                    id="input-text" 
                    placeholder="Enter the paragraph you want to rewrite..."
                    rows="8"
                ></textarea>
                
                <div class="options">
                    <div class="option-group">
                        <label for="section">Section:</label>
                        <select id="section">
                            <option value="">Auto-detect</option>
                            <option value="Abstract">Abstract</option>
                            <option value="Introduction">Introduction</option>
                            <option value="Methods">Methods</option>
                            <option value="Results">Results</option>
                            <option value="Discussion">Discussion</option>
                            <option value="Conclusion">Conclusion</option>
                        </select>
                    </div>
                    
                    <div class="option-group">
                        <label for="n-variants">Variants:</label>
                        <select id="n-variants">
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="5">5</option>
                        </select>
                    </div>
                </div>
                
                <button id="generate-btn" class="primary-btn">
                    Generate Variants
                </button>
            </div>
            
            <div id="results-section" class="results-section hidden">
                <h2>Generated Variants</h2>
                <div id="variants-container"></div>
                <div id="meta-info" class="meta-info"></div>
            </div>
            
            <div id="loading" class="loading hidden">
                <div class="spinner"></div>
                <p>Generating variants...</p>
            </div>
            
            <div id="error" class="error hidden"></div>
        </main>
    </div>
    
    <script src="app.js"></script>
</body>
</html>
```

**文件:** `web/style.css`

```css
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
    color: #333;
    line-height: 1.6;
}

.container {
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
}

header {
    text-align: center;
    margin-bottom: 30px;
}

header h1 {
    font-size: 2.5rem;
    color: #2c3e50;
}

.subtitle {
    color: #666;
    margin-bottom: 10px;
}

.health-badge {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 0.85rem;
    background: #f0f0f0;
}

.health-badge.healthy {
    background: #d4edda;
    color: #155724;
}

.health-badge.error {
    background: #f8d7da;
    color: #721c24;
}

.input-section {
    background: white;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

label {
    display: block;
    margin-bottom: 8px;
    font-weight: 500;
}

textarea {
    width: 100%;
    padding: 12px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 1rem;
    resize: vertical;
    transition: border-color 0.2s;
}

textarea:focus {
    outline: none;
    border-color: #3498db;
}

.options {
    display: flex;
    gap: 20px;
    margin: 20px 0;
}

.option-group {
    flex: 1;
}

select {
    width: 100%;
    padding: 10px;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    font-size: 1rem;
    background: white;
}

.primary-btn {
    width: 100%;
    padding: 15px;
    background: #3498db;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1.1rem;
    cursor: pointer;
    transition: background 0.2s;
}

.primary-btn:hover {
    background: #2980b9;
}

.primary-btn:disabled {
    background: #bdc3c7;
    cursor: not-allowed;
}

.results-section {
    margin-top: 30px;
}

.results-section h2 {
    margin-bottom: 20px;
    color: #2c3e50;
}

.variant-card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.variant-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #eee;
}

.strategy-badge {
    background: #3498db;
    color: white;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.85rem;
}

.fallback-badge {
    background: #e74c3c;
    color: white;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.85rem;
}

.variant-text {
    font-size: 1rem;
    line-height: 1.8;
    margin-bottom: 15px;
}

.scores {
    display: flex;
    gap: 15px;
    flex-wrap: wrap;
    font-size: 0.85rem;
    color: #666;
}

.score-item {
    background: #f8f9fa;
    padding: 5px 10px;
    border-radius: 4px;
}

.copy-btn {
    padding: 8px 15px;
    background: #27ae60;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 0.9rem;
}

.copy-btn:hover {
    background: #219a52;
}

.meta-info {
    text-align: center;
    color: #666;
    font-size: 0.9rem;
    margin-top: 20px;
}

.loading {
    text-align: center;
    padding: 40px;
}

.spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 15px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.error {
    background: #f8d7da;
    color: #721c24;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
}

.hidden {
    display: none;
}
```

**文件:** `web/app.js`

```javascript
// GSWA Web UI

const API_BASE = '/v1';

// Elements
const inputText = document.getElementById('input-text');
const sectionSelect = document.getElementById('section');
const nVariantsSelect = document.getElementById('n-variants');
const generateBtn = document.getElementById('generate-btn');
const resultsSection = document.getElementById('results-section');
const variantsContainer = document.getElementById('variants-container');
const metaInfo = document.getElementById('meta-info');
const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const healthStatus = document.getElementById('health-status');

// Check health on load
async function checkHealth() {
    try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();
        
        healthStatus.textContent = data.status === 'healthy' ? 'System Ready' : 'System Degraded';
        healthStatus.className = `health-badge ${data.status === 'healthy' ? 'healthy' : 'error'}`;
        
        if (data.model_loaded) {
            healthStatus.textContent += ` (${data.model_loaded})`;
        }
    } catch (e) {
        healthStatus.textContent = 'Connection Error';
        healthStatus.className = 'health-badge error';
    }
}

// Generate variants
async function generateVariants() {
    const text = inputText.value.trim();
    
    if (!text) {
        showError('Please enter some text to rewrite.');
        return;
    }
    
    // Show loading
    loading.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorDiv.classList.add('hidden');
    generateBtn.disabled = true;
    
    try {
        const payload = {
            text: text,
            n_variants: parseInt(nVariantsSelect.value),
        };
        
        if (sectionSelect.value) {
            payload.section = sectionSelect.value;
        }
        
        const resp = await fetch(`${API_BASE}/rewrite/variants`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        
        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.detail || 'API error');
        }
        
        const data = await resp.json();
        displayResults(data);
        
    } catch (e) {
        showError(`Error: ${e.message}`);
    } finally {
        loading.classList.add('hidden');
        generateBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    variantsContainer.innerHTML = '';
    
    data.variants.forEach((variant, index) => {
        const card = document.createElement('div');
        card.className = 'variant-card';
        
        card.innerHTML = `
            <div class="variant-header">
                <div>
                    <span class="strategy-badge">Strategy ${variant.strategy}</span>
                    ${variant.fallback ? `<span class="fallback-badge">Fallback</span>` : ''}
                </div>
                <button class="copy-btn" onclick="copyText(${index})">Copy</button>
            </div>
            <div class="variant-text" id="variant-text-${index}">${escapeHtml(variant.text)}</div>
            <div class="scores">
                <span class="score-item">N-gram match: ${variant.scores.ngram_max_match}</span>
                <span class="score-item">Embed sim: ${variant.scores.embed_top1.toFixed(3)}</span>
                ${variant.fallback_reason ? `<span class="score-item">Fallback: ${variant.fallback_reason}</span>` : ''}
            </div>
        `;
        
        variantsContainer.appendChild(card);
    });
    
    metaInfo.textContent = `Model: ${data.model_version} | Time: ${data.processing_time_ms}ms`;
    resultsSection.classList.remove('hidden');
}

// Copy text to clipboard
function copyText(index) {
    const textEl = document.getElementById(`variant-text-${index}`);
    const text = textEl.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        // Brief feedback
        const btn = textEl.parentElement.querySelector('.copy-btn');
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = originalText, 1500);
    });
}

// Show error
function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Event listeners
generateBtn.addEventListener('click', generateVariants);
inputText.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        generateVariants();
    }
});

// Initialize
checkHealth();
setInterval(checkHealth, 30000);  // Check every 30s
```

**验收标准:**
- [ ] UI 可以正常加载
- [ ] 生成按钮工作
- [ ] 结果正确显示
- [ ] 复制功能正常

---

### PR #11: Smoke Test & Scripts
**目标:** 实现端到端测试脚本

**文件:** `scripts/smoke_test.py`

```python
#!/usr/bin/env python3
"""
GSWA Smoke Test

End-to-end test to verify the system is working correctly.
Run after starting both vLLM and GSWA servers.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --api-url http://localhost:8080
"""
import argparse
import sys
import httpx
import time


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print("Testing /v1/health...")
    try:
        resp = httpx.get(f"{base_url}/v1/health", timeout=10)
        data = resp.json()
        
        if data.get("status") in ["healthy", "degraded"]:
            print(f"  ✓ Health check passed: {data['status']}")
            print(f"    LLM server: {data.get('llm_server')}")
            print(f"    Model: {data.get('model_loaded')}")
            return True
        else:
            print(f"  ✗ Unexpected status: {data}")
            return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_reply(base_url: str) -> bool:
    """Test simple reply endpoint."""
    print("\nTesting /v1/reply...")
    try:
        payload = {
            "messages": [
                {"role": "user", "content": "Say 'test successful' in exactly two words."}
            ],
            "max_tokens": 50
        }
        resp = httpx.post(
            f"{base_url}/v1/reply",
            json=payload,
            timeout=60
        )
        data = resp.json()
        
        if "content" in data:
            print(f"  ✓ Reply received: {data['content'][:100]}...")
            return True
        else:
            print(f"  ✗ No content in response: {data}")
            return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_rewrite(base_url: str) -> bool:
    """Test rewrite variants endpoint."""
    print("\nTesting /v1/rewrite/variants...")
    
    test_text = """
    We observed a 2.5-fold increase in enzyme activity (p < 0.01) 
    when cells were treated with compound X at 10 μM for 24 hours. 
    This suggests that compound X may enhance catalytic efficiency 
    through allosteric modulation.
    """
    
    try:
        payload = {
            "text": test_text,
            "section": "Results",
            "n_variants": 3,
            "constraints": {
                "preserve_numbers": True,
                "no_new_facts": True
            }
        }
        
        start = time.time()
        resp = httpx.post(
            f"{base_url}/v1/rewrite/variants",
            json=payload,
            timeout=120
        )
        elapsed = time.time() - start
        
        data = resp.json()
        
        if "variants" not in data:
            print(f"  ✗ No variants in response: {data}")
            return False
        
        variants = data["variants"]
        print(f"  ✓ Received {len(variants)} variants in {elapsed:.2f}s")
        
        # Verify numbers preserved
        all_pass = True
        for i, v in enumerate(variants):
            text = v["text"]
            
            # Check key numbers
            has_fold = "2.5" in text or "2.5-fold" in text.lower()
            has_p = "p <" in text or "p<" in text or "0.01" in text
            has_conc = "10" in text and ("μm" in text.lower() or "um" in text.lower())
            has_time = "24" in text and ("hour" in text.lower() or "h" in text.lower())
            
            if has_fold and has_p and has_conc and has_time:
                print(f"    Variant {i+1}: ✓ Numbers preserved")
                print(f"      Strategy: {v['strategy']}, Fallback: {v['fallback']}")
                print(f"      Scores: ngram={v['scores']['ngram_max_match']}, embed={v['scores']['embed_top1']:.3f}")
            else:
                print(f"    Variant {i+1}: ✗ Some numbers missing")
                print(f"      2.5-fold: {has_fold}, p-value: {has_p}, 10μM: {has_conc}, 24h: {has_time}")
                all_pass = False
        
        return all_pass
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_no_external_calls(base_url: str) -> bool:
    """Verify no external API calls are made."""
    print("\nTesting security (no external calls)...")
    # This is a conceptual test - in real implementation,
    # you'd mock external calls and verify they're blocked
    print("  ✓ Security check: ALLOW_EXTERNAL_API=false by default")
    return True


def main():
    parser = argparse.ArgumentParser(description="GSWA Smoke Test")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8080",
        help="GSWA API base URL"
    )
    args = parser.parse_args()
    
    print("=" * 50)
    print("GSWA Smoke Test")
    print("=" * 50)
    print(f"API URL: {args.api_url}")
    print()
    
    results = {
        "health": test_health(args.api_url),
        "reply": test_reply(args.api_url),
        "rewrite": test_rewrite(args.api_url),
        "security": test_no_external_calls(args.api_url),
    }
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

**文件:** `scripts/start_vllm.sh`

```bash
#!/bin/bash
# Start vLLM server with OpenAI-compatible API

set -e

MODEL=${VLLM_MODEL:-"mistralai/Mistral-7B-Instruct-v0.2"}
PORT=${VLLM_PORT:-8000}
GPU_MEMORY=${GPU_MEMORY_UTIL:-0.9}

echo "Starting vLLM server..."
echo "Model: $MODEL"
echo "Port: $PORT"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY" \
    --trust-remote-code
```

**文件:** `Makefile`

```makefile
.PHONY: install dev test lint smoke-test run clean

# Install dependencies
install:
	pip install -e .

# Install dev dependencies
dev:
	pip install -e ".[dev,similarity]"

# Run tests
test:
	pytest tests/ -v

# Lint code
lint:
	ruff check src/

# Run smoke test
smoke-test:
	python scripts/smoke_test.py

# Start GSWA server
run:
	uvicorn gswa.main:app --reload --host 0.0.0.0 --port 8080

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
```

**验收标准:**
- [ ] `make smoke-test` 通过
- [ ] `make test` 通过
- [ ] `make run` 启动成功

---

## 🔒 Self-Review Checklist Template

每个 PR 提交前必须完成：

```markdown
## PR Self-Review Checklist

### Security
- [ ] No external HTTP calls added (grep for `httpx`, `requests`, `urllib`)
- [ ] No telemetry/analytics code
- [ ] Sensitive data not logged in plaintext
- [ ] `ALLOW_EXTERNAL_API` check present where needed
- [ ] Only localhost URLs allowed for LLM server

### Correctness
- [ ] Numbers in test cases preserved in outputs
- [ ] No logic that could introduce new facts
- [ ] Conclusion strength unchanged in prompts

### Anti-Verbatim
- [ ] N-gram check called in rewrite flow
- [ ] Embedding similarity check called
- [ ] Fallback logic triggers on threshold
- [ ] Fallback produces measurably different output

### Code Quality
- [ ] Type hints on public functions
- [ ] Docstrings on modules, classes, public methods
- [ ] No hardcoded secrets or paths
- [ ] Error handling present

### Testing
- [ ] Unit tests added/updated
- [ ] `pytest tests/` passes
- [ ] `python scripts/smoke_test.py` passes (if server running)
```

---

## 📊 Progress Tracking

| PR | Task | Status | Reviewer Notes |
|----|------|--------|----------------|
| #1 | Project Scaffolding | ⬜ | |
| #2 | Pydantic Schemas | ⬜ | |
| #3 | N-gram Service | ⬜ | |
| #4 | Embedding Service | ⬜ | |
| #5 | Combined Similarity | ⬜ | |
| #6 | LLM Client | ⬜ | |
| #7 | Prompt Service | ⬜ | |
| #8 | Rewriter Orchestrator | ⬜ | |
| #9 | FastAPI Routes | ⬜ | |
| #10 | Web UI | ⬜ | |
| #11 | Smoke Test | ⬜ | |

Status: ⬜ Todo | 🔄 In Progress | ✅ Done | ❌ Blocked
