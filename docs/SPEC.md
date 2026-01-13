# GSWA Project Specification
# Gilles-Style Writing Assistant - 本地离线科学论文重写器

**Version:** v1.0  
**Date:** 2026-01-09  
**Target:** Claude Code / AI Coding Agents

---

## 1. Executive Summary

GSWA 是一个**完全本地运行**的科学论文段落重写工具。核心功能：
- 将用户段落重写为符合 Gilles 写作风格的版本
- 生成 3-5 个不同组织策略的候选版本
- 自动检测与语料库的相似度，超阈值时触发回退重写
- 保持科学含义不变（数值、实验条件、结论强度）

**Hard Constraints:**
1. ❌ 默认禁止任何外部 API 调用
2. ❌ 禁止网络遥测
3. ✅ 所有语料库文本视为机密
4. ✅ 语义必须完全保持

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web UI (React/HTML)                      │
│                    paste → generate → copy                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Orchestrator                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ POST /v1/rewrite/variants                                │    │
│  │ POST /v1/reply (simple chat)                             │    │
│  │ GET  /v1/health                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                               │                                  │
│  ┌───────────────┬────────────┴─────────────┬───────────────┐   │
│  │  Prompt       │    Similarity Service    │   Fallback    │   │
│  │  Constructor  │    (n-gram + embed)      │   Logic       │   │
│  └───────────────┴──────────────────────────┴───────────────┘   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ OpenAI-compatible API
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 vLLM Inference Server                            │
│            (local model, OpenAI-compatible)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
gswa/
├── README.md
├── pyproject.toml              # Python package config
├── requirements.txt
├── .env.example                # Environment template
├── Makefile                    # Common commands
│
├── docs/
│   ├── SPEC.md                 # This file
│   ├── PROMPTS.md              # Prompt templates
│   └── API.md                  # API documentation
│
├── src/
│   └── gswa/
│       ├── __init__.py
│       ├── config.py           # Configuration & env vars
│       ├── main.py             # FastAPI app entry
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py       # API route definitions
│       │   └── schemas.py      # Pydantic models
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   ├── llm_client.py   # vLLM client wrapper
│       │   ├── similarity.py   # Similarity service
│       │   ├── prompt.py       # Prompt construction
│       │   └── rewriter.py     # Orchestration logic
│       │
│       └── utils/
│           ├── __init__.py
│           ├── ngram.py        # N-gram overlap functions
│           ├── embedding.py    # Embedding similarity
│           └── logging.py      # Audit logging
│
├── data/
│   ├── corpus/
│   │   ├── raw/                # Original PDFs/DOCX (gitignored)
│   │   └── parsed/             # Paragraph JSONL
│   └── index/                  # FAISS index (gitignored)
│
├── web/                        # Minimal Web UI
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── scripts/
│   ├── start_vllm.sh           # vLLM server startup
│   ├── build_index.py          # Build similarity index
│   └── smoke_test.py           # End-to-end test
│
└── tests/
    ├── __init__.py
    ├── conftest.py             # Pytest fixtures
    ├── test_similarity.py
    ├── test_api.py
    └── test_rewriter.py
```

---

## 4. API Specification (MVP)

### 4.1 POST /v1/rewrite/variants

**Request:**
```json
{
  "text": "The input paragraph to rewrite...",
  "section": "Discussion",           // Optional: Abstract|Introduction|Methods|Results|Discussion|Conclusion
  "n_variants": 3,                   // 1-5, default 3
  "strategies": ["A", "B", "C"],     // Optional subset of A,B,C,D
  "constraints": {
    "preserve_numbers": true,        // Default true
    "no_new_facts": true             // Default true
  }
}
```

**Response:**
```json
{
  "variants": [
    {
      "text": "Rewritten paragraph variant 1...",
      "strategy": "A",
      "scores": {
        "ngram_max_match": 5,        // Longest consecutive n-gram match
        "ngram_overlap": 0.03,       // Overlap ratio
        "embed_top1": 0.71           // Cosine similarity to most similar corpus paragraph
      },
      "fallback": false,             // Whether fallback was triggered
      "fallback_reason": null
    },
    // ... more variants
  ],
  "model_version": "mistral-7b-instruct@v1",
  "processing_time_ms": 2450
}
```

### 4.2 POST /v1/reply

Simple chat endpoint for testing/debugging.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Help me understand..."}
  ],
  "max_tokens": 512
}
```

**Response:**
```json
{
  "content": "Response text...",
  "model": "mistral-7b-instruct"
}
```

### 4.3 GET /v1/health

**Response:**
```json
{
  "status": "healthy",
  "llm_server": "connected",
  "model_loaded": "mistral-7b-instruct",
  "corpus_paragraphs": 1234,
  "index_loaded": true
}
```

---

## 5. Similarity Service Specification

### 5.1 N-gram Overlap

```python
def compute_ngram_overlap(candidate: str, corpus_ngrams: Set[Tuple]) -> dict:
    """
    Compute n-gram overlap metrics.
    
    Returns:
        {
            "max_consecutive_match": int,  # Longest matching n-gram length
            "overlap_5gram": float,        # 5-gram Jaccard similarity
            "overlap_8gram": float         # 8-gram Jaccard similarity
        }
    """
```

**Thresholds (初始值，可调):**
- `max_consecutive_match >= 12` tokens → 触发回退
- `overlap_8gram >= 0.15` → 触发回退

### 5.2 Embedding Similarity

```python
def compute_embed_similarity(candidate: str, faiss_index, top_k: int = 5) -> dict:
    """
    Compute embedding-based similarity.
    
    Returns:
        {
            "top1_similarity": float,      # Cosine sim to most similar
            "top1_doc_id": str,
            "top1_para_id": str,
            "topk_avg": float              # Average of top-k similarities
        }
    """
```

**Thresholds (初始值):**
- `top1_similarity >= 0.88` → 触发回退

### 5.3 Combined Gate

```python
def similarity_gate(candidate: str, corpus_index) -> Tuple[bool, dict]:
    """
    Combined similarity check.
    
    Returns:
        (should_fallback: bool, scores: dict)
    """
    ngram_scores = compute_ngram_overlap(candidate, corpus_index.ngrams)
    embed_scores = compute_embed_similarity(candidate, corpus_index.faiss)
    
    trigger = (
        ngram_scores["max_consecutive_match"] >= 12 or
        embed_scores["top1_similarity"] >= 0.88
    )
    
    return trigger, {**ngram_scores, **embed_scores}
```

---

## 6. Prompt Templates

### 6.1 System Prompt (Style Card)

```
You are a scientific paper rewriter. Rewrite the user's paragraph in a style consistent with Gilles's published papers.

HARD CONSTRAINTS (MUST follow):
- Preserve meaning EXACTLY: do not change numbers, units, experimental conditions, comparisons, or conclusion strength
- Do not introduce new facts not present in the input
- Do not strengthen hedged claims (may/suggest → demonstrate) or weaken strong claims
- Avoid copying long phrases (>8 words) from reference corpus; prefer new sentence structures

Output ONLY the rewritten paragraph, no explanations.
```

### 6.2 Strategy Templates

**Strategy A (Conclusion-first):**
```
Rewrite with the main claim in the FIRST sentence, then provide supporting details and qualifiers.
```

**Strategy B (Background-first):**
```
Rewrite starting with brief context/motivation, then introduce the main claim, then qualifiers.
```

**Strategy C (Methods-first):**
```
Rewrite starting with the experimental setup/approach, then report the key finding, then interpretation.
```

**Strategy D (Cautious-first):**
```
Rewrite starting with cautious framing/limitations, then state the key claim conservatively, then implications.
```

### 6.3 Fallback Prompt (Stronger Diversification)

```
IMPORTANT: Your previous rewrite was too similar to existing text. 

Rewrite again with SIGNIFICANTLY different sentence structures:
- Split or merge sentences differently
- Change active/passive voice
- Reorder clauses and ideas
- Use different transition words
- AVOID any phrase longer than 6 consecutive words from the original

Preserve the exact same meaning and all numerical values.
```

---

## 7. Configuration

### 7.1 Environment Variables (.env)

```bash
# === Security ===
ALLOW_EXTERNAL_API=false          # MUST be false for production
GSWA_LOG_LEVEL=INFO

# === LLM Server ===
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=mistral-7b-instruct
VLLM_API_KEY=dummy                # vLLM doesn't require real key locally

# === Similarity Thresholds ===
THRESHOLD_NGRAM_MAX_MATCH=12
THRESHOLD_NGRAM_OVERLAP=0.15
THRESHOLD_EMBED_TOP1=0.88

# === Generation Parameters ===
DEFAULT_N_VARIANTS=3
MAX_N_VARIANTS=5
TEMPERATURE_BASE=0.3
TEMPERATURE_VARIANCE=0.15         # Variants use temp ± variance
MAX_NEW_TOKENS=1024

# === Paths ===
CORPUS_PATH=./data/corpus/parsed
INDEX_PATH=./data/index
LOG_PATH=./logs

# === Embedding Model (local) ===
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 7.2 Config Validation

```python
# src/gswa/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Security - HARDCODED default to prevent external calls
    allow_external_api: bool = False
    
    # Validate at startup
    def validate_security(self):
        if self.allow_external_api:
            raise ValueError(
                "ALLOW_EXTERNAL_API=true is FORBIDDEN. "
                "This system must run fully offline."
            )
```

---

## 8. Acceptance Criteria (MVP)

### 8.1 Functional Requirements

| ID | Requirement | Test |
|----|-------------|------|
| F1 | Generate 3-5 rewrite variants | API returns correct number |
| F2 | Each variant uses different strategy | Verify strategy field |
| F3 | Similarity scores computed | All score fields present |
| F4 | Fallback triggers on high similarity | Mock high-sim corpus, verify |
| F5 | Fallback produces lower-similarity output | Compare before/after |
| F6 | No external API calls | Network mock fails = test passes |

### 8.2 Performance Requirements

| Metric | Target |
|--------|--------|
| 3 variants for 200-word input | ≤ 10 seconds (GPU) |
| API response time (health) | ≤ 100ms |
| Memory usage (server) | ≤ 8GB (7B model quantized) |

### 8.3 Quality Gate Checklist (每个 PR 必须通过)

```markdown
## Self-Review Checklist

### Security
- [ ] No external HTTP calls added
- [ ] No telemetry/analytics code
- [ ] Sensitive data not logged (only hashes if needed)
- [ ] ALLOW_EXTERNAL_API check present

### Correctness  
- [ ] Numbers in input preserved in output (test cases)
- [ ] No new facts introduced (prompt enforces)
- [ ] Conclusion strength unchanged

### Anti-Verbatim
- [ ] N-gram check implemented/called
- [ ] Embedding similarity check implemented/called
- [ ] Fallback logic triggers correctly
- [ ] Fallback produces different output

### Code Quality
- [ ] Type hints on public functions
- [ ] Docstrings on modules and classes
- [ ] Tests pass: `pytest tests/`
- [ ] Smoke test passes: `python scripts/smoke_test.py`
```

---

## 9. Development Phases

### Phase 1: MVP (当前目标)

**Deliverables:**
1. FastAPI server with `/v1/rewrite/variants`, `/v1/reply`, `/v1/health`
2. Similarity service (n-gram + embedding)
3. vLLM integration (OpenAI-compatible client)
4. Minimal web UI (HTML/JS, no framework)
5. Smoke test script
6. Basic unit tests

**Timeline:** 1-2 weeks

### Phase 2: Preference Learning

**Deliverables:**
1. `/v1/feedback` endpoint
2. Preference data collection (best/worst)
3. DPO training script
4. Regression test harness

### Phase 3: Enhanced Retrieval

**Deliverables:**
1. Full corpus indexing (FAISS)
2. Exemplar retrieval for style grounding
3. Explainable similarity logs

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Model memorizes corpus | Similarity gate + fallback |
| Semantic drift | Hard constraints in prompt + manual review |
| Slow inference | Start with 7B model; use vLLM optimization |
| Section misclassification | UI allows manual section selection |

---

## Appendix A: Sample Test Cases

### Test Case 1: Basic Rewrite
```python
input_text = "We observed a 2.5-fold increase in enzyme activity (p < 0.01) when cells were treated with compound X at 10 μM for 24 hours."

# Expected: All numbers preserved, meaning unchanged
assert "2.5-fold" in output or "2.5 fold" in output
assert "p < 0.01" in output or "p<0.01" in output
assert "10 μM" in output or "10μM" in output
assert "24 hours" in output or "24 h" in output
```

### Test Case 2: Fallback Trigger
```python
# Mock corpus with exact match
corpus = ["We observed a 2.5-fold increase in enzyme activity..."]

# First generation returns high similarity
# System should auto-regenerate with fallback prompt
assert result.variants[0].fallback == True
assert result.variants[0].scores["ngram_max_match"] < 12  # After fallback
```

### Test Case 3: No External Calls
```python
import responses

@responses.activate  # Blocks all external HTTP
def test_no_external_calls():
    # If any external call is made, this will raise
    result = client.post("/v1/rewrite/variants", json={...})
    assert result.status_code == 200
```
