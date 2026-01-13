# GSWA Claude Code Development Prompt
# 完整开发指令

---

## 🎯 Project Identity

You are implementing **GSWA (Gilles-Style Writing Assistant)** - a fully local, offline scientific paper paragraph rewriter. This system must:

1. **Never make external API calls** - All processing is local
2. **Preserve scientific meaning exactly** - No changing numbers, conditions, or conclusions
3. **Detect and prevent verbatim copying** - N-gram + embedding similarity gates
4. **Generate diverse variants** - Multiple organizational strategies

---

## 📁 Reference Documents

Before starting any task, read these documents:

```
docs/SPEC.md    - Complete technical specification
docs/TASKS.md   - Task breakdown with code templates
```

---

## 🚀 Quick Start Instructions

### For Claude Code Agent:

```
# 1. Read the specification first
Read docs/SPEC.md carefully to understand:
- System architecture
- API endpoints
- Similarity thresholds
- Security constraints

# 2. Follow task breakdown
Read docs/TASKS.md and implement PRs in order:
PR #1 → PR #2 → ... → PR #11

# 3. After each PR, run self-review checklist
See "Self-Review Checklist" section in TASKS.md

# 4. Test frequently
make test        # Unit tests
make smoke-test  # End-to-end test
```

---

## 🔐 CRITICAL Security Rules

These rules are **NON-NEGOTIABLE**:

### 1. No External API Calls

```python
# ❌ FORBIDDEN - External API
httpx.get("https://api.openai.com/...")
requests.post("https://external-service.com/...")

# ✅ ALLOWED - Local only
httpx.get("http://localhost:8000/v1/...")
httpx.post("http://127.0.0.1:8000/...")
```

### 2. URL Validation

```python
# MUST validate all URLs before requests
def _validate_local_only(url: str) -> None:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    allowed_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
    
    if parsed.hostname not in allowed_hosts:
        raise ValueError(f"External API calls forbidden: {parsed.hostname}")
```

### 3. Configuration Default

```python
# In config.py - HARDCODED default
allow_external_api: bool = False  # NEVER change this default

def validate_security(self):
    if self.allow_external_api:
        raise ValueError("ALLOW_EXTERNAL_API=true is FORBIDDEN")
```

### 4. No Telemetry

```python
# ❌ FORBIDDEN
analytics.track("user_action", {...})
sentry.capture_exception(e)
logger.info(f"User input: {user_text}")  # Don't log sensitive content

# ✅ ALLOWED
logger.info("Rewrite request received")
logger.info(f"Generated {len(variants)} variants in {time_ms}ms")
```

---

## 📝 Implementation Checklist

### Phase 1 MVP Tasks

```
[ ] PR #1: Project scaffolding (pyproject.toml, config.py, .env.example)
[ ] PR #2: Pydantic schemas (api/schemas.py)
[ ] PR #3: N-gram similarity (utils/ngram.py)
[ ] PR #4: Embedding similarity (utils/embedding.py)
[ ] PR #5: Combined similarity service (services/similarity.py)
[ ] PR #6: LLM client for vLLM (services/llm_client.py)
[ ] PR #7: Prompt construction (services/prompt.py)
[ ] PR #8: Rewriter orchestrator (services/rewriter.py)
[ ] PR #9: FastAPI routes (api/routes.py, main.py)
[ ] PR #10: Minimal web UI (web/index.html, style.css, app.js)
[ ] PR #11: Smoke test and scripts (scripts/smoke_test.py)
```

---

## 🧪 Testing Requirements

### Every PR Must Include:

1. **Unit tests** in `tests/test_*.py`
2. **Type hints** on public functions
3. **Docstrings** on modules and classes

### Test Commands:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_similarity.py -v

# Run with coverage
pytest tests/ --cov=gswa --cov-report=html

# Smoke test (requires running server)
python scripts/smoke_test.py
```

### Key Test Cases:

```python
# 1. Numbers must be preserved
def test_numbers_preserved():
    input_text = "We observed a 2.5-fold increase (p < 0.01)..."
    result = rewriter.rewrite(input_text)
    assert "2.5" in result.text
    assert "0.01" in result.text

# 2. Fallback must trigger on high similarity
def test_fallback_triggers():
    # Mock corpus with exact match
    result = rewriter.rewrite(exact_corpus_text)
    assert result.fallback == True

# 3. No external calls
@responses.activate  # Blocks external HTTP
def test_no_external_calls():
    # If any external call is made, test fails
    result = client.post("/v1/rewrite/variants", json={...})
    assert result.status_code == 200
```

---

## 🏗️ Code Templates

### Config Template:

```python
# src/gswa/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Security - HARDCODED
    allow_external_api: bool = False
    
    # LLM
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "mistral-7b-instruct"
    
    # Thresholds
    threshold_ngram_max_match: int = 12
    threshold_embed_top1: float = 0.88
    
    def validate_security(self) -> None:
        if self.allow_external_api:
            raise ValueError("External API forbidden")

@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_security()
    return settings
```

### API Route Template:

```python
# src/gswa/api/routes.py
from fastapi import APIRouter, HTTPException
from gswa.api.schemas import RewriteRequest, RewriteResponse

router = APIRouter(prefix="/v1")

@router.post("/rewrite/variants", response_model=RewriteResponse)
async def rewrite_variants(request: RewriteRequest):
    try:
        rewriter = await get_rewriter_service()
        return await rewriter.rewrite(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Similarity Check Template:

```python
# src/gswa/services/similarity.py
def check_similarity(self, candidate: str) -> Tuple[bool, dict]:
    """
    Check candidate against corpus.
    Returns: (should_fallback, scores_dict)
    """
    scores = {
        "ngram_max_match": 0,
        "ngram_overlap": 0.0,
        "embed_top1": 0.0,
    }
    
    # N-gram check
    if self._ngram_index:
        result = compute_ngram_overlap(candidate, self._ngram_index)
        scores["ngram_max_match"] = result["max_consecutive_match"]
    
    # Embedding check
    if self._embedding_service:
        result = self._embedding_service.compute_similarity(candidate)
        scores["embed_top1"] = result["top1_similarity"]
    
    # Determine fallback
    should_fallback = (
        scores["ngram_max_match"] >= self.settings.threshold_ngram_max_match or
        scores["embed_top1"] >= self.settings.threshold_embed_top1
    )
    
    return should_fallback, scores
```

---

## 🎨 Prompt Templates

### System Prompt (Style Card):

```
You are a scientific paper rewriter. Rewrite the user's paragraph in a style consistent with Gilles's published papers.

HARD CONSTRAINTS (MUST follow):
- Preserve meaning EXACTLY: do not change numbers, units, experimental conditions, comparisons, or conclusion strength
- Do not introduce new facts not present in the input
- Do not strengthen hedged claims (may/suggest → demonstrate) or weaken strong claims
- Avoid copying long phrases (>8 words) from reference corpus

Output ONLY the rewritten paragraph, no explanations.
```

### Fallback Prompt (Stronger Diversification):

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

### Strategy Templates:

```
Strategy A: Rewrite with the main claim in the FIRST sentence, then provide supporting details and qualifiers.

Strategy B: Rewrite starting with brief context/motivation, then introduce the main claim, then qualifiers.

Strategy C: Rewrite starting with the experimental setup/approach, then report the key finding, then interpretation.

Strategy D: Rewrite starting with cautious framing/limitations, then state the key claim conservatively, then implications.
```

---

## 📊 Similarity Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| N-gram max match | ≥ 12 tokens | Trigger fallback |
| N-gram overlap (8-gram) | ≥ 0.15 | Trigger fallback |
| Embedding top-1 cosine | ≥ 0.88 | Trigger fallback |

---

## 🔄 Development Workflow

```
1. Read SPEC.md and TASKS.md
2. Implement PR in order
3. Write tests
4. Run self-review checklist
5. Test locally
6. Commit with descriptive message
7. Move to next PR
```

### Commit Message Format:

```
[PR#X] Brief description

- Detail 1
- Detail 2

Checklist:
- [x] Security validated
- [x] Tests pass
- [x] No external calls
```

---

## 🆘 Troubleshooting

### vLLM not responding:
```bash
# Check if vLLM is running
curl http://localhost:8000/v1/models

# Start vLLM
./scripts/start_vllm.sh
```

### Import errors:
```bash
# Reinstall package
pip install -e ".[dev,similarity]"
```

### Similarity service not working:
```bash
# Build index first
python scripts/build_index.py

# Check corpus files exist
ls -la data/corpus/parsed/
```

---

## 📚 Key Files Reference

```
src/gswa/
├── config.py           # Configuration (security defaults)
├── main.py             # FastAPI app entry
├── api/
│   ├── routes.py       # API endpoints
│   └── schemas.py      # Request/Response models
├── services/
│   ├── llm_client.py   # vLLM client
│   ├── similarity.py   # Similarity checking
│   ├── prompt.py       # Prompt construction
│   └── rewriter.py     # Main orchestration
└── utils/
    ├── ngram.py        # N-gram functions
    └── embedding.py    # Embedding functions
```

---

## ✅ Definition of Done (MVP)

- [ ] All 11 PRs implemented
- [ ] `make test` passes
- [ ] `make smoke-test` passes
- [ ] Web UI functional
- [ ] No external API calls
- [ ] Numbers preserved in rewrites
- [ ] Fallback triggers correctly
- [ ] Documentation complete
