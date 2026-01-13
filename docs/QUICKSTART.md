# GSWA Quick Start Guide for Claude Code

## 🎯 Your Mission

Build a **local offline** scientific paper rewriter that:
1. Generates 3-5 rewrite variants per input
2. Detects similarity to corpus (n-gram + embedding)
3. Auto-regenerates if too similar (fallback)
4. **NEVER** makes external API calls

---

## 📋 Step-by-Step Instructions

### Step 1: Create Project Structure

```bash
mkdir -p gswa/src/gswa/{api,services,utils}
mkdir -p gswa/{data/{corpus/{raw,parsed},index},web,scripts,tests,docs,logs}
cd gswa
```

### Step 2: Implement in Order

| Order | File | What to Build |
|-------|------|---------------|
| 1 | `pyproject.toml` | Package config |
| 2 | `src/gswa/config.py` | Settings with security defaults |
| 3 | `src/gswa/api/schemas.py` | Pydantic models |
| 4 | `src/gswa/utils/ngram.py` | N-gram overlap detection |
| 5 | `src/gswa/utils/embedding.py` | Vector similarity |
| 6 | `src/gswa/services/similarity.py` | Combined similarity gate |
| 7 | `src/gswa/services/llm_client.py` | vLLM client (local only!) |
| 8 | `src/gswa/services/prompt.py` | Prompt templates |
| 9 | `src/gswa/services/rewriter.py` | Orchestration logic |
| 10 | `src/gswa/api/routes.py` | API endpoints |
| 11 | `src/gswa/main.py` | FastAPI app |
| 12 | `web/*.html,css,js` | Simple UI |
| 13 | `scripts/smoke_test.py` | End-to-end test |

### Step 3: Key Code Snippets

#### Security Check (REQUIRED in config.py)
```python
class Settings(BaseSettings):
    allow_external_api: bool = False  # HARDCODED default
    
    def validate_security(self):
        if self.allow_external_api:
            raise ValueError("External API FORBIDDEN")
```

#### URL Validation (REQUIRED in llm_client.py)
```python
def _validate_local_only(self):
    from urllib.parse import urlparse
    parsed = urlparse(self._base_url)
    if parsed.hostname not in ["localhost", "127.0.0.1", "0.0.0.0"]:
        raise ValueError(f"External host forbidden: {parsed.hostname}")
```

#### Similarity Gate (REQUIRED in similarity.py)
```python
def check_similarity(candidate: str) -> Tuple[bool, dict]:
    ngram_match = compute_ngram_overlap(candidate, corpus_ngrams)
    embed_sim = compute_embed_similarity(candidate, faiss_index)
    
    should_fallback = (
        ngram_match["max_consecutive_match"] >= 12 or
        embed_sim["top1_similarity"] >= 0.88
    )
    return should_fallback, {**ngram_match, **embed_sim}
```

### Step 4: Test Your Work

```bash
# Run unit tests
pytest tests/ -v

# Start server
uvicorn gswa.main:app --reload --port 8080

# Run smoke test
python scripts/smoke_test.py
```

---

## ⚠️ Critical Constraints

### DO:
- ✅ Use `http://localhost:8000` for vLLM
- ✅ Validate all URLs are local
- ✅ Preserve numbers in rewrites
- ✅ Trigger fallback on high similarity
- ✅ Log only hashes, not original text

### DON'T:
- ❌ Call external APIs (OpenAI, Anthropic, etc.)
- ❌ Send telemetry
- ❌ Change numbers or conclusion strength
- ❌ Log sensitive content

---

## 📚 Reference Documents

For complete details, read:
- `docs/SPEC.md` - Full specification
- `docs/TASKS.md` - Detailed task breakdown
- `docs/PROMPTS.md` - Prompt templates
- `docs/API.md` - API documentation

---

## ✅ Definition of Done

- [ ] `/v1/rewrite/variants` returns 3 variants
- [ ] Similarity scores included in response
- [ ] Fallback triggers when similarity high
- [ ] Numbers preserved in output
- [ ] No external API calls
- [ ] `make smoke-test` passes
