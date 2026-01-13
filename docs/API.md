# GSWA API Documentation

## Base URL

```
http://localhost:8080/v1
```

---

## Endpoints

### 1. POST /v1/rewrite/variants

Generate multiple rewrite variants of a paragraph.

#### Request

```json
{
  "text": "string (required, 10-10000 chars)",
  "section": "string (optional): Abstract|Introduction|Methods|Results|Discussion|Conclusion",
  "n_variants": "integer (1-5, default 3)",
  "strategies": "array (optional): subset of [A, B, C, D]",
  "constraints": {
    "preserve_numbers": "boolean (default true)",
    "no_new_facts": "boolean (default true)"
  }
}
```

#### Response

```json
{
  "variants": [
    {
      "text": "string - rewritten paragraph",
      "strategy": "string - A|B|C|D",
      "scores": {
        "ngram_max_match": "integer - longest consecutive n-gram match",
        "ngram_overlap": "float - overlap ratio",
        "embed_top1": "float - cosine similarity to most similar corpus paragraph",
        "top1_doc_id": "string|null",
        "top1_para_id": "string|null"
      },
      "fallback": "boolean - whether fallback was triggered",
      "fallback_reason": "string|null"
    }
  ],
  "model_version": "string",
  "processing_time_ms": "integer"
}
```

#### Example

```bash
curl -X POST http://localhost:8080/v1/rewrite/variants \
  -H "Content-Type: application/json" \
  -d '{
    "text": "We observed a 2.5-fold increase in enzyme activity (p < 0.01) when cells were treated with compound X at 10 μM for 24 hours.",
    "section": "Results",
    "n_variants": 3
  }'
```

---

### 2. POST /v1/reply

Simple chat endpoint for testing.

#### Request

```json
{
  "messages": [
    {
      "role": "string: user|assistant|system",
      "content": "string"
    }
  ],
  "max_tokens": "integer (1-4096, default 512)"
}
```

#### Response

```json
{
  "content": "string",
  "model": "string"
}
```

#### Example

```bash
curl -X POST http://localhost:8080/v1/reply \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

---

### 3. GET /v1/health

Check system health.

#### Response

```json
{
  "status": "string: healthy|degraded|error",
  "llm_server": "string: connected|disconnected|error",
  "model_loaded": "string|null",
  "corpus_paragraphs": "integer",
  "index_loaded": "boolean"
}
```

#### Example

```bash
curl http://localhost:8080/v1/health
```

---

## Strategy Descriptions

| Strategy | Description |
|----------|-------------|
| A | **Conclusion-first**: Main claim in first sentence, then supporting details |
| B | **Background-first**: Context/motivation, then main claim, then qualifiers |
| C | **Methods-first**: Experimental setup, then key finding, then interpretation |
| D | **Cautious-first**: Limitations/framing, then conservative claim, then implications |

---

## Similarity Scores

| Score | Description | Fallback Threshold |
|-------|-------------|-------------------|
| `ngram_max_match` | Longest consecutive n-gram match with corpus | ≥ 12 tokens |
| `ngram_overlap` | 8-gram Jaccard similarity | ≥ 0.15 |
| `embed_top1` | Cosine similarity to most similar corpus paragraph | ≥ 0.88 |

When any threshold is exceeded, the system automatically regenerates with a stronger diversification prompt.

---

## Error Responses

```json
{
  "detail": "string - error message"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Invalid request (validation error) |
| 500 | Internal server error |
| 503 | LLM server unavailable |

---

## Rate Limits

No rate limits for local deployment. For shared deployments, implement rate limiting at the reverse proxy level.

---

## Security Notes

1. **No external API calls**: All processing is local
2. **No data logging**: Original text is not logged (only hashes if needed)
3. **CORS**: Configured for local development only
4. **Authentication**: Not implemented (add reverse proxy auth for shared deployments)
