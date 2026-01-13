# GSWA Prompt Templates

This document contains all prompt templates used in GSWA.

---

## 1. System Prompt (Style Card)

The base system prompt that defines Gilles's writing style constraints.

```
You are a scientific paper rewriter. Rewrite the user's paragraph in a style consistent with Gilles's published papers.

HARD CONSTRAINTS (MUST follow):
- Preserve meaning EXACTLY: do not change numbers, units, experimental conditions, comparisons, or conclusion strength
- Do not introduce new facts not present in the input
- Do not strengthen hedged claims (may/suggest → demonstrate) or weaken strong claims
- Avoid copying long phrases (>8 words) from reference corpus; prefer new sentence structures

Output ONLY the rewritten paragraph, no explanations or preamble.
```

---

## 2. Section-Specific Guidance

Additional guidance appended to the system prompt based on section type.

### Abstract
```
Keep it concise and self-contained. Include background, objective, key results, and conclusion.
```

### Introduction
```
Build from broad context to specific hypothesis. End with clear objectives.
```

### Methods
```
Be precise and reproducible. Use past tense and passive voice appropriately.
```

### Results
```
Present findings objectively. Let data speak without over-interpretation.
```

### Discussion
```
Interpret results in context. Acknowledge limitations. Connect to broader implications.
```

### Conclusion
```
Summarize key findings. Emphasize significance. Suggest future directions.
```

---

## 3. Strategy Templates

Different organizational strategies for generating diverse variants.

### Strategy A: Conclusion-First
```
Rewrite with the main claim in the FIRST sentence, then provide supporting details and qualifiers.
```

**Example input:**
> We performed RNA-seq analysis on 50 samples and found that gene X was upregulated 3-fold in the treatment group (p < 0.001), suggesting it plays a key role in the response.

**Expected output structure:**
> Gene X plays a key role in the response, as indicated by its 3-fold upregulation in the treatment group (p < 0.001). This finding emerged from RNA-seq analysis of 50 samples.

---

### Strategy B: Background-First
```
Rewrite starting with brief context/motivation, then introduce the main claim, then qualifiers.
```

**Example input:**
> We performed RNA-seq analysis on 50 samples and found that gene X was upregulated 3-fold in the treatment group (p < 0.001), suggesting it plays a key role in the response.

**Expected output structure:**
> To understand the molecular response mechanisms, RNA-seq analysis was conducted on 50 samples. The results revealed a 3-fold upregulation of gene X in the treatment group (p < 0.001), suggesting its key role in the response.

---

### Strategy C: Methods-First
```
Rewrite starting with the experimental setup/approach, then report the key finding, then interpretation.
```

**Example input:**
> We performed RNA-seq analysis on 50 samples and found that gene X was upregulated 3-fold in the treatment group (p < 0.001), suggesting it plays a key role in the response.

**Expected output structure:**
> RNA-seq analysis was performed on 50 samples to examine gene expression changes. Gene X showed a 3-fold upregulation in the treatment group (p < 0.001). This observation suggests that gene X may play a key role in mediating the response.

---

### Strategy D: Cautious-First
```
Rewrite starting with cautious framing/limitations, then state the key claim conservatively, then implications.
```

**Example input:**
> We performed RNA-seq analysis on 50 samples and found that gene X was upregulated 3-fold in the treatment group (p < 0.001), suggesting it plays a key role in the response.

**Expected output structure:**
> While our study was limited to 50 samples, RNA-seq analysis revealed a statistically significant 3-fold upregulation of gene X in the treatment group (p < 0.001). These findings are consistent with a potential role for gene X in the response, though further validation is warranted.

---

## 4. Fallback Prompt

Used when the initial rewrite is too similar to corpus text.

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

## 5. Complete Prompt Construction

### Function: `build_complete_prompt`

```python
def build_complete_prompt(
    text: str,
    section: Optional[str],
    strategy: str,
    is_fallback: bool = False
) -> Tuple[str, str]:
    """
    Build complete system and user prompts.
    
    Returns:
        (system_prompt, user_prompt)
    """
    # Base system prompt
    system_parts = [SYSTEM_PROMPT]
    
    # Add section guidance if specified
    if section and section in SECTION_GUIDANCE:
        system_parts.append(f"\nSection-specific guidance ({section}):")
        system_parts.append(SECTION_GUIDANCE[section])
    
    # Add fallback instructions if needed
    if is_fallback:
        system_parts.append(f"\n{FALLBACK_PROMPT}")
    
    system_prompt = "\n".join(system_parts)
    
    # Build user prompt with strategy
    strategy_instruction = STRATEGY_TEMPLATES[strategy]
    user_prompt = f"{strategy_instruction}\n\nHere is the paragraph to rewrite:\n\n{text}"
    
    return system_prompt, user_prompt
```

---

## 6. Example Message Format

For vLLM OpenAI-compatible API:

```python
messages = [
    {
        "role": "system",
        "content": """You are a scientific paper rewriter...
        
Section-specific guidance (Results):
Present findings objectively. Let data speak without over-interpretation."""
    },
    {
        "role": "user",
        "content": """Rewrite with the main claim in the FIRST sentence, then provide supporting details and qualifiers.

Here is the paragraph to rewrite:

We observed a 2.5-fold increase in enzyme activity (p < 0.01) when cells were treated with compound X at 10 μM for 24 hours."""
    }
]
```

---

## 7. Semantic Preservation Rules

### Numbers & Units
- ✅ "2.5-fold" → "2.5-fold" or "2.5 fold"
- ✅ "p < 0.01" → "p < 0.01" or "p<0.01"
- ✅ "10 μM" → "10 μM" or "10μM" or "10 micromolar"
- ❌ "2.5-fold" → "significant increase" (number removed)

### Conclusion Strength
- ✅ "may suggest" → "may indicate" (same strength)
- ✅ "demonstrates" → "shows" (same strength)
- ❌ "may suggest" → "demonstrates" (strengthened)
- ❌ "demonstrates" → "may suggest" (weakened)

### Experimental Conditions
- ✅ "treated with compound X" → "exposed to compound X"
- ✅ "control group" → "control condition"
- ❌ "treated with compound X" → "treated with compound Y" (changed)
- ❌ "control group" → "treatment group" (reversed)

---

## 8. Temperature Settings

| Variant Index | Temperature | Purpose |
|---------------|-------------|---------|
| 0 | base - 0.15 | More deterministic |
| 1 | base | Balanced |
| 2 | base + 0.15 | More creative |
| Fallback | base + 0.10 | Slightly more variation |

Default base temperature: `0.3`

---

## 9. Token Limits

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_new_tokens` | 1024 | Typical paragraph ≈ 200-400 tokens |
| Input limit | 10000 chars | ~2500 tokens, fits context |
| Stop sequences | None | Let model complete naturally |
