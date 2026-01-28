#!/usr/bin/env python3
"""
Compare two GSWA models on unseen text using style metrics.

This script evaluates how well each model captures Gilles's writing style
by analyzing linguistic features of the generated text.

Usage:
    python scripts/compare_models.py \
        models/gswa-lora-Mistral-20260126-131024-3ep \
        models/gswa-lora-Mistral-20260126-131024 \
        --num-samples 5
"""
import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter


# Novel test prompts - generic scientific text NOT from Gilles's papers
NOVEL_TEST_PROMPTS = [
    {
        "instruction": "Rewrite this paragraph in Gilles van Wezel's academic style:",
        "input": """The bacteria grew faster when we added more nutrients to the medium.
We saw that the cells divided every 30 minutes under optimal conditions.
When we removed the carbon source, growth stopped completely.
These results show that nutrients are important for bacterial growth.""",
    },
    {
        "instruction": "Edit this scientific text to match the writing style of a Nature Microbiology paper:",
        "input": """We found that the gene was expressed at high levels in the mutant strain.
The protein localized to the cell membrane. Western blot analysis confirmed
the presence of the protein. Deletion of the gene resulted in a growth defect.
The complemented strain showed normal growth.""",
    },
    {
        "instruction": "Polish this research paragraph for publication in a high-impact journal:",
        "input": """The enzyme activity was measured using a spectrophotometric assay.
We observed a linear increase in activity with substrate concentration.
The Km value was determined to be 2.5 mM. Inhibitor studies showed that
the enzyme was sensitive to metal chelators. This suggests that the enzyme
requires metal ions for activity.""",
    },
    {
        "instruction": "Enhance this text with more sophisticated academic phrasing:",
        "input": """Microscopy showed that the cells changed shape during stress conditions.
Normal cells were rod-shaped but stressed cells became round. The cell wall
appeared to be damaged in stressed cells. Adding osmotic stabilizers prevented
the shape change. This indicates that the cell wall is important for maintaining
cell shape.""",
    },
    {
        "instruction": "Rewrite this methods description in formal scientific prose:",
        "input": """We grew the bacteria in LB medium at 37 degrees overnight. The next day,
we diluted the culture 1:100 and grew it until OD600 reached 0.5. Then we
induced gene expression by adding IPTG. After 4 hours we collected the cells
and extracted the protein.""",
    },
]


def calculate_style_metrics(text: str) -> dict:
    """Calculate linguistic style metrics for a text.

    Returns metrics that characterize Gilles's writing style:
    - Sentence complexity
    - Vocabulary richness
    - Use of discourse markers
    - Academic hedging
    - Subordination patterns
    """
    if not text or len(text) < 50:
        return {"error": "Text too short"}

    # Tokenize into sentences and words
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    if not sentences or not words:
        return {"error": "Could not parse text"}

    metrics = {}

    # 1. Sentence length statistics
    sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    metrics["avg_sentence_length"] = sum(sent_lengths) / len(sent_lengths)
    metrics["max_sentence_length"] = max(sent_lengths)
    metrics["sentence_length_variance"] = (
        sum((l - metrics["avg_sentence_length"])**2 for l in sent_lengths) / len(sent_lengths)
    ) if len(sent_lengths) > 1 else 0

    # 2. Vocabulary richness (Type-Token Ratio)
    metrics["vocabulary_richness"] = len(set(words)) / len(words) if words else 0

    # 3. Discourse markers (Gilles uses these frequently)
    discourse_markers = [
        'indeed', 'notably', 'interestingly', 'importantly', 'remarkably',
        'strikingly', 'intriguingly', 'surprisingly', 'unexpectedly',
        'furthermore', 'moreover', 'however', 'nevertheless', 'conversely',
        'accordingly', 'consequently', 'thus', 'hence', 'therefore',
        'specifically', 'particularly', 'especially', 'significantly'
    ]
    marker_count = sum(1 for w in words if w in discourse_markers)
    metrics["discourse_marker_density"] = marker_count / len(words) * 100
    metrics["discourse_markers_found"] = [w for w in words if w in discourse_markers]

    # 4. Subordination markers (complex sentence structure)
    subordinators = [
        'which', 'that', 'who', 'whom', 'whose', 'where', 'when', 'while',
        'although', 'whereas', 'whereby', 'wherein', 'because', 'since',
        'unless', 'until', 'after', 'before', 'if', 'whether'
    ]
    sub_count = sum(1 for w in words if w in subordinators)
    metrics["subordination_density"] = sub_count / len(words) * 100

    # 5. Academic hedging (tentative language)
    hedging_words = [
        'may', 'might', 'could', 'would', 'possibly', 'potentially',
        'likely', 'unlikely', 'perhaps', 'presumably', 'apparently',
        'suggests', 'indicates', 'appears', 'seems'
    ]
    hedge_count = sum(1 for w in words if w in hedging_words)
    metrics["hedging_density"] = hedge_count / len(words) * 100

    # 6. Passive voice indicators
    passive_indicators = ['was', 'were', 'been', 'being', 'is', 'are']
    passive_verbs = ['observed', 'found', 'shown', 'demonstrated', 'reported',
                     'determined', 'identified', 'detected', 'measured', 'analyzed']
    # Simple heuristic: count "was/were + past participle" patterns
    passive_count = 0
    for i, w in enumerate(words[:-1]):
        if w in passive_indicators and words[i+1] in passive_verbs:
            passive_count += 1
    metrics["passive_voice_indicators"] = passive_count

    # 7. Scientific precision markers
    precision_words = [
        'specifically', 'exclusively', 'predominantly', 'primarily',
        'essentially', 'virtually', 'substantially', 'considerably',
        'markedly', 'drastically', 'dramatically', 'profoundly'
    ]
    precision_count = sum(1 for w in words if w in precision_words)
    metrics["precision_marker_density"] = precision_count / len(words) * 100

    # 8. Transition phrases (multi-word)
    text_lower = text.lower()
    transition_phrases = [
        'taken together', 'in contrast', 'in addition', 'as a result',
        'on the other hand', 'in this context', 'to this end',
        'of particular interest', 'it is noteworthy', 'it should be noted'
    ]
    transition_count = sum(1 for p in transition_phrases if p in text_lower)
    metrics["transition_phrases"] = transition_count

    # 9. Word count
    metrics["word_count"] = len(words)
    metrics["sentence_count"] = len(sentences)

    return metrics


def compute_style_score(metrics: dict, reference_metrics: dict = None) -> float:
    """Compute an overall style score based on metrics.

    Higher score = more similar to Gilles's style.

    Reference values are based on analysis of Gilles's papers:
    - Average sentence length: 20-30 words
    - High discourse marker usage
    - Complex subordination
    - Moderate hedging
    """
    if "error" in metrics:
        return 0.0

    score = 0.0
    max_score = 100.0

    # 1. Sentence length (target: 20-30 words average)
    avg_len = metrics.get("avg_sentence_length", 0)
    if 20 <= avg_len <= 30:
        score += 20
    elif 15 <= avg_len <= 35:
        score += 10
    elif avg_len > 10:
        score += 5

    # 2. Discourse markers (target: > 0.5% density)
    dm_density = metrics.get("discourse_marker_density", 0)
    if dm_density >= 1.0:
        score += 20
    elif dm_density >= 0.5:
        score += 15
    elif dm_density >= 0.2:
        score += 10
    elif dm_density > 0:
        score += 5

    # 3. Subordination (target: > 3% density)
    sub_density = metrics.get("subordination_density", 0)
    if sub_density >= 4.0:
        score += 20
    elif sub_density >= 3.0:
        score += 15
    elif sub_density >= 2.0:
        score += 10
    elif sub_density > 0:
        score += 5

    # 4. Hedging (target: 0.5-2% - balanced)
    hedge = metrics.get("hedging_density", 0)
    if 0.5 <= hedge <= 2.0:
        score += 15
    elif 0.2 <= hedge <= 3.0:
        score += 10
    elif hedge > 0:
        score += 5

    # 5. Precision markers
    precision = metrics.get("precision_marker_density", 0)
    if precision >= 0.5:
        score += 15
    elif precision >= 0.2:
        score += 10
    elif precision > 0:
        score += 5

    # 6. Transition phrases
    trans = metrics.get("transition_phrases", 0)
    if trans >= 2:
        score += 10
    elif trans >= 1:
        score += 5

    return min(score, max_score)


def load_model(model_dir: Path, device: str = "cuda:0"):
    """Load base model + LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    config_file = model_dir / "training_config.json"
    if not config_file.exists():
        print(f"ERROR: training_config.json not found in {model_dir}")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    base_model_name = config["base_model"]
    quantize = config.get("quantization", "4bit")

    print(f"  Loading {model_dir.name}...")

    bnb_config = None
    if quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": device} if bnb_config else device,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, str(model_dir))
    model.eval()

    return model, tokenizer, config


def generate_text(model, tokenizer, prompt_data: dict, max_new_tokens: int = 300) -> str:
    """Generate text for a prompt."""
    import torch

    user_content = prompt_data['instruction']
    if prompt_data.get("input"):
        user_content += f"\n\n{prompt_data['input']}"

    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"### Instruction:\n{user_content}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    return generated


def main():
    parser = argparse.ArgumentParser(description="Compare two GSWA models on style metrics")
    parser.add_argument("model1", type=Path, help="First model directory")
    parser.add_argument("model2", type=Path, help="Second model directory")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of test samples")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference")
    parser.add_argument("--output", type=Path, default=None, help="Output file for results")

    args = parser.parse_args()

    print("=" * 70)
    print("GSWA Model Style Comparison")
    print("=" * 70)
    print(f"\nModel 1: {args.model1.name}")
    print(f"Model 2: {args.model2.name}")
    print(f"Test samples: {args.num_samples}")
    print()

    # Get test prompts
    prompts = NOVEL_TEST_PROMPTS[:args.num_samples]

    # Load models
    print("Loading models...")
    model1, tok1, cfg1 = load_model(args.model1, args.device)

    # Generate with model 1
    print(f"\nGenerating with {args.model1.name}...")
    results1 = []
    for i, p in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}]", end=" ", flush=True)
        text = generate_text(model1, tok1, p)
        metrics = calculate_style_metrics(text)
        score = compute_style_score(metrics)
        results1.append({"text": text, "metrics": metrics, "score": score})
        print(f"score={score:.1f}")

    # Free memory
    del model1
    import torch
    torch.cuda.empty_cache()

    # Load and generate with model 2
    model2, tok2, cfg2 = load_model(args.model2, args.device)

    print(f"\nGenerating with {args.model2.name}...")
    results2 = []
    for i, p in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}]", end=" ", flush=True)
        text = generate_text(model2, tok2, p)
        metrics = calculate_style_metrics(text)
        score = compute_style_score(metrics)
        results2.append({"text": text, "metrics": metrics, "score": score})
        print(f"score={score:.1f}")

    del model2
    torch.cuda.empty_cache()

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    avg1 = sum(r["score"] for r in results1) / len(results1)
    avg2 = sum(r["score"] for r in results2) / len(results2)

    print(f"\n{'Metric':<30} {args.model1.name:<20} {args.model2.name:<20}")
    print("-" * 70)

    # Average scores
    print(f"{'Average Style Score':<30} {avg1:<20.1f} {avg2:<20.1f}")

    # Individual metric averages
    metric_names = [
        "avg_sentence_length",
        "discourse_marker_density",
        "subordination_density",
        "hedging_density",
        "precision_marker_density",
        "transition_phrases"
    ]

    for metric in metric_names:
        val1 = sum(r["metrics"].get(metric, 0) for r in results1) / len(results1)
        val2 = sum(r["metrics"].get(metric, 0) for r in results2) / len(results2)
        print(f"{metric:<30} {val1:<20.2f} {val2:<20.2f}")

    print("-" * 70)

    # Winner
    if avg1 > avg2:
        winner = args.model1.name
        margin = avg1 - avg2
    elif avg2 > avg1:
        winner = args.model2.name
        margin = avg2 - avg1
    else:
        winner = "TIE"
        margin = 0

    print(f"\nWinner: {winner}")
    if margin > 0:
        print(f"Margin: +{margin:.1f} points")

    # Sample outputs for inspection
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS (for manual inspection)")
    print("=" * 70)

    for i, (r1, r2, p) in enumerate(zip(results1, results2, prompts), 1):
        print(f"\n--- Sample {i} ---")
        print(f"Input: {p['input'][:100]}...")
        print(f"\n[{args.model1.name}] (score={r1['score']:.1f}):")
        print(f"  {r1['text'][:300]}...")
        print(f"\n[{args.model2.name}] (score={r2['score']:.1f}):")
        print(f"  {r2['text'][:300]}...")

    # Save detailed results
    if args.output:
        output_data = {
            "model1": str(args.model1),
            "model2": str(args.model2),
            "timestamp": datetime.now().isoformat(),
            "avg_score_model1": avg1,
            "avg_score_model2": avg2,
            "winner": winner,
            "results_model1": results1,
            "results_model2": results2,
            "prompts": prompts,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
