#!/usr/bin/env python3
"""Test the Anthropic Claude API pipeline end-to-end.

Usage:
    # Set API key first:
    export API_KEY=sk-ant-...

    # Run standalone (no server needed):
    PYTHONPATH=src python scripts/test_api_pipeline.py

    # Or test via running FastAPI server:
    PYTHONPATH=src python scripts/test_api_pipeline.py --server http://localhost:8000
"""
import asyncio
import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


TEST_PARAGRAPHS = [
    # Paragraph 1: Streptomyces biology
    "Streptomyces are Gram-positive filamentous bacteria that undergo a complex developmental life cycle. They produce aerial hyphae that differentiate into chains of spores. The genome of Streptomyces coelicolor encodes over 20 secondary metabolite gene clusters. These organisms are the source of approximately two-thirds of all known antibiotics. The regulation of antibiotic production is tightly linked to morphological development.",

    # Paragraph 2: Cell signaling
    "The Wnt signaling pathway plays a crucial role in embryonic development and tissue homeostasis. Aberrant activation of this pathway has been implicated in various types of cancer, including colorectal carcinoma. Beta-catenin acts as a key mediator of canonical Wnt signaling. In the absence of Wnt ligands, beta-catenin is phosphorylated and targeted for proteasomal degradation. Upon Wnt stimulation, beta-catenin accumulates in the cytoplasm and translocates to the nucleus.",

    # Paragraph 3: Proteomics
    "Mass spectrometry-based proteomics has emerged as a powerful tool for the comprehensive analysis of protein expression in complex biological samples. Recent advances in instrumentation and data analysis have significantly improved the depth and accuracy of proteomic measurements. Data-independent acquisition strategies enable reproducible quantification of thousands of proteins across large sample cohorts. Integration of proteomics with transcriptomics data provides complementary information about gene expression regulation.",
]


async def test_standalone():
    """Test the API pipeline without a running server."""
    # Clear cached settings (will load from .env)
    from gswa.config import get_settings
    get_settings.cache_clear()

    from gswa.config import Settings
    settings = Settings()

    if not settings.api_key:
        print("ERROR: API_KEY not set — add it to .env: API_KEY=sk-ant-...")
        sys.exit(1)
    print(f"Backend: {settings.llm_backend}")
    print(f"Model:   {settings.api_model}")
    print(f"API Key: {settings.api_key[:12]}...")
    print()

    from gswa.services.api_client import AnthropicClient
    client = AnthropicClient()

    # Health check
    health = await client.check_health()
    print(f"Health: {health['status']}")
    if health['status'] != 'connected':
        print(f"  Error: {health.get('error')}")
        sys.exit(1)
    print()

    from gswa.utils.ai_detector import get_ai_detector, correct_ai_traces, calculate_burstiness
    from gswa.services.prompt import get_prompt_service
    from gswa.api.schemas import Strategy

    detector = get_ai_detector()
    prompt_service = get_prompt_service()

    for idx, para in enumerate(TEST_PARAGRAPHS):
        print(f"{'='*60}")
        print(f"Paragraph {idx+1} ({len(para.split())} words)")
        print(f"{'='*60}")

        # Input metrics
        input_cv = calculate_burstiness(para)
        input_ai = detector.detect(para)
        print(f"Input:  CV={input_cv:.3f}  AI={input_ai.ai_score:.3f}")

        # Generate 3 variants at different temperatures
        strategies = [Strategy.A, Strategy.B, Strategy.C]
        temps = [0.15, 0.30, 0.45]

        for i, (strat, temp) in enumerate(zip(strategies, temps)):
            system_prompt = prompt_service.build_system_prompt(
                section=None,
                is_fallback=False,
                include_anti_ai=True,
                include_style=True,
            )
            user_prompt = prompt_service.build_user_prompt(
                text=para,
                strategy=strat,
                variant_index=i,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            result = await client.complete(
                messages=messages,
                temperature=temp,
                max_tokens=1024,
            )

            # Post-process
            corrected = correct_ai_traces(result)

            # Metrics
            cv = calculate_burstiness(corrected)
            ai_result = detector.detect(corrected)

            quality = 0.5 * cv + 0.5 * (1.0 - ai_result.ai_score)

            print(f"\n  Variant {i+1} (strategy={strat.value}, temp={temp}):")
            print(f"    CV={cv:.3f}  AI={ai_result.ai_score:.3f}  Quality={quality:.3f}")
            print(f"    Words: {len(corrected.split())}")
            if ai_result.pattern_issues:
                print(f"    Issues: {', '.join(ai_result.pattern_issues[:3])}")
            # Show first 150 chars
            preview = corrected[:150].replace('\n', ' ')
            print(f"    Preview: {preview}...")

        print()


async def test_server(base_url: str):
    """Test via running FastAPI server."""
    import httpx

    print(f"Testing server at {base_url}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Health
        resp = await client.get(f"{base_url}/v1/health")
        print(f"Health: {resp.json()}")
        print()

        for idx, para in enumerate(TEST_PARAGRAPHS[:1]):  # Just test first paragraph
            print(f"Paragraph {idx+1}: {para[:80]}...")

            resp = await client.post(
                f"{base_url}/v1/rewrite/variants",
                json={
                    "text": para,
                    "n_variants": 3,
                }
            )
            data = resp.json()

            for i, v in enumerate(data["variants"]):
                scores = v["scores"]
                print(f"\n  Variant {i+1} (strategy={v['strategy']}):")
                print(f"    AI={scores.get('ai_score', 'N/A')}  Fallback={v['fallback']}")
                print(f"    Text: {v['text'][:150]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Anthropic API pipeline")
    parser.add_argument("--server", type=str, help="Server URL (e.g. http://localhost:8000)")
    args = parser.parse_args()

    if args.server:
        asyncio.run(test_server(args.server))
    else:
        asyncio.run(test_standalone())
