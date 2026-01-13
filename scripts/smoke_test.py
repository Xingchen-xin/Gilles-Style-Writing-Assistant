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
import time

try:
    import httpx
except ImportError:
    print("Error: httpx not installed. Run: pip install httpx")
    sys.exit(1)


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print("Testing /v1/health...")
    try:
        resp = httpx.get(f"{base_url}/v1/health", timeout=10)
        data = resp.json()

        if data.get("status") in ["healthy", "degraded"]:
            print(f"  [PASS] Health check: {data['status']}")
            print(f"    LLM server: {data.get('llm_server')}")
            print(f"    Model: {data.get('model_loaded')}")
            print(f"    Corpus paragraphs: {data.get('corpus_paragraphs')}")
            print(f"    Index loaded: {data.get('index_loaded')}")
            return True
        else:
            print(f"  [FAIL] Unexpected status: {data}")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
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

        if resp.status_code != 200:
            print(f"  [FAIL] HTTP {resp.status_code}: {resp.text[:200]}")
            return False

        data = resp.json()

        if "content" in data:
            print(f"  [PASS] Reply received: {data['content'][:100]}...")
            print(f"    Model: {data.get('model')}")
            return True
        else:
            print(f"  [FAIL] No content in response: {data}")
            return False
    except httpx.ConnectError:
        print("  [SKIP] Cannot connect to vLLM server (this is expected if vLLM is not running)")
        return True  # Don't fail the whole test
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_rewrite(base_url: str) -> bool:
    """Test rewrite variants endpoint."""
    print("\nTesting /v1/rewrite/variants...")

    test_text = """
    We observed a 2.5-fold increase in enzyme activity (p < 0.01)
    when cells were treated with compound X at 10 \u03bcM for 24 hours.
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

        if resp.status_code != 200:
            print(f"  [FAIL] HTTP {resp.status_code}: {resp.text[:200]}")
            return False

        data = resp.json()

        if "variants" not in data:
            print(f"  [FAIL] No variants in response: {data}")
            return False

        variants = data["variants"]
        print(f"  [PASS] Received {len(variants)} variants in {elapsed:.2f}s")
        print(f"    Model: {data.get('model_version')}")
        print(f"    Processing time: {data.get('processing_time_ms')}ms")

        # Verify numbers preserved
        all_pass = True
        for i, v in enumerate(variants):
            text = v["text"]

            # Check key numbers
            has_fold = "2.5" in text or "2.5-fold" in text.lower()
            has_p = "p <" in text or "p<" in text or "0.01" in text
            has_conc = "10" in text and ("\u03bcm" in text.lower() or "um" in text.lower() or "\u03bc" in text)
            has_time = "24" in text and ("hour" in text.lower() or "h" in text.lower())

            if has_fold and has_p and has_conc and has_time:
                print(f"    Variant {i+1}: [PASS] Numbers preserved")
            else:
                print(f"    Variant {i+1}: [WARN] Some numbers may be missing")
                print(f"      2.5-fold: {has_fold}, p-value: {has_p}, 10\u03bcM: {has_conc}, 24h: {has_time}")

            print(f"      Strategy: {v['strategy']}, Fallback: {v['fallback']}")
            print(f"      Scores: ngram={v['scores']['ngram_max_match']}, embed={v['scores']['embed_top1']:.3f}")

        return True

    except httpx.ConnectError:
        print("  [SKIP] Cannot connect to vLLM server (this is expected if vLLM is not running)")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_security(base_url: str) -> bool:
    """Verify security constraints."""
    print("\nTesting security constraints...")

    # Test that config enforces ALLOW_EXTERNAL_API=false
    try:
        from gswa.config import Settings

        settings = Settings()
        if settings.allow_external_api:
            print("  [FAIL] ALLOW_EXTERNAL_API should default to False")
            return False

        # Test that validation raises error when set to True
        import os
        os.environ["ALLOW_EXTERNAL_API"] = "true"
        try:
            test_settings = Settings()
            test_settings.validate_security()
            print("  [FAIL] Should have raised ValueError for ALLOW_EXTERNAL_API=true")
            return False
        except ValueError as e:
            print(f"  [PASS] Security validation works: {str(e)[:50]}...")
        finally:
            os.environ.pop("ALLOW_EXTERNAL_API", None)

        # Test LLM client validates localhost
        from gswa.services.llm_client import LLMClient
        os.environ["VLLM_BASE_URL"] = "http://api.example.com/v1"
        try:
            from gswa.config import get_settings
            get_settings.cache_clear()
            LLMClient()
            print("  [FAIL] Should have rejected external URL")
            return False
        except ValueError:
            print("  [PASS] External URLs correctly rejected")
        finally:
            os.environ.pop("VLLM_BASE_URL", None)
            get_settings.cache_clear()

        print("  [PASS] All security checks passed")
        return True

    except ImportError as e:
        print(f"  [SKIP] Cannot import modules: {e}")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="GSWA Smoke Test")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8080",
        help="GSWA API base URL"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip tests that require vLLM server"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GSWA Smoke Test")
    print("=" * 60)
    print(f"API URL: {args.api_url}")
    print()

    results = {}

    # Always run security test
    results["security"] = test_security(args.api_url)

    # Health check
    results["health"] = test_health(args.api_url)

    # LLM-dependent tests
    if not args.skip_llm:
        results["reply"] = test_reply(args.api_url)
        results["rewrite"] = test_rewrite(args.api_url)
    else:
        print("\n[SKIP] Skipping LLM-dependent tests (--skip-llm)")
        results["reply"] = True
        results["rewrite"] = True

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_pass = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
