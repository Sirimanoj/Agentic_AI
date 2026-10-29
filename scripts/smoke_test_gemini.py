"""Minimal optional live Gemini smoke test for Legal UQ.

Performs exactly ONE live API request only when an API key is explicitly provided via:
1. Environment variable: GEMINI_API_KEY (or GOOGLE_API_KEY)
2. CLI argument: --api-key <YOUR_KEY>

SAFETY & SECURITY GUARANTEES:
- NEVER run during pytest or automated test runs.
- NEVER prints, logs, or persists the API key.
- All errors are sanitized to redact credentials.
- When no key is provided, clearly reports NOT TESTED and exits cleanly.
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure src directory is in sys.path
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from legal_uq.generation.generator import (
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_GEMINI_MODELS,
    _sanitize_error_message,
)


def run_live_smoke_test(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> int:
    """Execute exactly one minimal prompt against the Gemini API.

    Returns:
        0 on success or cleanly skipped when key is missing,
        1 on live API failure.
    """
    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )

    if not resolved_key or not resolved_key.strip():
        print("=" * 70)
        print("LEGAL UQ - GEMINI LIVE API SMOKE TEST: SKIPPED (NOT TESTED)")
        print("=" * 70)
        print("Reason: No GEMINI_API_KEY or GOOGLE_API_KEY found.")
        print()
        print("To run a live verification call against Google Gemini:")
        print("  1. Obtain an API key from Google AI Studio:")
        print("     https://aistudio.google.com/app/apikey")
        print("  2. Set the key in your terminal session:")
        print("     Windows PowerShell: $env:GEMINI_API_KEY=\"your_key_here\"")
        print("     Linux/macOS:        export GEMINI_API_KEY=\"your_key_here\"")
        print("  3. Run this script:")
        print("     python scripts/smoke_test_gemini.py")
        print("     (Or pass: python scripts/smoke_test_gemini.py --api-key <KEY>)")
        print("=" * 70)
        return 0

    clean_key = resolved_key.strip()
    target_model = (
        model_name
        or os.environ.get("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    ).strip()

    print("=" * 70)
    print("LEGAL UQ - GEMINI LIVE API SMOKE TEST")
    print("=" * 70)
    print(f"Target Model:      {target_model}")
    print(f"Supported Models:  {', '.join(SUPPORTED_GEMINI_MODELS)}")
    print("API Key:           [PRESENT - REDACTED FOR SECURITY]")

    # Check SDK availability
    sdk_interface = "unknown"
    client = None
    try:
        from google import genai
        client = genai.Client(api_key=clean_key)
        if hasattr(client, "interactions") and hasattr(client.interactions, "create"):
            sdk_interface = "google.genai (Interactions API: client.interactions.create)"
        elif hasattr(client, "models") and hasattr(client.models, "generate_content"):
            sdk_interface = "google.genai (Models API: client.models.generate_content)"
        else:
            sdk_interface = "google.genai (Client instance)"
    except Exception as e:
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=clean_key)
            client = legacy_genai.GenerativeModel(target_model)
            sdk_interface = "google.generativeai (Legacy: GenerativeModel.generate_content)"
        except Exception as e_legacy:
            sanitized_e = _sanitize_error_message(f"{e} | {e_legacy}", clean_key)
            print(f"[FAILED] Could not initialize Gemini SDK: {sanitized_e}")
            return 1

    print(f"SDK Interface:     {sdk_interface}")
    print("Executing 1 minimal prompt (consumes quota)...")
    print("-" * 70)

    prompt = (
        "In exactly one clear, concise sentence, state the primary legal purpose "
        "of a Non-Disclosure Agreement (NDA)."
    )

    start_time = time.perf_counter()
    try:
        response_text = ""
        # 1. Interactions API (if supported)
        if hasattr(client, "interactions") and hasattr(client.interactions, "create"):
            resp = client.interactions.create(
                model=target_model,
                input=prompt,
                generation_config={"temperature": 0.2},
            )
            response_text = str(getattr(resp, "output_text", None) or getattr(resp, "text", "")).strip()

        # 2. Modern google.genai Client.models.generate_content
        elif hasattr(client, "models") and hasattr(client.models, "generate_content"):
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=0.2,
            )
            resp = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config,
            )
            response_text = str(resp.text or "").strip()

        # 3. Legacy google.generativeai GenerativeModel
        elif hasattr(client, "generate_content"):
            resp = client.generate_content(prompt, generation_config={"temperature": 0.2})
            response_text = str(resp.text or "").strip()

        else:
            raise TypeError(f"Unrecognized client interface: {type(client)}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if not response_text:
            raise RuntimeError("API returned empty response text.")

        print("[SUCCESS] Live Gemini call succeeded!")
        print(f"Latency:           {elapsed_ms:.1f} ms")
        print(f"Model:             {target_model}")
        print(f"Response Preview:  \"{response_text}\"")
        print("=" * 70)
        return 0

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        sanitized_msg = _sanitize_error_message(exc, clean_key)
        print(f"[FAILED] Live Gemini call failed after {elapsed_ms:.1f} ms:")
        print(f"Details: {sanitized_msg}")
        print("=" * 70)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional live Gemini smoke test for Legal UQ. Never logs or prints API keys."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (if omitted, reads $env:GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Target model identifier (default: {DEFAULT_GEMINI_MODEL} or $env:GEMINI_MODEL).",
    )
    args = parser.parse_args()

    exit_code = run_live_smoke_test(api_key=args.api_key, model_name=args.model)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

