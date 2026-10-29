"""Pytest fixtures and test environment isolation for Legal UQ.

Guarantees:
- Zero live API calls during automated tests.
- Complete test isolation regardless of whether a local .env exists.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure pytest tests execute 100% offline without live API credentials."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

