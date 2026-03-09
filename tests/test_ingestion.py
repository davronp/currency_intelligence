"""tests/test_ingestion.py

Unit tests for src/ingestion/fetch_rates.py.

All HTTP calls are mocked so no real API access is required.
"""

from __future__ import annotations

import json
import time
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.ingestion.fetch_rates import (
    fetch_exchange_rates,
    save_raw_rates,
)

MOCK_API_RESPONSE = {
    "result": "success",
    "time_last_update_utc": "2024-01-15 00:00:01 +0000",
    "base_code": "USD",
    "rates": {
        "EUR": 0.9201,
        "SEK": 10.4532,
        "UZS": 12650.0,
        "GBP": 0.7920,
    },
}


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = json_data
    mock.status_code = status_code
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}")
    return mock


class TestFetchExchangeRates:
    def test_returns_expected_keys(self):
        with patch("requests.get", return_value=_mock_response(MOCK_API_RESPONSE)):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR", "SEK", "UZS"],
                base_url="https://mock.api/latest",
            )
        assert "base" in result
        assert "rates" in result
        assert "fetched_at" in result

    def test_filters_target_currencies(self):
        with patch("requests.get", return_value=_mock_response(MOCK_API_RESPONSE)):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR", "SEK"],
                base_url="https://mock.api/latest",
            )
        assert set(result["rates"].keys()) == {"EUR", "SEK"}
        assert "UZS" not in result["rates"]

    def test_base_currency_in_result(self):
        with patch("requests.get", return_value=_mock_response(MOCK_API_RESPONSE)):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR"],
                base_url="https://mock.api/latest",
            )
        assert result["base"] == "USD"

    def test_rate_values_are_floats(self):
        with patch("requests.get", return_value=_mock_response(MOCK_API_RESPONSE)):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR", "SEK", "UZS"],
                base_url="https://mock.api/latest",
            )
        for v in result["rates"].values():
            assert isinstance(v, float)

    def test_missing_currency_does_not_raise(self):
        """Currencies not in the API response should be silently skipped."""
        with patch("requests.get", return_value=_mock_response(MOCK_API_RESPONSE)):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR", "XYZ_NONEXISTENT"],
                base_url="https://mock.api/latest",
            )
        assert "EUR" in result["rates"]
        assert "XYZ_NONEXISTENT" not in result["rates"]

    def test_retries_on_failure_then_succeeds(self):
        [
            requests.ConnectionError("network error"),
            _mock_response(MOCK_API_RESPONSE),
        ]

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "network error"
                raise requests.ConnectionError(msg)
            return _mock_response(MOCK_API_RESPONSE)

        with patch("requests.get", side_effect=side_effect):
            with patch("time.sleep"):  # skip backoff delay
                result = fetch_exchange_rates(
                    base_currency="USD",
                    target_currencies=["EUR"],
                    base_url="https://mock.api/latest",
                    retry_attempts=3,
                    retry_backoff=0,
                )
        assert result["base"] == "USD"
        assert call_count == 2

    def test_exhausted_retries_raise_runtime_error(self):
        with patch("requests.get", side_effect=requests.ConnectionError("down")), patch("time.sleep"):
            with pytest.raises(RuntimeError, match="All 2 fetch attempts"):
                fetch_exchange_rates(
                    base_currency="USD",
                    target_currencies=["EUR"],
                    base_url="https://mock.api/latest",
                    retry_attempts=2,
                    retry_backoff=0,
                )


class TestSaveRawRates:
    def test_file_is_created(self, tmp_path):
        payload = {
            "base": "USD",
            "date": "2024-01-15",
            "rates": {"EUR": 0.92},
            "fetched_at": "2024-01-15T10:00:00+00:00",
            "source": "https://mock.api",
        }
        out = save_raw_rates(payload, tmp_path, run_date=date(2024, 1, 15))
        assert out.exists()
        assert out.name == "rates_USD.json"

    def test_saved_content_is_valid_json(self, tmp_path):
        payload = {
            "base": "USD",
            "date": "2024-01-15",
            "rates": {"EUR": 0.92, "SEK": 10.45},
            "fetched_at": "2024-01-15T10:00:00+00:00",
            "source": "https://mock.api",
        }
        out = save_raw_rates(payload, tmp_path, run_date=date(2024, 1, 15))
        with open(out) as fh:
            loaded = json.load(fh)
        assert loaded["base"] == "USD"
        assert loaded["rates"]["EUR"] == 0.92

    def test_no_overwrite_without_force(self, tmp_path):
        payload = {
            "base": "USD",
            "date": "2024-01-15",
            "rates": {"EUR": 0.92},
            "fetched_at": "2024-01-15T10:00:00+00:00",
            "source": "https://mock.api",
        }
        out1 = save_raw_rates(payload, tmp_path, run_date=date(2024, 1, 15))
        mtime1 = out1.stat().st_mtime

        # Second call without force should skip
        out2 = save_raw_rates(payload, tmp_path, run_date=date(2024, 1, 15), force=False)
        assert out2.stat().st_mtime == mtime1

    def test_force_overwrites_existing(self, tmp_path):
        payload = {
            "base": "USD",
            "date": "2024-01-15",
            "rates": {"EUR": 0.92},
            "fetched_at": "2024-01-15T10:00:00+00:00",
            "source": "https://mock.api",
        }
        out1 = save_raw_rates(payload, tmp_path, run_date=date(2024, 1, 15))
        mtime1 = out1.stat().st_mtime

        time.sleep(0.05)
        payload["rates"]["EUR"] = 0.95
        out2 = save_raw_rates(payload, tmp_path, run_date=date(2024, 1, 15), force=True)
        assert out2.stat().st_mtime > mtime1

    def test_date_partition_directory_created(self, tmp_path):
        payload = {
            "base": "USD",
            "date": "2024-03-20",
            "rates": {},
            "fetched_at": "2024-03-20T00:00:00+00:00",
            "source": "test",
        }
        save_raw_rates(payload, tmp_path, run_date=date(2024, 3, 20))
        assert (tmp_path / "2024-03-20").is_dir()
