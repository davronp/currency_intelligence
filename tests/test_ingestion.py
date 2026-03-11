"""tests/test_ingestion.py.

Unit tests for src/ingestion/fetch_rates.py.

Covers:
- Frankfurter API response parsing
- URL routing (latest vs historical)
- JSONL upsert / idempotency
- Retry behaviour
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.ingestion.fetch_rates import (
    _build_url,
    fetch_exchange_rates,
    save_raw_rates,
)

# Frankfurter response shape
MOCK_FRANKFURTER_RESPONSE = {
    "amount": 1.0,
    "base": "USD",
    "date": "2024-01-15",
    "rates": {
        "EUR": 0.9201,
        "SEK": 10.4532,
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


def _make_payload(d: date = date(2024, 1, 15), rates: dict | None = None) -> dict:
    return {
        "base": "USD",
        "date": d.isoformat(),
        "rates": rates or {"EUR": 0.92, "SEK": 10.45},
        "fetched_at": f"{d.isoformat()}T10:00:00+00:00",
        "source": "https://api.frankfurter.app",
    }


class TestBuildUrl:
    def test_historical_date_uses_date_path(self):
        past = date(2024, 1, 15)
        url = _build_url("https://api.frankfurter.app", past)
        assert url == "https://api.frankfurter.app/2024-01-15"

    def test_today_uses_latest(self):
        url = _build_url("https://api.frankfurter.app", date.today())
        assert url == "https://api.frankfurter.app/latest"

    def test_future_date_uses_latest(self):
        future = date.today() + timedelta(days=5)
        url = _build_url("https://api.frankfurter.app", future)
        assert url == "https://api.frankfurter.app/latest"


class TestFetchExchangeRates:
    def test_returns_expected_keys(self):
        with patch("requests.get", return_value=_mock_response(MOCK_FRANKFURTER_RESPONSE)):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR", "SEK", "GBP"],
                base_url="https://api.frankfurter.app",
                run_date=date(2024, 1, 15),
            )
        assert {"base", "date", "rates", "fetched_at", "source"}.issubset(result.keys())

    def test_base_currency_preserved(self):
        with patch("requests.get", return_value=_mock_response(MOCK_FRANKFURTER_RESPONSE)):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR"],
                base_url="https://api.frankfurter.app",
                run_date=date(2024, 1, 15),
            )
        assert result["base"] == "USD"

    def test_date_from_api_response(self):
        with patch("requests.get", return_value=_mock_response(MOCK_FRANKFURTER_RESPONSE)):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR"],
                base_url="https://api.frankfurter.app",
                run_date=date(2024, 1, 15),
            )
        assert result["date"] == "2024-01-15"

    def test_rate_values_are_floats(self):
        with patch("requests.get", return_value=_mock_response(MOCK_FRANKFURTER_RESPONSE)):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR", "SEK", "GBP"],
                base_url="https://api.frankfurter.app",
                run_date=date(2024, 1, 15),
            )
        for v in result["rates"].values():
            assert isinstance(v, float)

    def test_missing_currency_does_not_raise(self):
        with patch("requests.get", return_value=_mock_response(MOCK_FRANKFURTER_RESPONSE)):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR", "XYZ_NONEXISTENT"],
                base_url="https://api.frankfurter.app",
                run_date=date(2024, 1, 15),
            )
        assert "EUR" in result["rates"]
        assert "XYZ_NONEXISTENT" not in result["rates"]

    def test_retries_on_failure_then_succeeds(self):
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "network error"
                raise requests.ConnectionError(msg)
            return _mock_response(MOCK_FRANKFURTER_RESPONSE)

        with patch("requests.get", side_effect=side_effect), patch("time.sleep"):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR"],
                base_url="https://api.frankfurter.app",
                run_date=date(2024, 1, 15),
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
                    base_url="https://api.frankfurter.app",
                    run_date=date(2024, 1, 15),
                    retry_attempts=2,
                    retry_backoff=0,
                )

    def test_http_error_triggers_retry(self):
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response({}, status_code=503)
            return _mock_response(MOCK_FRANKFURTER_RESPONSE)

        with patch("requests.get", side_effect=side_effect), patch("time.sleep"):
            result = fetch_exchange_rates(
                base_currency="USD",
                target_currencies=["EUR"],
                base_url="https://api.frankfurter.app",
                run_date=date(2024, 1, 15),
                retry_attempts=3,
                retry_backoff=0,
            )
        assert "EUR" in result["rates"]


class TestSaveRawRates:
    def test_jsonl_file_created(self, tmp_path):
        out = save_raw_rates(_make_payload(), tmp_path)
        assert out.exists()
        assert out.name == "rates_USD.jsonl"

    def test_file_contains_valid_json_lines(self, tmp_path):
        save_raw_rates(_make_payload(), tmp_path)
        lines = (tmp_path / "rates_USD.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["base"] == "USD"
        assert rec["rates"]["EUR"] == 0.92

    def test_multiple_dates_appended(self, tmp_path):
        save_raw_rates(_make_payload(date(2024, 1, 13)), tmp_path)
        save_raw_rates(_make_payload(date(2024, 1, 14)), tmp_path)
        save_raw_rates(_make_payload(date(2024, 1, 15)), tmp_path)
        lines = (tmp_path / "rates_USD.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

    def test_records_sorted_by_date(self, tmp_path):
        save_raw_rates(_make_payload(date(2024, 1, 15)), tmp_path)
        save_raw_rates(_make_payload(date(2024, 1, 13)), tmp_path)
        save_raw_rates(_make_payload(date(2024, 1, 14)), tmp_path)
        lines = (tmp_path / "rates_USD.jsonl").read_text().strip().splitlines()
        dates = [json.loads(l)["date"] for l in lines]
        assert dates == sorted(dates)

    def test_idempotent_same_date_no_duplicate(self, tmp_path):
        save_raw_rates(_make_payload(date(2024, 1, 15)), tmp_path)
        save_raw_rates(_make_payload(date(2024, 1, 15)), tmp_path)
        lines = (tmp_path / "rates_USD.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    def test_no_overwrite_without_force(self, tmp_path):
        payload = _make_payload(date(2024, 1, 15), rates={"EUR": 0.92})
        save_raw_rates(payload, tmp_path)

        payload2 = _make_payload(date(2024, 1, 15), rates={"EUR": 0.99})
        save_raw_rates(payload2, tmp_path, force=False)

        lines = (tmp_path / "rates_USD.jsonl").read_text().strip().splitlines()
        rec = json.loads(lines[0])
        assert rec["rates"]["EUR"] == 0.92  # original preserved

    def test_force_overwrites_existing_date(self, tmp_path):
        payload = _make_payload(date(2024, 1, 15), rates={"EUR": 0.92})
        save_raw_rates(payload, tmp_path)

        payload2 = _make_payload(date(2024, 1, 15), rates={"EUR": 0.99})
        save_raw_rates(payload2, tmp_path, force=True)

        lines = (tmp_path / "rates_USD.jsonl").read_text().strip().splitlines()
        rec = json.loads(lines[0])
        assert rec["rates"]["EUR"] == 0.99  # updated

    def test_no_subdirectory_created(self, tmp_path):
        save_raw_rates(_make_payload(), tmp_path)
        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert subdirs == []
