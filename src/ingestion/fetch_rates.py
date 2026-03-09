"""src/ingestion/fetch_rates.py

Ingestion stage: fetch live exchange rates and persist raw JSON.

The raw file is written as:
  <raw_path>/<YYYY-MM-DD>/rates_<base>.json

Idempotency: if the file for today already exists the fetch is
skipped unless ``force=True`` is passed.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import requests

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


def fetch_exchange_rates(
    base_currency: str,
    target_currencies: list[str],
    base_url: str,
    timeout: int = 30,
    retry_attempts: int = 3,
    retry_backoff: int = 5,
) -> dict[str, Any]:
    """Fetch exchange rates for *base_currency* from the public API.

    Parameters
    ----------
    base_currency:
        The base currency code (e.g. ``"USD"``).
    target_currencies:
        List of target currency codes to extract from the response.
    base_url:
        API endpoint template - ``{base_currency}`` will be appended.
    timeout:
        HTTP request timeout in seconds.
    retry_attempts:
        Number of retry attempts on transient failures.
    retry_backoff:
        Seconds to wait between retries.

    Returns
    -------
    dict
        Normalised payload with keys: ``base``, ``date``, ``rates``,
        ``fetched_at``.

    Raises
    ------
    RuntimeError
        When all retry attempts are exhausted.

    """
    url = f"{base_url}/{base_currency}"
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, retry_attempts + 1):
        try:
            logger.info(
                "Fetching rates [attempt %d/%d]: %s",
                attempt,
                retry_attempts,
                url,
            )
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()

            rates_all: dict[str, float] = payload.get("rates", {})
            rates_filtered = {ccy: rates_all[ccy] for ccy in target_currencies if ccy in rates_all}

            missing = set(target_currencies) - set(rates_filtered)
            if missing:
                logger.warning("Currencies not found in API response: %s", missing)

            result = {
                "base": base_currency,
                "date": payload.get("time_last_update_utc", str(date.today())),
                "rates": rates_filtered,
                "fetched_at": datetime.now(UTC).isoformat(),
                "source": base_url,
            }
            logger.info("Fetched %d rate(s) for base=%s", len(rates_filtered), base_currency)
            return result

        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            logger.warning(
                "Fetch attempt %d failed: %s. Retrying in %ds…",
                attempt,
                exc,
                retry_backoff,
            )
            if attempt < retry_attempts:
                time.sleep(retry_backoff)

    msg = f"All {retry_attempts} fetch attempts failed for {base_currency}"
    raise RuntimeError(msg) from last_exc


def save_raw_rates(
    payload: dict[str, Any],
    raw_dir: Path,
    run_date: date | None = None,
    force: bool = False,
) -> Path:
    """Persist the raw API payload as a JSON file.

    Parameters
    ----------
    payload:
        The dict returned by :func:`fetch_exchange_rates`.
    raw_dir:
        Root directory for raw data (``data/raw``).
    run_date:
        Partition date (defaults to today UTC).
    force:
        Overwrite existing file even if it exists.

    Returns
    -------
    Path
        Full path of the written JSON file.

    """
    run_date = run_date or date.today()
    date_str = run_date.isoformat()
    base = payload["base"]

    out_dir = raw_dir / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"rates_{base}.json"

    if out_file.exists() and not force:
        logger.info("Raw file already exists, skipping: %s", out_file)
        return out_file

    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Saved raw rates → %s", out_file)
    return out_file


def run_ingestion(
    base_currency: str,
    target_currencies: list[str],
    raw_dir: Path,
    base_url: str,
    timeout: int = 30,
    retry_attempts: int = 3,
    retry_backoff: int = 5,
    force: bool = False,
    run_date: date | None = None,
) -> Path:
    """Orchestrate fetch + persist in one call.

    Returns the Path of the saved JSON file.
    """
    payload = fetch_exchange_rates(
        base_currency=base_currency,
        target_currencies=target_currencies,
        base_url=base_url,
        timeout=timeout,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
    )
    return save_raw_rates(payload, raw_dir, run_date=run_date, force=force)
