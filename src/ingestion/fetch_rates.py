"""src/ingestion/fetch_rates.py.

Ingestion stage: fetch exchange rates from the Frankfurter API
and persist to a single JSONL file per base currency.

Storage layout
--------------
    data/raw/rates_USD.jsonl   <- one JSON line per date, upserted on each run
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


def _build_url(base_url: str, run_date: date) -> str:
    today = date.today()
    if run_date >= today:
        return f"{base_url}/latest"
    return f"{base_url}/{run_date.isoformat()}"


def fetch_exchange_rates(
    base_currency: str,
    target_currencies: list[str],
    base_url: str,
    run_date: date,
    timeout: int = 30,
    retry_attempts: int = 3,
    retry_backoff: int = 5,
) -> dict[str, Any]:
    """Fetch exchange rates from Frankfurter for *run_date*.

    Returns a normalised dict: ``base``, ``date``, ``rates``, ``fetched_at``.
    Raises RuntimeError when all retry attempts are exhausted.
    """
    url = _build_url(base_url, run_date)
    params = {"from": base_currency, "to": ",".join(target_currencies)}
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, retry_attempts + 1):
        try:
            logger.info(
                "Fetching rates [%d/%d]: %s (date=%s)",
                attempt,
                retry_attempts,
                url,
                run_date,
            )
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()

            rates: dict[str, float] = payload.get("rates", {})
            missing = set(target_currencies) - set(rates)
            if missing:
                logger.warning("Currencies missing for %s: %s", run_date, missing)

            result = {
                "base": base_currency,
                "date": payload.get("date", run_date.isoformat()),
                "rates": rates,
                "fetched_at": datetime.now(UTC).isoformat(),
                "source": base_url,
            }
            logger.info(
                "Fetched %d rate(s) base=%s date=%s",
                len(rates),
                base_currency,
                result["date"],
            )
            return result

        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            logger.warning("Attempt %d failed for %s: %s", attempt, run_date, exc)
            if attempt < retry_attempts:
                time.sleep(retry_backoff)

    msg = f"All {retry_attempts} fetch attempts failed for {base_currency} on {run_date}"
    raise RuntimeError(msg) from last_exc


def save_raw_rates(
    payload: dict[str, Any],
    raw_dir: Path,
    force: bool = False,
) -> Path:
    """Upsert one date record into ``data/raw/rates_<BASE>.jsonl``.

    Idempotent: re-running the same date replaces the existing record.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_date = payload["date"]
    out_file = raw_dir / f"rates_{payload['base']}.jsonl"

    existing: dict[str, dict] = {}
    if out_file.exists():
        with open(out_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        existing[rec["date"]] = rec
                    except json.JSONDecodeError:
                        pass

    if record_date in existing and not force:
        logger.info("Record for %s already exists, skipping", record_date)
        return out_file

    existing[record_date] = payload
    with open(out_file, "w", encoding="utf-8") as fh:
        for rec in sorted(existing.values(), key=lambda r: r["date"]):
            fh.write(json.dumps(rec) + "\n")

    logger.info("Saved raw rates -> %s (%d total records)", out_file.name, len(existing))
    return out_file


def run_ingestion(
    base_currency: str,
    target_currencies: list[str],
    raw_dir: Path,
    base_url: str,
    run_date: date,
    timeout: int = 30,
    retry_attempts: int = 3,
    retry_backoff: int = 5,
    force: bool = False,
) -> Path:
    """Orchestrate fetch + persist for a single date."""
    payload = fetch_exchange_rates(
        base_currency=base_currency,
        target_currencies=target_currencies,
        base_url=base_url,
        run_date=run_date,
        timeout=timeout,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
    )
    return save_raw_rates(payload, raw_dir, force=force)
