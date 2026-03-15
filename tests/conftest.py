"""tests/conftest.py.

Shared pytest fixtures for the Currency Intelligence test suite.

The SparkSession is created once per test session (``scope="session"``)
to avoid the expensive per-test JVM startup overhead.
"""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Provide a local SparkSession for the entire test session.

    Configuration is minimal to keep tests fast:
    - ``local[2]`` - two cores
    - ``shuffle.partitions=2`` - low for small test data
    - ``WARN`` log level - suppress Spark's verbose INFO output
    """
    session = (
        SparkSession.builder.appName("CurrencyIntelligence-Tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("WARN")
    yield session
    session.stop()
