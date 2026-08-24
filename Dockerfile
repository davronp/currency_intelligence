# Container image for running the ingestion + forecasting pipeline.
# The Streamlit dashboard does NOT need this image - it runs from the base
# dependencies on Streamlit Cloud. This image adds the heavy `pipeline` extra
# (PySpark + Prophet) together with their system requirements.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# PySpark bundles Spark but still needs a JVM at runtime; Prophet's CmdStan
# backend is compiled from C++ (build-essential).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Resolve dependencies first so this layer is cached when only source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --extra pipeline

# Precompile CmdStan so the first forecast run is not slowed by a build.
RUN uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

COPY . .

# Default to the pipeline CLI help. Override, for example:
#   docker run --rm currency-intelligence --backfill-days 90
ENTRYPOINT ["uv", "run", "python", "run_pipeline.py"]
CMD ["--help"]
