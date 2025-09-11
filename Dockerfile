# ----------------------------------------------------------------------------
# Financial-Market-Project – Dockerfile
# - Small, reproducible image for running the EDA CLI (`dsproj`)
# - Non-root user, pip cache optimization, and sensible Python env
# - Optional JupyterLab target for interactive EDA
# ----------------------------------------------------------------------------

# ---- Base ----
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (kept minimal; wheels cover numpy/pandas/sklearn/torch)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG APP_USER=app
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} ${APP_USER} \
 && useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash ${APP_USER}

WORKDIR /app

# ---- Dependency layer (better caching) ----
# Copy only files that affect dependency resolution.
COPY pyproject.toml README.md /app/
# Copy package source; if you want even tighter caching, split this further.
COPY src/ /app/src/

# Install project (pulls deps from pyproject.toml). If you prefer no dev deps,
# keep dev tools out of [project.dependencies].
RUN pip install --upgrade pip \
 && pip install -e .

# Copy runtime assets (configs, Makefile, etc.)
COPY configs/ /app/configs/
COPY reports/ /app/reports/
COPY data/ /app/data/
COPY models/ /app/models/
COPY notebooks/ /app/notebooks/

# Ensure writable directories exist
RUN mkdir -p /app/data/raw /app/data/interim /app/data/processed /app/reports/figures /app/reports/metrics \
 && chown -R ${APP_USER}:${APP_USER} /app

USER ${APP_USER}

# Healthcheck: CLI should respond
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD dsproj --help >/dev/null 2>&1 || exit 1

ENTRYPOINT ["dsproj"]
CMD ["--help"]

# ----------------------------------------------------------------------------
# Optional target: JupyterLab image for interactive EDA
#   docker build --target notebook -t youruser/financial-market-notebook .
#   docker run -p 8888:8888 -v "$PWD":/work -w /work youruser/financial-market-notebook
# ----------------------------------------------------------------------------
FROM base AS notebook
USER root
RUN pip install jupyterlab && mkdir -p /workspace && chown -R ${APP_USER}:${APP_USER} /workspace
USER ${APP_USER}
WORKDIR /workspace
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token="]
