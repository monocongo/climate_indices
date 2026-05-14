FROM python:3.14-slim AS builder

LABEL authors="monocongo@gmail.com"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency and project metadata files
COPY pyproject.toml uv.lock ./
COPY README.md LICENSE ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.14-slim

# Install runtime dependencies
RUN apt-get update && apt-get --no-install-recommends install -y \
    build-essential \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy uv and virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /bin/uv /bin/uv

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash climate
USER climate

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Set entrypoint
ENTRYPOINT ["python", "-m", "climate_indices"]
