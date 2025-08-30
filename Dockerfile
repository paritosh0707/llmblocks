# LLMBlocks Production Docker Image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Create app user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy dependency files
COPY --chown=app:app pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY --chown=app:app src/ ./src/
COPY --chown=app:app examples/ ./examples/
COPY --chown=app:app README.md ./

# Install the package
RUN uv pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import llmblocks; print('LLMBlocks is healthy')" || exit 1

# Default command
CMD ["python", "-c", "import llmblocks; print('LLMBlocks container is running')"]

# Development image
FROM base as development

# Install dev dependencies
RUN uv sync --frozen

# Install additional dev tools
RUN uv pip install jupyter ipython

# Expose Jupyter port
EXPOSE 8888

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production image
FROM base as production

# Copy only necessary files
COPY --chown=app:app --from=base /home/app/src /home/app/src
COPY --chown=app:app --from=base /home/app/.venv /home/app/.venv

# Set production environment
ENV LLMBLOCKS_ENV=production \
    LLMBLOCKS_ENABLE_LOGGING=true \
    LLMBLOCKS_LOG_FORMAT=json

# Production command
CMD ["python", "-m", "llmblocks.cli.main", "--help"]
