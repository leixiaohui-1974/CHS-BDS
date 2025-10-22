# Multi-stage build for CHS-BDS
FROM python:3.10-slim as builder

LABEL maintainer="Lei Xiaohui <leixiaohui@example.com>"
LABEL description="CHS-BDS: GNSS Comprehensive Monitoring System"
LABEL version="0.1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash chsbds && \
    mkdir -p /app /data /output /logs && \
    chown -R chsbds:chsbds /app /data /output /logs

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=chsbds:chsbds . .

# Install the package
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER chsbds

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CHS_BDS_DATA_DIR=/data
ENV CHS_BDS_OUTPUT_DIR=/output
ENV CHS_BDS_LOG_DIR=/logs

# Create volumes
VOLUME ["/data", "/output", "/logs"]

# Expose port for future web interface
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import gnss_monitoring; print('OK')" || exit 1

# Default command
ENTRYPOINT ["chs-bds"]
CMD ["--help"]
