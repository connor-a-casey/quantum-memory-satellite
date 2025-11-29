# Multi-Mode Quantum Memory Satellite Simulation
# Docker container for reproducible simulation environment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/input data/processed plots/memory_efficiency plots/downlink_probability plots/skr_analysis

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MPLBACKEND=Agg

# Create a non-root user for security
RUN useradd -m -u 1000 quantum && chown -R quantum:quantum /app
USER quantum

# Expose port for potential web interface (future use)
EXPOSE 8080

# Default command - run interactive shell
CMD ["/bin/bash"]
