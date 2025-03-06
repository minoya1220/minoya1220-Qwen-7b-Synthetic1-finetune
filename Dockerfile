FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install Rust and UV
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install uv && uv --version

# Create a non-root user
RUN useradd -m appuser

# Copy requirements file first for better caching
COPY requirements.txt /app/
# Install PyTorch with CUDA support explicitly
RUN uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

# Set proper permissions
RUN chown -R appuser:appuser /app

# Copy application code
COPY . /app/
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Use CMD instead of ENTRYPOINT for more flexibility
CMD ["python", "main.py"] 