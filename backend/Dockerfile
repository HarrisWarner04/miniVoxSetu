FROM python:3.12-slim

# WHY: We use the slim image to minimize attack surface and size.
# We explicitly avoid the -alpine image because some Python C-extensions
# (like numpy, scipy, librosa) struggle to compile with alpine's musl libc.

WORKDIR /app

# Install system dependencies needed for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies with CPU-only PyTorch explicitly to save ~1.5GB
# of unused CUDA libraries on our cloud instance.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy source code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run Uvicorn server
# WHY: We run 1 worker because the ThreadPoolExecutor handles the blocking acoustic tasks,
# and we want to conserve RAM for the PyTorch models.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
