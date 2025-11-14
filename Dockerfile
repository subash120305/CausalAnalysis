FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    graphviz \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create non-root user
RUN useradd -m -u 1000 causalbench && \
    chown -R causalbench:causalbench /app
USER causalbench

# Set default entrypoint
ENTRYPOINT ["bash", "scripts/run_all.sh"]
