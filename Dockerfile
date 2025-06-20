FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pricing_api_service.py .
COPY dqn_dynamic_pricing.py .
COPY lstm_demand_forecasting.py .
COPY xgboost_price_elasticity.py .

# Create non-root user
RUN useradd --create-home --shell /bin/bash pricing
USER pricing

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "pricing_api_service:app", "--host", "0.0.0.0", "--port", "8000"]
