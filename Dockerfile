# Customer Churn Predictor - Streamlit app in a slim, non-root image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the app needs at runtime: code + the three model artifacts
COPY app.py random_forest.pkl scaler.pkl feature_names.pkl ./

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501

# Docker marks the container unhealthy if Streamlit stops responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
