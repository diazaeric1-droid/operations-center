# Operations Center — container image for the Kubernetes / cloud path.
# (Streamlit Cloud deploys straight from requirements.txt and ignores this file;
#  this image is for the self-hosted k8s topology in k8s/ and for AWS ECS/EKS.)
#
# Includes the RAG extras so the Note Search page + MCP server work in-cluster
# against the pgvector Service. The fastembed ONNX model downloads on first use.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System deps: psycopg2-binary ships wheels, so we only need a slim base.
# Install Python deps first (layer-cached across code changes).
COPY requirements.txt requirements-rag.txt ./
RUN pip install -r requirements.txt -r requirements-rag.txt

COPY . .

EXPOSE 8501

# Liveness/readiness hit Streamlit's built-in health endpoint (see k8s probes).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health',timeout=3).status==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
