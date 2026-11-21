# Blast Radius: per-test sys.monitoring coverage + AST dependency maps + fast verified selector
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock && rm /tmp/requirements.lock

RUN git clone --filter=blob:none https://github.com/tobymao/sqlglot.git /app/sqlglot \
    && cd /app/sqlglot \
    && git checkout eb11aec4be8cb2122afe713a4816662c3f0fb2fa \
    && rm -rf /app/sqlglot/.git

COPY build_maps.py /app/build_maps.py
RUN SKIP_INTEGRATION=0 PYTHONPATH=/app/sqlglot python /app/build_maps.py

COPY selector.py /app/selector.py
RUN chmod -R a+rX /app

WORKDIR /app
CMD ["python", "/app/selector.py"]
