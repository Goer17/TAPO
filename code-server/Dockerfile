FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -e . && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

ENV METHOD=inline \
    DEBUG=True \
    TIMEOUT=2000

CMD ["uvicorn", "code_server.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "8"]