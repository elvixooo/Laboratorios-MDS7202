FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY requirements.txt .
RUN uv pip install --no-cache --system -r requirements.txt

COPY main.py .
COPY models/ models/

VOLUME ["/app/models"]

EXPOSE 8000

CMD ["python", "main.py"]
