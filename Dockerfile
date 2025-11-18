FROM python:3.12-slim

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY "pyproject.toml" "uv.lock" ".python-version" ./

RUN uv sync --locked

COPY "model/final_model.bin" ./

COPY "src/predict.py" ./

EXPOSE 9696

ENTRYPOINT [ "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]