FROM python:3.12-slim

# Same MLflow version as the project's uv.lock, so the DB schema matches.
RUN pip install --no-cache-dir mlflow==3.14.0

# Hugging Face Spaces run as uid 1000.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
COPY --chown=user mlflow.db ./mlflow.db
COPY --chown=user start.sh ./start.sh

EXPOSE 7860
CMD ["bash", "start.sh"]
