# Written by ChatGPT with help from Scott Feister on 2026-07-02.
#
# Dockerfile for running train.py in a clean Python 3.12 container.
# The container contains the software environment.
# The outputs/ folder should be mounted from the host machine at runtime.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

CMD ["python", "train.py"]