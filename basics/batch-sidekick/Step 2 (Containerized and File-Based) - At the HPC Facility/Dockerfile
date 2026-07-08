# Written by ChatGPT with help from Scott Feister on 2026-07-02.
#
# Dockerfile for running train.py in a clean Python 3.12 container.
#
# Build from the directory containing this Dockerfile, train.py, and requirements.txt:
#
#   docker build -t sidekick-ridge-train .
#
# Run from a host directory containing the input HDF5 file.
# The output model file will be written wherever -o points inside the mounted folder.
# The filenames below are examples; choose any input and output names you want.
#
#   docker run --rm -v "$PWD/outputs:/work" sidekick-ridge-train \
#       -i /work/experiment_data.h5 \
#       -o /work/mymodel.joblib
#
# In this example:
#
#   Host current directory       ->  container path /work
#   ./outputs/experiment_data.h5         ->  /work/experiment_data.h5
#   ./outputs/mymodel.joblib             <-  /work/mymodel.joblib

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt train.py ./

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "train.py"]