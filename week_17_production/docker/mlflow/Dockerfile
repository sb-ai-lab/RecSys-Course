FROM python:3.10-slim

WORKDIR /mlflow

RUN pip install mlflow==2.7.0

ENV BACKEND_URI sqlite:///mlflow.db

EXPOSE 5000

CMD mlflow ui --backend-store-uri $BACKEND_URI --host 0.0.0.0
