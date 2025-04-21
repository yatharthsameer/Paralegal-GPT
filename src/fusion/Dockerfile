FROM python:3.9-slim
WORKDIR /app
RUN pip install fastapi uvicorn sentence-transformers requests "huggingface-hub<0.18.0,>=0.15.1"
COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"] 