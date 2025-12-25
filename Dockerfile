FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and modules
COPY app/ /app/app/
COPY core/ /app/core/
COPY artifacts/ /app/artifacts/

# Set PYTHONPATH to include the current directory
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
