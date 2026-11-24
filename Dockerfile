# 1. Start from a lightweight base image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Run updates
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \ 
    && rm -rf /var/lib/apt/lists/*

# 4. Copy dependency files and install them first (for faster rebuilds)
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# 5. Copy the application code
COPY app/ ./app/
COPY api/ ./api/
COPY fusion/models/lgbm_v1.txt fusion/models/lgbm_v1.meta.json ./fusion/models/

# 6. Expose the API endpoint
EXPOSE 8000

# 7. Define the commend to run when the container starts
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]