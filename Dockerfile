FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY workflow.py video_providers.py frame_assets.py service.py ./
CMD ["python", "service.py"]
