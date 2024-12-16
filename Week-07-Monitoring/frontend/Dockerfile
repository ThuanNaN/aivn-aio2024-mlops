FROM python:3.11.9-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app /app

WORKDIR /app

EXPOSE 7860

CMD ["python", "main.py"]