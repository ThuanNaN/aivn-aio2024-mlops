FROM python:3.11.9-slim
    
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app /app

WORKDIR /app

EXPOSE 3000

CMD ["python", "main.py"]