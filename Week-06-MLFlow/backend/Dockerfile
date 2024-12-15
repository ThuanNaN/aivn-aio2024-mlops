FROM python:3.12.7-slim
    
WORKDIR /src

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app /src/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]