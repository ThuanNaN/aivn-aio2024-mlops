FROM python:3.11.9-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app

CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000