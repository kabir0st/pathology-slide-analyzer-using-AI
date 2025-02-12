FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    python3-psycopg2 \
    python3-dev openslide-tools python3-openslide

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app

EXPOSE 8235

HEALTHCHECK CMD curl --fail http://localhost:8235/health || exit 1
