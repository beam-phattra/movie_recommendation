FROM python:3.10

RUN pip --no-cache-dir install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

WORKDIR /home/

COPY /models/ /models/
COPY /dataset/ /dataset/
COPY /configs/ /configs/
COPY /app/ /home/

EXPOSE 9070

CMD uvicorn app:app --workers 1 --host 0.0.0.0 --port 9070