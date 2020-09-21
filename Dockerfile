FROM python:3.7.5-slim

WORKDIR /usr/src/app

RUN python -m pip install \
        tensorflow \
        tensorflow-hub

COPY server.py .

CMD ["python", "server.py"]