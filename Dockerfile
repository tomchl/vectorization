FROM python:3.8

WORKDIR /usr/src/app

RUN python -m pip install --upgrade pip

RUN python -m pip install tensorflow
RUN python -m pip install tensorflow-hub
RUN python -m pip install tf-nightly

COPY server.py .

CMD ["python","-u", "server.py"]