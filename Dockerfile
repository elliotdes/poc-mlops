FROM python:3.8.5

RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false

WORKDIR /code

COPY . .

ENV PYTHONPATH .

RUN poetry install --no-root
