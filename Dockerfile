FROM python:3.11.3

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["python", "app.py"]
