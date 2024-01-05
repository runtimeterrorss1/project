FROM python:3.10
RUN mkdir -p /app
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip 
EXPOSE 5002
RUN pip install -r requirements.txt
CMD ["python","app.py"]
