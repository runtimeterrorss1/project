FROM python
RUN mkdir -p /app
WORKDIR /app
COPY . /app
EXPOSE 5002
RUN pip install -r requirements.txt
CMD ["python","app.py"]
