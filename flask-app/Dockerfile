FROM python
RUN mkdir -p /app
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 5002
CMD ["python","app.py"]
