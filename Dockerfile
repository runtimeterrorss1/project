FROM python
WORKDIR /app
COPY . /app
COPY Mlflow /app/models
RUN pip install Flask pandas mlflow scikit-learn
EXPOSE 5000
ENV FLASK_ENV=development
CMD ["python", "app.py"]