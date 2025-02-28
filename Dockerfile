FROM python:3.9-slim
WORKDIR /app
RUN pip install prophet flask pandas numpy
COPY server.py .
EXPOSE 5000
CMD ["python", "server.py"]