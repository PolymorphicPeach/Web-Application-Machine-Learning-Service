FROM python:3.8-slim
WORKDIR /app
COPY . /app
# Install Flask and other dependencies
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]
