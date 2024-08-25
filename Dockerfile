# Dockerfile since my application is first containerized via Docker.
# Stage 1: Build
FROM python:3.10-slim AS build

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Stage 2: Final Image
FROM python:3.10-slim

WORKDIR /app

# Copy the dependencies from the build stage
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Copy the model file directly into the container
COPY models/cifar10_resnet18.pth /app/models/cifar10_resnet18.pth

# Copy the rest of the application code
COPY . /app

EXPOSE 5000

ENV FLASK_APP=app.py

# Use Gunicorn to serve the app. Can use flask run instead if its just for testing purpose
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "600", "app:app"]















