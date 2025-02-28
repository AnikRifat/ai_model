# Stage 1: Build H2O-3 from source
FROM gradle:7.2.0-jdk11 AS builder

# Install Node.js, npm, and Python
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip && \
    curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Set Python 3 as the default 'python' command
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory
WORKDIR /opt/h2o-3

# Clone the H2O-3 repository
RUN git clone https://github.com/h2oai/h2o-3.git .

# Build H2O-3, skipping tests and avoiding file system watcher issues
RUN ./gradlew build -x test --no-daemon

# Stage 2: Create a lightweight runtime image
FROM openjdk:11-jre-slim

# Copy the built H2O-3 JAR from the builder stage
COPY --from=builder /opt/h2o-3/h2o.jar /app/h2o.jar

# Install Python and necessary libraries for runtime
RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install flask h2o pandas numpy

# Set the working directory
WORKDIR /app

# Copy your application files (e.g., a Flask server)
COPY server.py /app/server.py

# Expose port 5000
EXPOSE 5000

# Run your application
CMD ["python3", "server.py"]