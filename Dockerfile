# Use a lightweight Python base image to stay within the 8GB RAM limit
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# The command that the automated grader will execute
CMD ["python", "inference.py"]