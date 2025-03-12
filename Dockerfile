# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5001 available to the world outside the container
EXPOSE 5001

# Run the application using Gunicorn for better performance in production
CMD ["gunicorn", "-b", "0.0.0.0:5001", "app:app"]
