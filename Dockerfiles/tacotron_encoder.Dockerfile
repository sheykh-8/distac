# Use an official Python runtime as a parent image
FROM pytorch/torchserve:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip install gunicorn

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define environment variable
ENV FLASK_ENV production

# Run app.py when the container launches
CMD ["gunicorn", "-w", "2", "--timeout", "1000", "-b", "0.0.0.0:5000", "model.encoder_server:app", "--log-level", "debug"]
