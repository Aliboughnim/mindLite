# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Django project files to the container
COPY . .

# Expose the port that Django runs on (default is 8000)
EXPOSE 8000

# Set environment variables (if needed)
# ENV DJANGO_SETTINGS_MODULE=myproject.settings.production

# Run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
