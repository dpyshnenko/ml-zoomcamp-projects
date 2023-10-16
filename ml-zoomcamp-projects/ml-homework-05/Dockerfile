# Start with the provided image
FROM svizor/zoomcamp-model:3.10.12-slim

# Set the working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy the Pipenv and Pipfile.lock files for dependency resolution
COPY Pipfile Pipfile.lock ./

# Install dependencies from the Pipenv file
RUN pipenv install --deploy --ignore-pipfile --skip-lock

# Copy the Flask script and models to the container
COPY flask-app.py .
COPY dv.bin .
COPY model1.bin .

# Run the Flask app with Gunicorn
CMD ["pipenv", "run", "gunicorn", "flask-app:app", "--bind", "0.0.0.0:9696"]