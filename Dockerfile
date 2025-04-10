# Use Python 3.11 image as base
FROM python:3.11

ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /code

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# Install required packages
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of your application
COPY . /code/

RUN chmod +x /code/start.sh

# Expose the port your app runs on
EXPOSE 8000

CMD ["/code/start.sh"]

# Command to run the application
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]