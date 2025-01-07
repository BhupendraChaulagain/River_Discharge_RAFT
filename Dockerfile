FROM python:3.10-slim

# Install required libraries including libGL.so.1
RUN apt-get update && apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && apt-get clean

# Set the working directory
WORKDIR /opt/VELOCITY_ML/River_Discharge

# Copy the current directory contents into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the FastAPI application
CMD ["sh", "-c", "pip install -r requirements.txt && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]
