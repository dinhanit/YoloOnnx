# Step 1: Clone the Docker image
docker pull dinhanit/fastapi_detect:latest

# Step 2: Run the Docker container
docker run -d -p 8501:8501 dinhanit/fastapi_detect:latest
