FROM python:3.7-slim

# Copy required libraries
COPY ./requirements.txt ./

# Install required libraries
RUN apt update && apt -y upgrade && \
    pip install -r ./requirements.txt flask redisai


# Define user and group
ARG USER_ID
ARG GROUP_ID

# Create a new user with group
RUN addgroup --gid $GROUP_ID project
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID project

# Switch to a new user
USER project

# Select working directory
WORKDIR /home/project
COPY ./iris.onnx ./
COPY ./redisai-py.py ./

