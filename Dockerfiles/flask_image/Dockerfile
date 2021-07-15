FROM python:3.7-slim

# Define user and group
ARG USER_ID
ARG GROUP_ID

# Copy required libraries
COPY ./requirements.txt ./
# Copy RedMask Module



# Install required libraries
RUN apt update && apt -y upgrade && \
    apt -yq install tk libffi-dev gcc libssl-dev libglib2.0-0 && \
    apt install ffmpeg libsm6 libxext6  -y && \
    python -m pip install --upgrade pip && \
    pip install -r ./requirements.txt
# Create a new user with group
RUN addgroup --gid $GROUP_ID mask && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID mask
COPY ./data ./data
COPY ./RedMask ./RedMask
COPY ./setup.py ./
RUN python setup.py install

# Switch to a new user
USER mask

# Select working directory
WORKDIR /home/mask


