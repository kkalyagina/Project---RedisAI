#!/bin/bash

docker build \
    -t training_img \
    -f ./Dockerfiles/train_models/Dockerfile_GPU .
docker run \
    --name=training_cont \
    -v $(pwd)/data/datasets:/app/data/Datasets \
    -v $(pwd)/scripts:/app/scripts \
    -v $(pwd)/data/models:/app/data/models \
    -v $(pwd)/.env:/app/.env \
    --rm \
    -e AWS_ACCESS_KEY_ID=AKIA4TULZOVHUHV7E74B \
    -e AWS_SECRET_ACCESS_KEY=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 \
    -it \
    training_img \
    bash
