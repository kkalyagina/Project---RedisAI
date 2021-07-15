#!/bin/bash

export PATH=/opt/venv/bin:$PATH
source /opt/venv/bin/activate
mlflow server \
        --backend-store-uri postgresql://$PG_USER:$PG_PASS@$PG_CONTAINER:$PG_PORT/postgres \
        --default-artifact-root s3://$AWS_BUCKET/artifacts \
        --host 0.0.0.0 \
        --port $MLFLOW_PORT


/bin/bash