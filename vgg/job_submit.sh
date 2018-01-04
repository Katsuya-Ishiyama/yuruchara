#!/bin/sh

GCP_REGION="asia-east1"

PROJECT_ID=$(gcloud config list project --format "value(core.project)")

CHARACTER_TYPE="gotochi"

BUCKET_NAME="yuruchara"
TRAIN_BUCKET=gs://$BUCKET_NAME

JOB_NAME=${BUCKET_NAME}_$(date +%Y%m%d%H%M%S)
gcloud ml-engine jobs submit training $JOB_NAME \
    --config=job_config.yaml \
    --package-path=trainer \
    --module-name=trainer.task\
    --staging-bucket=$TRAIN_BUCKET \
    --region=$GCP_REGION \
    -- \
    --train-steps 1000 \
    --input-path $TRAIN_BUCKET/2017/${CHARACTER_TYPE}/${CHARACTER_TYPE}.json

