#!/bin/sh

GCP_REGION="asia-east1"

PROJECT_ID=$(gcloud config list project --format "value(core.project)")

UPLOAD_DATA_DIR="mnist"

BUCKET_NAME="vgg19"
# BUCKET_NAME=${PROJECT_ID}-ml
TRAIN_BUCKET=gs://$BUCKET_NAME

# 既にバケットがある場合は削除する
gsutil rm -rf $TRAIN_BUCKET

# バケットを作成する
gsutil mb -l $GCP_REGION $TRAIN_BUCKET

# 画像データをpickleに変換する
# python /home/ishiyama/vgg/pickle_mnist.py

# 画像データをアップロードする
gsutil cp -r $UPLOAD_DATA_DIR $TRAIN_BUCKET

JOB_NAME=${BUCKET_NAME}_$(date +%Y%m%d%H%M%S)
GCP_SOURCE_DATA_DIR=$TRAIN_BUCKET/${UPLOAD_DATA_DIR}
gcloud ml-engine jobs submit training $JOB_NAME \
    --package-path=trainer \
    --module-name=trainer.task \
    --staging-bucket=$TRAIN_BUCKET \
    --region=$GCP_REGION \
    -- \
    --train-steps 1000 \
    --input-data-dir $GCP_SOURCE_DATA_DIR

