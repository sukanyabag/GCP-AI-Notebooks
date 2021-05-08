!sudo chown -R jupyter:jupyter /home/jupyter/training-data-analyst
!pip install --user google-cloud-bigquery==1.25.0
# change these to try this notebook out
BUCKET = 'cloud-training-demos-ml' # Replace with the your bucket name
PROJECT = 'cloud-training-demos' # Replace with your project-id
REGION = 'us-central1'

import os

from google.cloud import bigquery
os.environ["PROJECT"] = PROJECT
os.environ["BUCKET"] = BUCKET
os.environ["REGION"] = REGION
os.environ["TFVERSION"] = "2.1"
os.environ["PYTHONVERSION"] = "3.7"
%%bash
export PROJECT=$(gcloud config list project --format "value(core.project)")
echo "Your current GCP Project Name is: "$PROJECT

%%bash

# Create a BigQuery dataset for babyweight if it doesn't exist
datasetexists=$(bq ls -d | grep -w babyweight)

if [ -n "$datasetexists" ]; then
    echo -e "BigQuery dataset already exists, let's not recreate it."

else
    echo "Creating BigQuery dataset titled: babyweight"
    
    bq --location=US mk --dataset \
        --description "Babyweight" \
        $PROJECT:babyweight
    echo "Here are your current datasets:"
    bq ls
fi
    
## Create GCS bucket if it doesn't exist already...
exists=$(gsutil ls -d | grep -w gs://${BUCKET}/)

if [ -n "$exists" ]; then
    echo -e "Bucket exists, let's not recreate it."
    
else
    echo "Creating a new GCS bucket."
    gsutil mb -l ${REGION} gs://${BUCKET}
    echo "Here are your current buckets:"
    gsutil ls
fi
%%bigquery
CREATE OR REPLACE TABLE
    babyweight.babyweight_data AS
SELECT
    weight_pounds,
    CAST(is_male AS STRING) AS is_male,
    mother_age,
    CASE
        WHEN plurality = 1 THEN "Single(1)"
        WHEN plurality = 2 THEN "Twins(2)"
        WHEN plurality = 3 THEN "Triplets(3)"
        WHEN plurality = 4 THEN "Quadruplets(4)"
        WHEN plurality = 5 THEN "Quintuplets(5)"
    END AS plurality,
    gestation_weeks,
    FARM_FINGERPRINT(
        CONCAT(
            CAST(year AS STRING),
            CAST(month AS STRING)
        )
    ) AS hashmonth
FROM
    publicdata.samples.natality
WHERE
    year > 2000
    AND weight_pounds > 0
    AND mother_age > 0
    AND plurality > 0
    AND gestation_weeks > 0
%%bigquery
CREATE OR REPLACE TABLE
    babyweight.babyweight_augmented_data AS
SELECT
    weight_pounds,
    is_male,
    mother_age,
    plurality,
    gestation_weeks,
    hashmonth
FROM
    babyweight.babyweight_data
UNION ALL
SELECT
    weight_pounds,
    "Unknown" AS is_male,
    mother_age,
    CASE
        WHEN plurality = "Single(1)" THEN plurality
        ELSE "Multiple(2+)"
    END AS plurality,
    gestation_weeks,
    hashmonth
FROM
    babyweight.babyweight_data
%%bigquery
CREATE OR REPLACE TABLE
    babyweight.babyweight_data_train AS
SELECT
    weight_pounds,
    is_male,
    mother_age,
    plurality,
    gestation_weeks
FROM
    babyweight.babyweight_augmented_data
WHERE
    ABS(MOD(hashmonth, 4)) < 3
%%bigquery
CREATE OR REPLACE TABLE
    babyweight.babyweight_data_eval AS
SELECT
    weight_pounds,
    is_male,
    mother_age,
    plurality,
    gestation_weeks
FROM
    babyweight.babyweight_augmented_data
WHERE
    ABS(MOD(hashmonth, 4)) = 3
  
%%bigquery
-- LIMIT 0 is a free query; this allows us to check that the table exists.
SELECT * FROM babyweight.babyweight_data_train
LIMIT 0

%%bigquery
-- LIMIT 0 is a free query; this allows us to check that the table exists.
SELECT * FROM babyweight.babyweight_data_eval
LIMIT 0

# Construct a BigQuery client object.
client = bigquery.Client()

dataset_name = "babyweight"

# Create dataset reference object
dataset_ref = client.dataset(
    dataset_id=dataset_name, project=client.project)

# Export both train and eval tables
for step in ["train", "eval"]:
    destination_uri = os.path.join(
        "gs://", BUCKET, dataset_name, "data", "{}*.csv".format(step))
    table_name = "babyweight_data_{}".format(step)
    table_ref = dataset_ref.table(table_name)
    extract_job = client.extract_table(
        table_ref,
        destination_uri,
        # Location must match that of the source table.
        location="US",
    )  # API request
    extract_job.result()  # Waits for job to complete.

    print("Exported {}:{}.{} to {}".format(
        client.project, dataset_name, table_name, destination_uri))
%%bash
gsutil ls gs://${BUCKET}/babyweight/data/*.csv
%%bash
gsutil ls gs://${BUCKET}/babyweight/data/*000000000000.csv
%%bash

OUTDIR=gs://${BUCKET}/babyweight/trained_model
JOBID=babyweight_$(date -u +%y%m%d_%H%M%S)

gcloud ai-platform jobs submit training ${JOBID} \
    --region=${REGION} \
    --module-name=trainer.task \
    --package-path=$(pwd)/babyweight/trainer \
    --job-dir=${OUTDIR} \
    --staging-bucket=gs://${BUCKET} \
    --master-machine-type=n1-standard-8 \
    --scale-tier=CUSTOM \
    --runtime-version=${TFVERSION} \
    --python-version=${PYTHONVERSION} \
    -- \
    --train_data_path=gs://${BUCKET}/babyweight/data/train*.csv \
    --eval_data_path=gs://${BUCKET}/babyweight/data/eval*.csv \
    --output_dir=${OUTDIR} \
    --num_epochs=10 \
    --train_examples=10000 \
    --eval_steps=100 \
    --batch_size=32 \
    --nembeds=8
 
%%bash
gsutil ls gs://${BUCKET}/babyweight/trained_model
  
%%bash
MODEL_LOCATION=$(gsutil ls -ld -- gs://${BUCKET}/babyweight/trained_model/2* \
                 | tail -1)
gsutil ls ${MODEL_LOCATION}

%%bash
%%bash
gcloud config set ai_platform/region global

%%bash
MODEL_NAME="babyweight"
MODEL_VERSION="ml_on_gcp"
MODEL_LOCATION=$(gsutil ls -ld -- gs://${BUCKET}/babyweight/trained_model/2* \
                 | tail -1 | tr -d '[:space:]')
echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION"
# gcloud ai-platform versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
# gcloud ai-platform models delete ${MODEL_NAME}
gcloud ai-platform models create ${MODEL_NAME} --regions ${REGION}
gcloud ai-platform versions create ${MODEL_VERSION} \
    --model=${MODEL_NAME} \
    --origin=${MODEL_LOCATION} \
    --runtime-version=2.1 \
    --python-version=3.7
