%%bash
pip install --upgrade tensorflow==1.4
pip install --ignore-installed --upgrade pytz==2018.4
pip uninstall -y google-cloud-dataflow
pip install --upgrade apache-beam[gcp]==2.6

# change these to try this notebook out
BUCKET = 'cloud-training-demos-ml'
PROJECT = 'cloud-training-demos'
REGION = 'us-central1'
import os
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION
%bash
gcloud config set project $PROJECT
gcloud config set compute/region $REGION

%%bash
if ! gsutil ls | grep -q gs://${BUCKET}/; then
  gsutil mb -l ${REGION} gs://${BUCKET}
fi

query="""
SELECT
  weight_pounds,
  is_male,
  mother_age,
  plurality,
  gestation_weeks,
  FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING))) AS hashmonth
FROM
  publicdata.samples.natality
WHERE year > 2000
"""

import google.datalab.bigquery as bq
df = bq.Query(query + " LIMIT 100").execute().result().to_dataframe()
df.head()

def get_distinct_values(column_name):
    sql = """
SELECT
  {0},
  COUNT(1) AS num_babies,
  AVG(weight_pounds) AS avg_wt
FROM
  publicdata.samples.natality
WHERE
  year > 2000
GROUP BY
  {0}
    """.format(column_name)
    return bq.Query(sql).execute().result().to_dataframe()
  
df = get_distinct_values('is_male')
df.plot(x='is_male', y='num_babies', kind='bar');
df.plot(x='is_male', y='avg_wt', kind='bar');

df = get_distinct_values('mother_age')
df = df.sort_values('mother_age')
df.plot(x='mother_age', y='num_babies');
df.plot(x='mother_age', y='avg_wt');

df = get_distinct_values('plurality')
df = df.sort_values('plurality')
df.plot(x='plurality', y='num_babies', logy=True, kind='bar');
df.plot(x='plurality', y='avg_wt', kind='bar');

df = get_distinct_values('gestation_weeks')
df = df.sort_values('gestation_weeks')
df.plot(x='gestation_weeks', y='num_babies', logy=True, kind='bar', color='royalblue');
df.plot(x='gestation_weeks', y='avg_wt', kind='bar', color='royalblue');

import apache_beam as beam
import datetime

def to_csv(rowdict):
    # pull columns from BQ and create a line
    import hashlib
    import copy
    CSV_COLUMNS = 'weight_pounds,is_male,mother_age,plurality,gestation_weeks'.split(',')
    
    # create synthetic data where we assume that no ultrasound has been performed
    # and so we don't know sex of the baby. Let's assume that we can tell the difference
    # between single and multiple, but that the errors rates in determining exact number
    # is difficult in the absence of an ultrasound.
    no_ultrasound = copy.deepcopy(rowdict)
    w_ultrasound = copy.deepcopy(rowdict)

    no_ultrasound['is_male'] = 'Unknown'
    if rowdict['plurality'] > 1:
      no_ultrasound['plurality'] = 'Multiple(2+)'
    else:
      no_ultrasound['plurality'] = 'Single(1)'
      
    # Change the plurality column to strings
    w_ultrasound['plurality'] = ['Single(1)', 'Twins(2)', 'Triplets(3)', 'Quadruplets(4)', 'Quintuplets(5)'][rowdict['plurality']-1]
    
    # Write out two rows for each input row, one with ultrasound and one without
    for result in [no_ultrasound, w_ultrasound]:
      data = ','.join([str(result[k]) if k in result else 'None' for k in CSV_COLUMNS])
      key = hashlib.sha224(data).hexdigest()  # hash the columns to form a key
      yield str('{},{}'.format(data, key))
  
def preprocess(in_test_mode):
    job_name = 'preprocess-babyweight-features' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    
    if in_test_mode:
        OUTPUT_DIR = './preproc'
    else:
        OUTPUT_DIR = 'gs://{0}/babyweight/preproc/'.format(BUCKET)
    
    options = {
        'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
        'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
        'job_name': job_name,
        'project': PROJECT,
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'max_num_workers': 3,   # CHANGE THIS IF YOU HAVE MORE QUOTA
        'no_save_main_session': True
    }
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    if in_test_mode:
        RUNNER = 'DirectRunner'
    else:
        RUNNER = 'DataflowRunner'
    p = beam.Pipeline(RUNNER, options=opts)
    query = """
SELECT
  weight_pounds,
  is_male,
  mother_age,
  plurality,
  gestation_weeks,
  FARM_FINGERPRINT(CONCAT(CAST(YEAR AS STRING), CAST(month AS STRING))) AS hashmonth
FROM
  publicdata.samples.natality
WHERE year > 2000
AND weight_pounds > 0
AND mother_age > 0
AND plurality > 0
AND gestation_weeks > 0
AND month > 0
    """
  
    if in_test_mode:
        query = query + ' LIMIT 100' 
  
    for step in ['train', 'eval']:
        if step == 'train':
            selquery = 'SELECT * FROM ({}) WHERE ABS(MOD(hashmonth, 4)) < 3'.format(query)
        else:
            selquery = 'SELECT * FROM ({}) WHERE ABS(MOD(hashmonth, 4)) = 3'.format(query)

        (p 
         | '{}_read'.format(step) >> beam.io.Read(beam.io.BigQuerySource(query=selquery, use_standard_sql=True))
         | '{}_csv'.format(step) >> beam.FlatMap(to_csv)
         | '{}_out'.format(step) >> beam.io.Write(beam.io.WriteToText(os.path.join(OUTPUT_DIR, '{}.csv'.format(step))))
        )
 
    job = p.run()
  
preprocess(in_test_mode=False)

%bash
gsutil ls gs://${BUCKET}/babyweight/preproc/*-00000*
  
import shutil
import numpy as np
import tensorflow as tf

CSV_COLUMNS = 'weight_pounds,is_male,mother_age,plurality,gestation_weeks,key'.split(',')
LABEL_COLUMN = 'weight_pounds'
KEY_COLUMN = 'key'
DEFAULTS = [[0.0], ['null'], [0.0], ['null'], [0.0], ['nokey']]
TRAIN_STEPS = 1000

def read_dataset(prefix, pattern, batch_size=512):
    # use prefix to create filename
    filename = 'gs://{}/babyweight/preproc/{}*{}*'.format(BUCKET, prefix, pattern)
    if prefix == 'train':
        mode = tf.estimator.ModeKeys.TRAIN
        num_epochs = None # indefinitely
    else:
        mode = tf.estimator.ModeKeys.EVAL
        num_epochs = 1 # end-of-input after this
    
    # the actual input function passed to TensorFlow
    def _input_fn():
        # could be a path to one file or a file pattern.
        input_file_names = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(
            input_file_names, shuffle=True, num_epochs=num_epochs)
 
        # read CSV
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(filename_queue, num_records=batch_size)
        if mode == tf.estimator.ModeKeys.TRAIN:
          value = tf.train.shuffle_batch([value], batch_size, capacity=10*batch_size, 
                                         min_after_dequeue=batch_size, enqueue_many=True, 
                                         allow_smaller_final_batch=False)
        value_column = tf.expand_dims(value, -1)
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        features.pop(KEY_COLUMN)
        label = features.pop(LABEL_COLUMN)
        return features, label
  return _input_fn

def get_wide_deep():
    # define column types
    is_male,mother_age,plurality,gestation_weeks = \
        [\
            tf.feature_column.categorical_column_with_vocabulary_list('is_male', 
                        ['True', 'False', 'Unknown']),
            tf.feature_column.numeric_column('mother_age'),
            tf.feature_column.categorical_column_with_vocabulary_list('plurality',
                        ['Single(1)', 'Twins(2)', 'Triplets(3)',
                         'Quadruplets(4)', 'Quintuplets(5)','Multiple(2+)']),
            tf.feature_column.numeric_column('gestation_weeks')
        ]

    # discretize
    age_buckets = tf.feature_column.bucketized_column(mother_age, 
                        boundaries=np.arange(15,45,1).tolist())
    gestation_buckets = tf.feature_column.bucketized_column(gestation_weeks, 
                        boundaries=np.arange(17,47,1).tolist())
      
    # sparse columns are wide 
    wide = [is_male,
            plurality,
            age_buckets,
            gestation_buckets]
    
    # feature cross all the wide columns and embed into a lower dimension
    crossed = tf.feature_column.crossed_column(wide, hash_bucket_size=20000)
    embed = tf.feature_column.embedding_column(crossed, 3)
    
    # continuous columns are deep
    deep = [mother_age,
            gestation_weeks,
            embed]
    return wide, deep
  
  def serving_input_fn():
    feature_placeholders = {
        'is_male': tf.placeholder(tf.string, [None]),
        'mother_age': tf.placeholder(tf.float32, [None]),
        'plurality': tf.placeholder(tf.string, [None]),
        'gestation_weeks': tf.placeholder(tf.float32, [None])
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)
  
  from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.contrib.learn.python.learn import learn_runner

PATTERN = "00000-of-"  # process only one of the shards, for testing purposes

def train_and_evaluate(output_dir):
    wide, deep = get_wide_deep()
    estimator = tf.estimator.DNNLinearCombinedRegressor(
                         model_dir=output_dir,
                         linear_feature_columns=wide,
                         dnn_feature_columns=deep,
                         dnn_hidden_units=[64, 32])
    train_spec=tf.estimator.TrainSpec(
                         input_fn=read_dataset('train', PATTERN),
                         max_steps=TRAIN_STEPS)
    exporter = tf.estimator.FinalExporter('exporter',serving_input_fn)
    eval_spec=tf.estimator.EvalSpec(
                         input_fn=read_dataset('eval', PATTERN),
                         steps=None,
                         exporters=exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
shutil.rmtree('babyweight_trained', ignore_errors=True) # start fresh each time
train_and_evaluate('babyweight_trained')

%bash
grep "^def" babyweight/trainer/model.py

%bash
echo "bucket=${BUCKET}"
rm -rf babyweight_trained
export PYTHONPATH=${PYTHONPATH}:${PWD}/babyweight
python -m trainer.task \
   --bucket=${BUCKET} \
   --output_dir=babyweight_trained \
   --job-dir=./tmp \
--pattern="00000-of-" --train_steps=1000

%bash
OUTDIR=gs://${BUCKET}/babyweight/trained_model
JOBNAME=babyweight_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
#gsutil -m rm -rf $OUTDIR
gcloud ml-engine jobs submit training $JOBNAME \
   --region=$REGION \
   --module-name=trainer.task \
   --package-path=$(pwd)/babyweight/trainer \
   --job-dir=$OUTDIR \
   --staging-bucket=gs://$BUCKET \
   --scale-tier=STANDARD_1 \
   --runtime-version 1.4 \
   -- \
   --bucket=${BUCKET} \
   --output_dir=${OUTDIR} \
   --train_steps=100000
  
  from google.datalab.ml import TensorBoard
TensorBoard().start('gs://{}/babyweight/trained_model'.format(BUCKET))

for pid in TensorBoard.list()['pid']:
    TensorBoard().stop(pid)
    print('Stopped TensorBoard with pid {}'.format(pid))
    
%bash
gsutil ls gs://${BUCKET}/babyweight/trained_model/export/exporter
  
%bash
MODEL_NAME="babyweight"
MODEL_VERSION="soln"
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/babyweight/trained_model/export/exporter/ | tail -1)
echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"
#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version 1.4

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import json

credentials = GoogleCredentials.get_application_default()
api = discovery.build('ml', 'v1', credentials=credentials)

request_data = {'instances':
  [
    {
      'is_male': 'True',
      'mother_age': 26.0,
      'plurality': 'Single(1)',
      'gestation_weeks': 39
    },
    {
      'is_male': 'False',
      'mother_age': 29.0,
      'plurality': 'Single(1)',
      'gestation_weeks': 38
    },
    {
      'is_male': 'True',
      'mother_age': 26.0,
      'plurality': 'Triplets(3)',
      'gestation_weeks': 39
    },
    {
      'is_male': 'Unknown',
      'mother_age': 29.0,
      'plurality': 'Multiple(2+)',
      'gestation_weeks': 38
    },
  ]
}

parent = 'projects/%s/models/%s/versions/%s' % (PROJECT, 'babyweight', 'soln')
response = api.projects().predict(body=request_data, name=parent).execute()
print(json.dumps(response, sort_keys = True, indent = 4))
