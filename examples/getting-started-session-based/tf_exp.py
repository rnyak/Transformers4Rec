import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import glob

from nvtabular.loader.tensorflow import KerasSequenceLoader

from transformers4rec import tf as tr
from transformers4rec.tf.ranking_metric import NDCGAt, RecallAt

import cudf
import dask_cudf
from cudf.io.parquet import ParquetWriter as pwriter_cudf
from dask_cudf.io.parquet import CudfEngine


from merlin_standard_lib import Schema
SCHEMA_PATH = "schema.pb"
schema = Schema().from_proto_text(SCHEMA_PATH)


# # You can select a subset of features for training
schema = schema.select_by_name(['item_id-list_trim', 
                                'category-list_trim'])

#'timestamp/age_days-list_trim', 'timestamp/weekday/sin-list_trim'

inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        masking="mlm",
)

# # Define XLNetConfig class and set default parameters for HF XLNet config  
transformer_config = tr.XLNetConfig.build(
    d_model=64, n_head=4, n_layer=2, total_seq_length=20
)
# # Define the model block including: inputs, masking, projection and transformer block.
body = tr.SequentialBlock(
    [inputs, tr.MLPBlock([64]), tr.TransformerBlock(transformer_config, masking=inputs.masking)]
)

# # Defines the evaluation top-N metrics and the cut-offs
metrics = (
    NDCGAt(top_ks=[1, 5, 20, 40], labels_onehot=True),  
    RecallAt(top_ks=[1, 5, 20, 40], labels_onehot=True)
          )

# # # link task to body and generate the end-to-end keras model
task = tr.NextItemPredictionTask(weight_tying=True, metrics=metrics)
 
model = task.to_model(body=body)
x_cat_names, x_cont_names = ['category-list_trim', 'item_id-list_trim'], []

# # dictionary representing max sequence length for column
sparse_features_max = {
    fname: 20
    for fname in x_cat_names + x_cont_names
}

def get_dataloader(paths_or_dataset, batch_size=64):
    dataloader = KerasSequenceLoader(
        paths_or_dataset,
        batch_size=batch_size,
        label_names=None,
        cat_names=x_cat_names,
        cont_names=x_cont_names,
        sparse_names=list(sparse_features_max.keys()),
        sparse_max=sparse_features_max,
        sparse_as_dense=True,
    )
    return dataloader.map(lambda X, y: (X, []))

import tensorflow as tf
# lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate=0.0005,
#     decay_steps=1.5)
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer="adam", run_eagerly=True)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./sessions_by_day")

import warnings
warnings.filterwarnings('ignore')

start_time_window_index = 1
final_time_window_index = 2
#Iterating over days of one week
for time_index in range(start_time_window_index, final_time_window_index):
    # Set data 
    time_index_train = time_index
    time_index_eval = time_index + 1
    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
    print(train_paths)

    # Train on day related to time_index 
    print('*'*20)
    print("Launch training for day %s are:" %time_index)
    print('*'*20 + '\n')
    train_loader = get_dataloader(train_paths) 
    losses = model.fit(train_loader, epochs=2, verbose=0, )
    model.reset_metrics()
    print('finished')
    # Evaluate on the following day
    eval_loader = get_dataloader(eval_paths) 
    eval_metrics = model.evaluate(eval_loader, return_dict=True)
    print('*'*20)
    print("Eval results for day %s are:\t" %time_index_eval)
    print('\n' + '*'*20 + '\n')
    for key in sorted(eval_metrics.keys()):
        print(" %s = %s" % (key, str(eval_metrics[key]))) 


# model.save('./tmp/tensorflow')

# # ### Reloads the model
# # custom_objects = dict([(e.__class__.__name__, e) for e in task.metrics])
# # custom_objects.update({"Model": tr.model.base.Model})
# model = tf.keras.models.load_model('./tmp/tensorflow')
# model.compile(optimizer="adam", run_eagerly=True)
# batch = next(iter(eval_loader))
# # model(batch[0])

# # # Evaluate on the following day
# time_index_eval = 5
# eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
# eval_loader = get_dataloader(eval_paths) 
# eval_metrics = model.evaluate(eval_loader, return_dict=True, )

# print('*'*20)
# print("Eval results for day %s are:\t" %time_index_eval)
# print('\n' + '*'*20 + '\n')
# for key in sorted(eval_metrics.keys()):
#     print(" %s = %s" % (key, str(eval_metrics[key]))) 

import nvtabular as nvt
workflow = nvt.Workflow.load("workflow_etl")

from nvtabular.inference.triton import export_tensorflow_ensemble
export_tensorflow_ensemble(model, workflow, 'tf4rec', '/workspace/models/tf4rec/', ['session_id', 'day-first', 'item_id-count'])