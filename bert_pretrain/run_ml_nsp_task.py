import os
import sys
import time
import logging
import numpy as np

import tensorflow as tf
print("tf version:", tf.__version__)

cur_ts = int(time.time())
tf.app.flags.DEFINE_string("train_paths", None, "HDFS paths to input files.")
tf.app.flags.DEFINE_string("eval_paths", "/Users/aodandan/data/tfrecord/eval/part-*", "eval data path")
tf.app.flags.DEFINE_string("model_path", "/Users/aodandan/data/model/dnn_estimator/", "Where to write output files.")
tf.app.flags.DEFINE_string("last_model_path", "", "Model path for the previous run.")
tf.app.flags.DEFINE_integer("train_epochs", 1, "train epochs")
tf.app.flags.DEFINE_integer("batch_size", 512, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "train learning rate")
tf.app.flags.DEFINE_float("dropout", 0.5, "dropout")
tf.app.flags.DEFINE_float("clip_norm", 10.0, "clip norm")
tf.app.flags.DEFINE_integer("num_cols", 242, "num cols")
tf.app.flags.DEFINE_string('f', '', 'kernel')


FLAGS = tf.app.flags.FLAGS
##print(FLAGS.train_paths)
print(FLAGS.model_path)
print(FLAGS.train_paths)

def build_feature_columns():
    columns = []
    for i in range(FLAGS.num_cols):
        num_column = tf.feature_column.numeric_column("slot_%s"%i)
        columns.append(num_column)
    return columns

def build_model(FLAGS):
    print(FLAGS.model_path)
    print(FLAGS.last_model_path)
    print(FLAGS.learning_rate)
    print(FLAGS.clip_norm)
    print(FLAGS.num_cols)
    print(FLAGS.dropout)
    checkpoint_dir = FLAGS.model_path
    if FLAGS.last_model_path and not tf.train.latest_checkpoint(checkpoint_dir):
        warmup_dir = FLAGS.last_model_path
    else:
        warmup_dir = None

    my_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, FLAGS.clip_norm)

    #DNNClassifier: Loss is calculated by using softmax cross entropy.

    print("start build model")
    model = tf.estimator.DNNClassifier(
        feature_columns=build_feature_columns(),
        hidden_units=[256, 64],
        optimizer = my_optimizer,
        n_classes = 2,
        dropout=FLAGS.dropout,
        config=tf.estimator.RunConfig(model_dir=checkpoint_dir),
        warm_start_from=warmup_dir)
    print("build model end")
    return model

def serving_input_receiver_fn():
    features = {}
    for i in range(FLAGS.num_cols):
        fname = "slot_%s"%(i)
        features[fname] = tf.placeholder(tf.float32, shape=[None], name=fname)
    return tf.estimator.export.ServingInputReceiver(features, features)

def read_data(paths, batch_size=512, num_epochs=1, shuffle=False, buffer_size=50000, num_cols=242, num_parallels=1, num_workers=1, worker_index=0):
    def parse(value):
        desc = {
                'slot_%s'%i: tf.FixedLenFeature([1], tf.float32, default_value=0.0) for i in range(0, num_cols)
            }
        desc["label"] = tf.FixedLenFeature([1], tf.int64, default_value=0)
        example = tf.parse_single_example(value, desc)
        label = example["label"]
        label = tf.cast(label,tf.int32)
        del example["label"]
        return example, label

    print('paths:', paths)
    data_files = tf.data.Dataset.list_files(paths)
    
    dataset = tf.data.TFRecordDataset(data_files, num_parallel_reads=num_parallels)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    # dataset = dataset.shard(num_workers, worker_index)

    return dataset.map(parse, num_parallel_calls=num_parallels) \
                  .repeat(num_epochs).batch(batch_size) \
                  .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def train_input_fn():
    return read_data(FLAGS.train_paths,
                     batch_size=FLAGS.batch_size,
                     num_epochs=FLAGS.train_epochs,
                     shuffle=True,
                     num_cols=FLAGS.num_cols,
                     num_parallels=1)

def eval_input_fn():
    return read_data(FLAGS.eval_paths,
                     batch_size=FLAGS.batch_size,
                     num_cols=FLAGS.num_cols,
                     num_parallels=1)

def load_model_and_print_variable():
    path = FLAGS.model_path
    init_vars = tf.train.list_variables(path)
    for name, shape in init_vars:
        array = tf.train.load_variable(path, name)
        print(name, shape)
 
def find_variable():
    # TODO: find the two variable:
    model = build_model(FLAGS)
    weight_name = 'dnn/logits/kernel:0' # dnn/logits/bias, dnn/head/beta1_power, dnn/hiddenlayer_0/bias
    score_tensor_name='dnn/head/predictions/logistic:0'
    label_tensor_name='IteratorGetNext:%s' % (FLAGS.num_cols)
    
    graph = tf.compat.v1.get_default_graph()
    score_tensor = graph.get_tensor_by_name(weight_name)
    sess = tf.compat.v1.Session(graph=graph)
    sess.run(model)
    print("score_tensor:", sess.run(score_tensor))

def get_x_map():
    x_map = {}
    for i in range(FLAGS.num_cols):
        fname = "slot_%s"%(i)
        #x_map[fname] = np.arange(2)
        x_map[fname] = [0, 1]
    return x_map

def start_train():
    FLAGS.train_paths = ["/Users/aodandan/data/tfrecord/train/part-*", "/Users/aodandan/data/tfrecord/eval/part-*"]
    #FLAGS.train_paths = "/Users/aodandan/data/tfrecord/{train,eval}/part-*"

    model = build_model(FLAGS)
    
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=6000000)
    
    feature_spec = tf.feature_column.make_parse_example_spec(build_feature_columns())
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    exporter = tf.estimator.FinalExporter('gandalf', export_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=100, throttle_secs=10, exporters=[exporter])
    
    #print("estimator-variable names:", tf.estimator.get_variable_names())
    print("start to train")
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    
    print("start to save model")
    model.export_saved_model(FLAGS.model_path + '/saved_model',
                serving_input_receiver_fn=serving_input_receiver_fn)
    
    x_map = get_x_map()
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(x=x_map, shuffle=False)
    pred_results = model.predict(input_fn=pred_input_fn)
    print("AA pred_results:", type(pred_results))
    for pred_dict in pred_results:
        score = pred_dict['probabilities'][1]
        print("AA pred_score:", score)
            
    print("finish save model")

def predict_with_build_model():
    FLAGS.model_path = "/Users/aodandan/data/model"
    estimator = build_model(FLAGS)        

    # predict with the model and print results
    x_map = get_x_map()
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(x=x_map, shuffle=False)
    
    pred_results = estimator.predict(input_fn=pred_input_fn)
    print("pred_results:", type(pred_results))
    for pred_dict in pred_results:
        score = pred_dict['probabilities'][1]
        print("pred_score:", score)

def predict_by_predictor():
    model_path = "/Users/aodandan/data/model/saved_model/1591186929"
    predict_fn = tf.contrib.predictor.from_saved_model(model_path, signature_def_key="predict")
    predictions = predict_fn(get_x_map())
    print("predict_by_predictor-probabilities:", np.asarray(predictions['probabilities'][:,1]))
    for key in predictions:
        print(key, predictions[key])
    

def simple_linear_train():
    input_columns = []
    input_columns.append(tf.feature_column.numeric_column("x"))
    input_columns.append(tf.feature_column.numeric_column("y"))

    estimator = tf.estimator.LinearClassifier(feature_columns=input_columns)

    def input_fn():
        return tf.data.Dataset.from_tensor_slices(
            ({"x": [1., 2., 3., 4.], "y": [1., 2., 3., 4.]}, [1, 1, 0, 0])).repeat(200).shuffle(64).batch(16)
    estimator.train(input_fn)

    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    tf.feature_column.make_parse_example_spec(input_columns))
    estimator_base_path = os.path.join(FLAGS.model_path, 'from_estimator')
    estimator_path = estimator.export_saved_model(estimator_base_path, serving_input_fn)
    return estimator_path

def predict_by_load():
    model_path = "/Users/aodandan/data/model/saved_model/1591186929"
    #model = tf.compat.v2.saved_model.load(model_path)
    model = tf.compat.v1.saved_model.load_v2(model_path)
    print(list(model.signatures.keys()))
    model_fn = model.signatures['predict']
    print(model_fn.structured_outputs)
    
    example = tf.train.Example()
    for i in range(FLAGS.num_cols):
        fname = "slot_%s"%(i)
        example.features.feature[fname].float_list.value.extend([0.0])
    #print(example)
    predictions=model_fn(examples=tf.constant([example.SerializeToString()]))
    print(predictions)

def simple_linear_test():
    estimator_path = "/Users/aodandan/data/model/from_estimator/1591239059"
    print("estimator_path:", estimator_path)
    imported = tf.compat.v1.saved_model.load_v2(estimator_path)
    print(list(imported.signatures.keys()))
    example = tf.train.Example()
    example.features.feature["x"].float_list.value.extend([1.5])
    example.features.feature["y"].float_list.value.extend([1.5])
    print(example)
    result = imported.signatures["predict"](
            examples=tf.constant([example.SerializeToString()]))
    print(result)
    
#start_train()           
#predict_with_build_model()
predict_by_predictor()
#predict_by_load()
#simple_linear_test()
