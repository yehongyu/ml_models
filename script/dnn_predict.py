#coding=utf-8
import tensorflow as tf
import numpy as np

def get_random_input_dict(num_cols, num_examples):
    x_map = {}
    for i in range(num_cols):
        fname = "slot_%s"%(i)
        x_map[fname] = np.arange(num_examples) / 10.0
    #print(x_map['slot_0'])
    return x_map

def predict(model_path, input_dict):
    predict_fn = tf.contrib.predictor.from_saved_model(model_path, signature_def_key="predict")
    predictions = predict_fn(input_dict)
    print("predict_by_predictor-probabilities:", np.asarray(predictions['probabilities'][:,1]))
    for key in predictions:
        print(key, predictions[key])

def load_data(in_file):
    data = {}
    return data


model_path = "/Users/aodandan/data/model/saved_model/1591186929"
batch_size = 64
in_file = ""
if __name__ == "__main__":
    print('start load data...')
    data = get_random_input_dict(242, 100)
    data = load_data(in_file)

    print('start load tf model...')
    predict_fn = tf.contrib.predictor.from_saved_model(model_path, signature_def_key="predict")
    print('start predict...')
    results = predict_fn(data)
    scores = np.asarray(results['probabilities'][:,1])
    print('scores:', len(scores))

