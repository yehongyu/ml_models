#coding=utf-8
import time
import numpy as np
import traceback
from sklearn import metrics
import tensorflow as tf
print("tf version:", tf.__version__)

cur_ts = str(int(time.time()))
tf.app.flags.DEFINE_string("train_paths", "/Users/aodandan/data/tfrecord/train/part-*", "HDFS paths to input files.")
tf.app.flags.DEFINE_string("eval_paths", "/Users/aodandan/data/tfrecord/eval/part-00093", "eval data path")
tf.app.flags.DEFINE_string("model_path", "/Users/aodandan/data/model/dnn_simple/"+cur_ts, "Where to write output files.")
tf.app.flags.DEFINE_string("last_model_path", "/Users/aodandan/data/model/dnn_simple/1591793607/ckpt/dnn-0", "Model path for the previous run.")
tf.app.flags.DEFINE_integer("train_epochs", 5, "train epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "train learning rate")
tf.app.flags.DEFINE_float("dropout", 0.5, "dropout")
tf.app.flags.DEFINE_float("clip_norm", 10.0, "clip norm")
tf.app.flags.DEFINE_integer("num_cols", 264, "num cols")
tf.app.flags.DEFINE_string('f', '', 'kernel')

FLAGS = tf.app.flags.FLAGS
print(FLAGS.model_path)
print(FLAGS.train_paths)

#if tf.python.eager.context.executing_eagerly():
#    print("Eager execution is enabled.")

# for binary classifier：non one-hot for labels and logits
def log_loss(logits, labels):  
    # labels: [batch_size, 1]
    # logits: [batch_size, 1], output of nn, before sigmoid
    probabilities = tf.nn.sigmoid(logits)
    logs = tf.losses.log_loss(labels=labels, predictions=probabilities)
    cost = tf.reduce_mean(logs) # average of reduce_sum
    return logs


def build_iterator(paths, shuffle=True, num_cols=264, batch_size=2, buffer_size = 8 * 1024 * 1024, num_parallels=1):
    def parse(value):
        desc = {
                'slot_%s'%i: tf.FixedLenFeature([1], tf.float32, default_value=0.0) for i in range(0, num_cols)
            }
        desc["label"] = tf.FixedLenFeature([1], tf.int64, default_value=0)
        example = tf.parse_single_example(value, desc)
        label = example["label"]
        label = tf.cast(label,tf.int32)
        del example["label"]
        instance = []
        for i in range(num_cols):
            instance.append(example['slot_%s'%i])
        return instance, label

    print('paths:', paths)
    data_files = tf.data.Dataset.list_files(paths, shuffle=True)
    dataset = tf.data.TFRecordDataset(data_files, buffer_size=buffer_size, 
                                          num_parallel_reads=num_parallels)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(parse, num_parallel_calls=num_parallels).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=2 * batch_size)
    return dataset.make_initializable_iterator()

X = tf.placeholder(tf.float32, shape=(None, FLAGS.num_cols, 1), name='X') #[batch_size, n_feature]
Y = tf.placeholder(tf.int32, shape=(None, 1), name='Y') #[batch_size, 1]

def build_dnn_model(lr=0.001, device="/cpu:0", dropout=0.5, n_feature=264, n_classes=2):
    width_1th_layer = 256
    width_2th_layer = 64
    width_3th_layer = n_classes
    ## device: "/cpu:0", "/gpu:{gpu_idx}"

    with tf.name_scope("dnn") as scope, tf.device(device):
        input_X = tf.reshape(tf.compat.v1.squeeze(X), [-1, n_feature])
        onehot_Y = tf.reshape(tf.one_hot(tf.compat.v1.squeeze(Y), depth=n_classes), [-1, n_classes])
        labels = onehot_Y[:,1]
        dropout_prob = tf.constant(dropout, dtype=tf.float32, name="dropout_prob")

        w1 = tf.get_variable(name="w1", shape=[n_feature, width_1th_layer],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name="b1", shape=[1, width_1th_layer], initializer=tf.compat.v1.zeros_initializer)

        w2 = tf.get_variable(name="w2", shape=[width_1th_layer, width_2th_layer],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name="b2", shape=[1, width_2th_layer], initializer=tf.compat.v1.zeros_initializer)

        w3 = tf.get_variable(name="w3", shape=[width_2th_layer, width_3th_layer],
                             initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable(name="b3", shape=[1, width_3th_layer], initializer=tf.compat.v1.zeros_initializer)

        z1 = tf.add(tf.matmul(input_X, w1), b1)
        a1 = tf.nn.relu(z1)
        z2 = tf.add(tf.matmul(a1, w2), b2)
        a2 = tf.nn.relu(z2)
        logits = tf.add(tf.matmul(a2, w3), b3)
        probabilities = tf.nn.softmax(logits) # [batch_size, n_classes]

        loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=onehot_Y
                    )
        loss_rsum = tf.reduce_sum(loss)
        loss_rmean = tf.reduce_mean(loss)

        loss_val, loss_op = tf.compat.v1.metrics.mean(loss, name='mean_loss_metric')
        auc_val, auc_op = tf.compat.v1.metrics.auc(labels, probabilities[:,1], name="auc_metric")
        pred_class = tf.cast(tf.round(probabilities[:,1]), tf.int32)
        acc_val, acc_op = tf.compat.v1.metrics.accuracy(labels, pred_class, name="acc_metric")

        tf.summary.scalar('loss', loss_val)
        tf.summary.scalar('auc', auc_val)
        tf.summary.scalar('acc', acc_val)
        summary_merged = tf.summary.merge_all()

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in variables: print("model global_variable:", v.name, v)
        train_vars = tf.trainable_variables()
        for v in train_vars: print("model train_var:", v.name, v)
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss_rmean, train_vars), clip_norm=5)
        optimizer = tf.train.AdamOptimizer(lr)
        #optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(grads, train_vars))

        task_ops = {
            "train_op": train_op,
            "loss_op": loss_op,
            "auc_op": auc_op,
            "acc_op": acc_op,
            "summary": summary_merged,
            "loss": loss,
            "loss_rsum": loss_rsum,
            "loss_rmean": loss_rmean,
            "logits": logits,
            "probabilities": probabilities,
            "pred_class": pred_class,
            #"labels": labels,
            #"loss_val": loss_val,
            #"auc_val": auc_val,
            #"acc_val": acc_val,
            #"grads": grads,
            #"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3  
        }
        return task_ops, train_vars

def reset_running_variables(sess, scope):
    #Isolate the variables stored behind the scenes by the metric operation
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
    #for run_var in running_vars:
    #    print("get run_var:{}, val={}.".format(run_var.name, sess.run(run_var)), run_var)

    # Define initializer to initialize/reset running variables
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    # initialize/reset the running variables
    sess.run(running_vars_initializer)
    #for run_var in running_vars:
    #    print("reset run_var:{}, val={}.".format(run_var.name, sess.run(run_var)), run_var)

def cal_sklearn_auc(all_y, all_pred, batch_labels, batch_probs):
    all_y.extend(batch_labels)
    all_pred.extend(batch_probs)
    auc = metrics.roc_auc_score(all_y, all_pred)
    acc = metrics.accuracy_score(all_y, np.around(all_pred).astype(int))
    return auc, acc

def train_one_epoch(sess, next_element, epoch, log_writer, step, task_ops):

    reset_running_variables(sess, "dnn/mean_loss_metric") # accumulative, reset for each epoch
    reset_running_variables(sess, "dnn/auc_metric") # approximate, lower than sklearn auc
    reset_running_variables(sess, "dnn/acc_metric")
    all_y = []
    all_pred = []
    while True:
        try:
            batch_instances, batch_labels = sess.run(next_element)
            feed = {X:batch_instances, Y:batch_labels}

            results = sess.run(task_ops, feed_dict=feed)
            auc, acc = cal_sklearn_auc(all_y, all_pred, batch_labels, results["probabilities"][:,1])
            '''
            for name, item in results.items():
                if name in ['loss', 'pred_class', 'probabilities', 'logits']:
                    print("res:", name, type(item), np.shape(item))
                else:
                    print("res:", name, type(item), np.shape(item), item)
            '''    
            step += 1
            log_writer.add_summary(results['summary'], step)

            if step % 100 == 0:
                print("Epoch-{}, step-{}: batch_loss={}, loss={}, auc={}, acc={}, tf_auc={}, tf_acc={}".format(
                    epoch, step, results["loss_rmean"], results["loss_op"], auc, acc, results["auc_op"], results["acc_op"])
                     )

        except tf.errors.OutOfRangeError:
            print("Consumed all examples.")
            break
        except Exception as e:
            err_msg = traceback.format_exc()
            print("err:", err_msg)
            break
    return step

def train():
    log_path = FLAGS.model_path + "/log/" # tensorboard -–logdir
    checkpoint_path = FLAGS.model_path + "/ckpt/dnn"

    graph = tf.get_default_graph()
    sess_config = tf.ConfigProto()
    sess_config.log_device_placement = True # log device placement
    sess_config.gpu_options.allow_growth = True # dynamic allocate mem
    sess_config.allow_soft_placement = True # auto select device

    task_ops, train_vars = build_dnn_model(lr=FLAGS.learning_rate, dropout=FLAGS.dropout, n_feature=FLAGS.num_cols)
    iterator = build_iterator(FLAGS.train_paths, num_cols=FLAGS.num_cols, batch_size=FLAGS.batch_size, shuffle=True)
    next_element = iterator.get_next()

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.train_epochs)

    with tf.Session(config=sess_config) as sess:
        if FLAGS.last_model_path:
            print("Restore variable from file:", FLAGS.last_model_path)
            saver.restore(sess, FLAGS.last_model_path)
        else:
            print("Init global variable:", [v.name for v in tf.global_variables()])
            sess.run(tf.global_variables_initializer())
        print("Init local variable:", [v.name for v in tf.local_variables()])
        sess.run(tf.local_variables_initializer())

        log_writer = tf.summary.FileWriter(log_path, sess.graph)
        step = 0
        for epoch in range(FLAGS.train_epochs):

            sess.run(iterator.initializer)
            step = train_one_epoch(sess, next_element, epoch, log_writer, step, task_ops)

            last_ckpt_path = saver.save(sess, checkpoint_path, global_step=step)
            print("Store model to", last_ckpt_path)

            # eval 
        log_writer.close()

train()
