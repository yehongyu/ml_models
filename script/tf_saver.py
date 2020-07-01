import sys
import tensorflow as tf

ckpt_path = './ckpt/test-model.ckpt'
def save():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Create some variables.
    v1 = tf.Variable([1.0, 2.3], name="v1")
    v2 = tf.Variable(55.5, name="v2")

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, save the
    # variables to disk.
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        save_path = saver.save(sess, ckpt_path, global_step=1)
        print("Model saved in file: %s" % save_path)

def load_model():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Create some variables.
    v1 = tf.Variable([11.0, 16.3], name="v1")
    v2 = tf.Variable(33.5, name="v2")

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    # Restore variables from disk.
    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt_path + '-'+ str(1))
        print("Model restored.")
        print(sess.run(v1))
        print(sess.run(v2))

def load_variable():
    v1 = tf.Variable([444.0, 666.3], name="v1")
    variable_list = tf.get_default_graph().get_collection('variables')
    variable_dict = {}
    for var_tensor in variable_list:
        variable_dict[var_tensor.name] = var_tensor # name has index, like "v1:0"
        print("Defined var {}, type {}, tensor {}".format(var_tensor.name, type(var_tensor), var_tensor))

    tf_path = ckpt_path + '-' + str(1)
    init_vars_list = tf.train.list_variables(tf_path)
    init_vars_map = {}
    for name, shape in init_vars_list:
        # init_vars_list: tuple(name:str, shape:list); name without index, like "v1"
        array_val = tf.train.load_variable(tf_path, name) # return A numpy ndarray
        init_vars_map[name] = array_val
        print("Loaded TF weight {} with shape {}, val {}".format(name, shape, array_val))

    assign_ops = []
    for var_key in variable_dict.keys():
        var_name = var_key.split(':')[0]
        if var_name in init_vars_map:
            print("init data from checkpoint:%s" % var_name, init_vars_map[var_name])
            assign_op = variable_dict[var_key].assign(init_vars_map[var_name])
            assign_ops.append(assign_op)

    print(len(assign_ops), assign_ops)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(v1))
        #assign_op = v1.assign(init_vars_map['v1'])
        for op in assign_ops: sess.run(op)
        print(sess.run(v1))


if __name__ == "__main__":
    if sys.argv[1] == 'save':
        save()
    elif sys.argv[1] == 'load_model':
        load_model()
    elif sys.argv[1] == 'load_variable':
        load_variable()
