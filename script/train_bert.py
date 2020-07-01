import os
import sys
import time
import logging
import numpy as np
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s:%(message)s')
logging.info(f"sys path:{os.sys.path}")
logging.info(f"tf version:{tf.__version__}")

cur_ts = str(int(time.time()))
tf.app.flags.DEFINE_string("input_path", "/Users/aodandan/data/tfrecord/bert_data/train_tfrecord/part*", "input files.")
tf.app.flags.DEFINE_string("model_path", "/Users/aodandan/data/tfrecord/bert_data/models/"+cur_ts, "model path")
tf.app.flags.DEFINE_string("vocab_path", "/Users/aodandan/data/tfrecord/bert_data/vocab/vocab.txt", "vocab path")

tf.app.flags.DEFINE_integer("batch_size", 2, "batch size")
tf.app.flags.DEFINE_integer("seq_len", 128, "seq_len size")
tf.app.flags.DEFINE_integer("masked_len", 20, "masked_len")
tf.app.flags.DEFINE_integer("max_train_steps", 100*1000, "max_train_steps")
tf.app.flags.DEFINE_boolean("use_nsp", True, "use next sentence prediction loss")

tf.app.flags.DEFINE_string('f', '', 'kernel')

FLAGS = tf.app.flags.FLAGS
logging.info(f"input_path:{FLAGS.input_path}")
logging.info(f"model_path:{FLAGS.model_path}")
logging.info(f"vocab_path:{FLAGS.vocab_path}")
logging.info(f"batch_size:{FLAGS.batch_size}")

class DataIterator(object):
    def __init__(self, paths, batch_size, seq_len, masked_len):
        self.dataset = self.generate_dataset(paths, batch_size, seq_len, masked_len)
        #self.iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)
        #self.initializer = self.iterator.initializer
        #self.next_element = self.iterator.get_next()

    def input_fn():
        return self.dataset
    
    def generate_dataset(self, paths, batch_size=2, seq_len=128, masked_len=20, shuffle=True, buffer_size = 8 * 1024 * 1024, num_parallels=1):
        def parse(value):
            desc = {
                "input_ids": tf.io.FixedLenFeature([seq_len], tf.int64, default_value=0),
                "input_mask": tf.io.FixedLenFeature([seq_len], tf.int64, default_value=0),
                "segment_ids": tf.io.FixedLenFeature([seq_len], tf.int64, default_value=0),
                "masked_lm_positions": tf.io.FixedLenFeature([masked_len], tf.int64, default_value=0),
                "masked_lm_ids": tf.io.FixedLenFeature([masked_len], tf.int64, default_value=0),
                "masked_lm_weights": tf.io.FixedLenFeature([masked_len], tf.float32, default_value=0.0),
                "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64, default_value=0)
                }
            example = tf.io.parse_single_example(value, desc)
            for name in example.keys():
                value = example[name]
                if value.dtype == tf.int64:
                    value = tf.dtypes.cast(value, tf.int32)
                    example[name] = value
            return example

        logging.info("Build iterator from file: {}".format(paths))
        data_files = tf.data.Dataset.list_files(paths, shuffle=shuffle)
        dataset = tf.data.TFRecordDataset(data_files, buffer_size=buffer_size,
                                          num_parallel_reads=num_parallels)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.map(parse, num_parallel_calls=num_parallels).batch(batch_size)
        dataset = dataset.prefetch(buffer_size=2 * batch_size)
        return dataset
    
class MyBertConfig():
    def __init__(self):
        # model config
        self.vocab_size = 81216
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_act = "gelu"
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.use_bias = True
        
        # train params
        self.dropout = 0.1
        self.clip_norm = 10.0
        self.use_nsp = True
        self.batch_size = 2
        self.seq_len = 128
        
        # optimizer params
        self.learning_rate = 1e-4

def get_initializer():
    return tf.compat.v1.truncated_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32)

class EmbeddingLookup(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
    def build(self, input_shape):
        self.embedding_table = self.add_weight(
                    name="word_embedding",
                    shape=[self.vocab_size, self.embedding_size],
                    initializer=get_initializer(), dtype=tf.float32)
        self.build = True
    def call(self, inputs):
        output = tf.gather(self.embedding_table, inputs)
        return output, self.embedding_table

class LayerProcess(tf.keras.layers.Layer):
    def __init__(self, sequence, dropout_prob=0.1, norm_type="layer", epsilon=1e-8, axis=-1):
        super(LayerProcess, self).__init__()
        self.sequence = sequence
        self.dropout_prob = dropout_prob
        self.norm_type = norm_type # batch or layer
        self.axis = axis
        self.epsilon = epsilon
        self.zero_add_gamma = None
        self.norm_layer = None
        if self.sequence != "none":
            if "z" in self.sequence:
                self.zero_add_gamma = self.add_weight("gamma", shape=(),
                    initializer=tf.zeros_initializer(), dtype=tf.float32)
            if "n" in self.sequence:
                if self.norm_type == "batch":
                    self.norm_layer = tf.keras.layers.BatchNormalization(
                    axis=axis, epsilon=epsilon, name="batch_normalization")
                elif self.norm_type == "layer":
                    self.norm_layer = tf.keras.layers.LayerNormalization(
                    axis=axis, epsilon=epsiilon, name="layer_normalization")
                
    def call(self, inputs):
        previous_value, input_tensor = inputs
        if self.sequence == "none": return input_tensor
        for operation in self.sequence:
            if operation == "a":
                input_tensor += previous_value
            elif operation == "z":
                input_tensor = previous + self.zero_add_gamma * input_tensor
            elif operation == "n" and self.norm_layer:
                input_tensor = self.norm_layer(input_tensor)
            elif operation == "d":
                input_tensor = tf.nn.dropout(input_tensor, rate=self.dropout_prob)
        return input_tensor
    
class EmbeddingPostProcessor(tf.keras.layers.Layer):
    def __init__(self, embedding_size, max_position_embeddings, token_type_vocab_size=2):
        super(EmbeddingPostProcessor, self).__init__()
        self.embedding_size = embedding_size
        self.max_position_embeddings = max_position_embeddings
        self.token_type_vocab_size = token_type_vocab_size
        
    def build(self, input_shape):
        width = self.embedding_size
        self.token_type_table = self.add_weight(
                    name="token_type_embedding",
                    shape=[self.token_type_vocab_size, width],
                    initializer=get_initializer(), dtype=tf.float32)
        self.full_position_embeddings = self.add_weight(
                    name="full_position_embeddings",
                    shape=[self.max_position_embeddings, width],
                    initializer=get_initializer(), dtype=tf.float32)
        self.embedding_post_process = LayerProcess(sequence="nd")
        self.build = True
    def call(self, inputs, **kwargs):
        input_shape = inputs.shape.as_list()
        num_dims = len(input_shape)
        batch_size, seq_len, width = input_shape[0:3]
        input_tensor = inputs
        token_type_ids = kwargs.pop("token_type_ids", None)
        output = input_tensor
        
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
        token_type_embeddings = tf.linalg.matmul(one_hot_ids, self.token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_len, width])
        output += token_type_embeddings
        
        position_embeddings = tf.slice(self.full_position_embeddings, [0, 0], [seq_len, -1])
        position_broadcast_shape = [1, seq_len, width]
        position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
        output += position_embeddings
        
        output = self.embedding_post_process([None, output])
        return output

class BasicAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(BasicAttentionLayer, self).__init__()
        self.config = config
        self.use_bias = config.use_bias
        self.num_attention_head = config.num_attention_head
        self.size_per_head = config.attention_head_size
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        
        self.query_layer = tf.keras.layers.Dense(self.num_attention_head * self.size_per_head, name="query",
                use_bias=self.use_bias, activation=None, kernel_initializer=get_initializer())
        self.key_layer = tf.keras.layers.Dense(self.num_attention_head * self.size_per_head, name="key",
                use_bias=self.use_bias, activation=None, kernel_initializer=get_initializer())
        self.value_layer = tf.keras.layers.Dense(self.num_attention_head * self.size_per_head, name="value",
                use_bias=self.use_bias, activation=None, kernel_initializer=get_initializer())
    
    def call(self, inputs, **kwargs):
        attention_mask = kwargs.pip("attention_mask", None) #[batch_size, f_seq_len, t_seq_len]
        from_tensor, to_tensor = inputs #[batch_size, seq_len, hidden_size]
        batch_size, seq_len, width = from_tensor.shape.as_list()
        
        from_tensor_2d = reshape_to_matrix(from_tensor) # [batch_size*seq_len, hidden_size]
        to_tensor_2d = reshape_to_matrix(to_tensor)
        
        
        query_output = self.query_layer(from_tensor_2d) # [batch_size*seq_len, hidden_size:num_att_head*size_head]
        key_output = self.key_layer(to_tensor_2d)
        value_output = self.value_layer(to_tensor_2d)
        
        # query_output: [B, num_att_head, seq_len, size_head]
        query_output = transpose_for_attention_dot(query_ouput,
                        batch_size, self.num_attention_head, seq_len, self.size_per_head)
        # key_output: [B, num_att_head, seq_len, size_head]
        key_output = transpose_for_attention_dot(key_ouput,
                        batch_size, self.num_attention_head, seq_len, self.size_per_head)
        # attention_probs: [B, num_att_head, f_seq_len, t_seq_len]
        attention_probs = dot_attention_prob(query_output, key_output,
                            size_per_head=self.size_per_head, attention_mask=attention_mask,
                            dropout_prob=self.attention_probs_dropout_prob,
                            do_clip_inf=self.do_clip_inf)
        # value_output: [B, num_att_head, seq_len, size_head]
        value_output = transpose_for_attention_dot(
                value_output, batch_size, self.num_attention_head, seq_len, self.size_per_head)
        # context_output: [B, num_att_head, seq_len, size_head]
        context_output = tf.linalg.matmul(attention_probs, value_output)
        # context_output: [B, seq_len, num_att_head, size_head]
        context_output = tf.transpose(context_output, [0, 2, 1, 3])
        context_output = tf.reshape(context_output, [batch_size, seq_len, 
                            self.num_attention_head * self.size_per_head])
        return context_output, attention_probs

def dot_attention_prob(query, key, size_per_head, attention_mask, dropout_prob=0.0, do_clip_inf=False):
    query = tf.math.multiply(query, 1.0/math.sqrt(float(size_per_head)))
    attention_scores = tf.linalg.matmul(query, key, transpose_b=True) #[B, num_att_head, f_seq_len, t_seq_len]
    attention_mask = tf.expand_dims(attention_mask, axis=[1]) #[batch_size, 1, query_len, key_len]
    adder = (1.0-tf.dtypes.cast(attention_mask, attention_score.dtype)) * -10000.0
    attention_score += adder
    
    if do_clip_inf:
        attention_scores = tf.clip_by_value(attention_scores, attention_scores.dtype.min, attention_scores.dtype.max)
    attention_probs = tf.nn.softmax(attention_scores)
    attention_probs = tf.nn.dropout(attention_probs, rate=dropout_prob)
    return attention_probs
    
    
def reshape_to_matrix(input_tensor):
    shape_list = input_tensor.shape.as_list()
    ndims = len(shape_list)
    if ndims <= 2: return input_tensor
    width = shape_list[-1]
    output_matrix = tf.reshape(input_tensor, [-1, width])
    return out_matrix

def reshape_from_matrix(input_matrix, orig_shape_list):
    if len(orig_shape_list) <= 2: return input_matrix
    return tf.reshape(input_matrix, orig_shape_list)

def transpose_for_attention_dot(input_tensor, batch_size, num_attention_heads, seq_len, size_per_head):
    output_tensor = tf.reshape(input_tensor, [batch_size, seq_len, num_attention_heads, size_per_head])
    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor
    
class TransformerFFN(tf.keras.layers.Layer):
    def __init__(self, config):
        super(TransformerFFN, self).__init__()
        self.config = config
        self.act_dropout_prob = config.act_dropout_prob
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.pre_layer_process = LayerProcess(sequence="none")
        self.intermdiate_output_dense = tf.keras.layers.Dense(self.intermediate_size, use_bias=True, 
                                    activation=tf.nn.gelu, kernel_initializer=get_initializer())
        self.layer_output_dense = tf.keras.layers.Dense(self.hidden_size, use_bias=True,
                                    kernel_initializer=get_initializer())
        self.post_layer_process = LayerProcess(sequence="dan")
        
    def call(self, inputs, **kwargs):
        batch_size, seq_len, width = inputs.shape.as_list()
        attention_output = inputs
        pre_input = self.pre_layer_process([None, attention_output])
        intermediate_output = self.intermediate_output_dense(pre_input)
        intermediate_output = tf.nn.dropout(intermediate_output, rate=self.act_dropout_prob)
        layer_output = self.layer_ouput_dense(intermediate_output)
        layer_output = self.post_layer_process([attention_output, layer_output])
        return layer_output
        
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_idx):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        
    def build(self, input_shape):  
        if not isinstance(input_shape, list):
            input_shape = input_shape.as_list()
        
        self.attention_input_process = LayerProcess(sequence="none")
        self.basic_attention_layer = BasicAttentionLayer(self.config)
        self.attention_output_dense = tf.keras.layers.Dense(self.hidden_size,
                    name="attention_output_dense", kernel_initializer=get_initializer())
        self.attention_output_process = LayerProcess(sequence="dan")
        self.transformer_ffn_layer = TransformerFFN(self.config)
        self.build = True
        
    def call(self, inputs, **kwargs):
        batch_size, seq_len, width = inputs.shape.as_list()
        layer_input = inputs
        attention_mask = kwargs.pop("attention_mask", None)
        attention_input = self.attention_input_process([None, layer_input])
        attention_output, attention_probs = self.basic_attention_layer(
                    [attention_input, attention_input], attention_mask=attention_mask)
        attention_output = self.attention_output_dense(attention_output)
        attention_output = self.attention_output_process([layer_input, attention_output])
        layer_output = self.transformer_ffn_layer(attention_output)
        return layer_output, attention_probs
        

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__() # init father class
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.hidden_size = config.hidden_size
        self.encoder_layers = {}
        
    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = input_shape.as_list()
        batch_size, seq_len, width = input_shape
        for layer_idx in range(self.num_hidden_layers):
            self.encoder_layers[layer_idx] = TransformerEncoderLayer(self.config, layer_idx)            
        self.build = True
        
    def call(self, inputs, **kwargs):
        input_tensor = inputs
        attention_mask = kwargs.pop("attention_maks", None)
        input_mask = kwargs.pop("input_mask", None)
        batch_size, seq_len, width = inputs.shape.as_list()
        
        all_layer_outputs = []
        all_attention_probs = []
        prev_output = input_tensor
        #prev_output = reshape_to_matrix(input_tensor)
        for layer_idx in range(self.num_hidden_layers):
            layer_output, attention_probs = self.encoder_layers[layer_idx](
                            prev_output, attention_mask=attention_mask)
            prev_output = layer_output
            #layer_output = reshape_from_matrix(layer_output, [batch_size, seq_len, width])
            all_layer_outputs.append(layer_output)
            all_attention_probs.append(attention_probs)
        return all_layer_outputs, all_attention_probs
        
    
class MyBertModel(tf.keras.layers.Layer):
    def __init__(self, config):
        super(MyBertModel, self).__init__() # init father class
        self.config = config
        
        self.pooled_output = None
        self.sequence_output = None
        self.all_encoder_layer_outputs = None
        self.all_attention_probs = None
        self.embedding_output = None
        self.embedding_table = None
        
    def build(self, input_shape):
        # when exec call(), exec build() first automaticaly
        # set self.build=True, to make sure exec build() only once
        # input_shape is the shape of inputs param of call()
        logging.info(f"exec build {input_shape}")
        
        self.embedding_lookup = EmbeddingLookup(config.vocab_size, config.hidden_size)
        self.embedding_postprocessor = EmbeddingPostProcessor(config.hidden_size, config.max_position_embeddings)
        self.bert_transformer_encoder = TransformerEncoder(self.config)
        self.pool_output_layer = tf.keras.layers.Dense(config.hidden_size, activation=tf.tanh,
                                kernel_initializer=get_initializer(), name="pooler/dense")
        self.build = True
        
    def call(self, inputs, **kwargs):
        batch_size, seq_len = inputs.shape.as_list()
        input_ids = inputs
        input_mask = kwargs.pop("input_mask", None)
        token_type_ids = kwargs.pop("token_type_ids", None)
        
        self.embedding_output, self.embedding_table = self.embedding_lookup(input_ids)
        self.embedding_output = self.embedding_postprocessor(self.embedding_output,
                                        token_type_ids=token_type_ids, **kwargs)
        
        transformer_input_tensor = self.embedding_output
        attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)
        self.all_encoder_layer_outputs, self.all_attention_probs = self.bert_transformer_encoder(
            transformer_input_tensor,
            attention_mask=attention_mask,
            input_mask=input_mask
        )
        
        self.sequence_output = tf.cast(self.all_encoder_layer_output[-1], tf.float32)
        first_token_tensor = tf.squeeze(self.sequence_output[:,0:1,:], axis=1)
        self.pooled_output = self.pooled_output_layer(first_token_tensor, **kwargs)
        
        return self.pooled_output
    
    def get_sequence_output(self):
        return self.sequence_output
    def get_embedding_table(self):
        return self.embedding_table
    def get_pooled_output(self):
        return pooled_output
  
def create_attention_mask_from_input_mask(from_tensor, to_mask, mask_type='bidi'):
    # create 3d attention mask from 2d, [batch_size, seq_len] to [b, f_seq_len, t_seq_len]
    # mask_type: one of (bidi, l2r, r2l)
    from_shape = from_tensor.shape.as_list()
    batch_size, from_seq_len = from_shape[0:2]
    to_shape = to_mask.shape.as_list()
    to_seq_len = to_shape[1]
    
    to_mask = tf.dtypes.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_len]), tf.float32)
    if mask_type == "bidi":
        brodcast_ones = tf.ones(shape=[batch_size, from_seq_len, 1], dtype=tf.float32) #TODO: not understand
    elif mask_type == "l2r":
        i = tf.range(from_seq_len)[:, None]
        j = tf.range(to_seq_len)
        brodcast_ones = tf.cast(i>=j-to_seq_len+from_seq_len, dtype=tf.float32)
    elif mask_type == "r2l":
        i = tf.range(from_seq_len)[:, None]
        j = tf.range(to_seq_len)
        brodcast_ones = tf.cast(i<=j-to_seq_len+from_seq_len, dtype=tf.float32)
    mask = broadcast_ones * to_mask
    return mask
        
    

def get_masked_lm_output(sequence_output, embedding_table, masked_lm_positions,
                        masked_lm_ids, masked_lm_weights):
    res = dict()
    res["loss"] = None
    res["logits"] = None
    return res

def get_next_sentence_output(get_pooled_output, next_sentence_labels):
    res = dict()
    res["loss"] = None
    res["logits"] = None
    return res

def build_bert_model():
    model_config = MyBertConfig()
    
    def model_fn(features, labels, mode):
        # params of model_fn:
        # features: first item of input_fn
        # labels: second item of input_fn
        # mode: optional, tf.estimator.ModeKeys
        # config: estimator.RunConfig object, pass to estimator
        
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]
        
        model = MyBertModel(model_config)
        model(features["input_ids"], input_mask=features["input_mask"], 
              token_type_ids=features["segment_ids"])
        
        # mlm loss
        masked_lm_output = get_masked_lm_output(model.get_sequence_output(),
                                                model.get_embedding_table(),
                                                masked_lm_positions,
                                                masked_lm_ids,
                                                masked_lm_weights)
        masked_lm_loss = masked_lm_output["loss"]
        
        #scalar_metrics = {}
        # eval mlm accuracy
        #mlm_acc = eval_mlm_accuracy(masked_lm_output["logits"], masked_lm_positions, masked_lm_ids, masked_lm_weights)
        #scalar_metrics["mlm_acc"] = mlm_acc
        
        # next sentence prediction loss
        next_sentence_loss = 0
        if model_config.use_nsp:
            next_sentence_output = get_next_sentence_output(model.get_pooled_output(),
                                                           next_sentence_labels)
            next_sentence_loss = next_sentence_output["loss"]
        
            # eval nsp accuracy
            #nsp_acc = eval_nsp_accuracy(next_sentence_output["logits"], next_sentence_labels)
            #scalar_metrics["nsp_acc"] = nsp_acc
                
        # calculate gradieent and update train variable
        total_loss = masked_lm_loss + next_sentence_loss
        tvars = tf.compat.v1.trainable_variables()
        for tvar in tvars: logging.info(f"trainable var: {tvar.name}, {tvar}")
        grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), clip_norm=model_config.clip_norm)
        optimizer = tf.train.AdamOptimizer(model_config.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, train_vars))
        
        # summary scalar
        tf.compat.v1.summary.scalar('losses/masked_lm_loss', masked_lm_loss)
        tf.compat.v1.summary.scalar('losses/next_sentence_loss', next_sentence_loss)
        tf.compat.v1.summary.scalar('losses/total_loss', total_loss)
        
        output_spec = tf.estimator.EstimatorSpec(
                                mode=mode, loss=total_loss, train_op=train_op)
                                
        return output_spec
    return model_fn

def main():
    model_config = BertConfig()
    
    train_data_iterator = DataIterator(FLAGS.input_path, FLAGS.batch_size, FLAGS.seq_len, FLAGS.masked_len)
    model_fn = build_bert_model()

    
    # GPU config
    run_config = tf.compat.v1.ConfigProto()    
    run_config.allow_soft_placement = True

    estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_path,
    config=tf.estimator.RunConfig(
      save_checkpoints_steps=500,
      save_checkpoints_secs=None,
      keep_checkpoint_every_n_hours=2,
      log_step_count_steps=100,
      session_config=run_config))
    
    tf.estimator.train(input_fn=train_data_iterator.input_fn, max_steps=FLAGS.max_train_steps)
    
    #tf.Session().run(loss_op, train_op)
    #train_one_epoch:
    #batch_instances, batch_labels = sess.run(next_element)
    #feed = {X:batch_instances, Y:batch_labels}
    #results = sess.run(task_ops, feed_dict=feed)

