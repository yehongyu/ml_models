#coding=utf-8
import io
import time
import re
import json
import argparse
import logging
import random
logging.basicConfig(level=logging.INFO)
import collections
import tensorflow as tf

from multiprocessing import Process
import numpy as np
from nlp_tokenization_chinese import CommentTokenizerChinese

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

def cut_sentence(para):
    para = convert_to_unicode(para)
    para = re.sub("[\n\r\t]", " ", para)
    para = re.sub("([。！？\?])([^”’])", r"\1\n\2", para)
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)
    para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)
    para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    para = para.rstrip()
    return para.split("\n")

class MaskedProcesser():
    def __init__(self, vocab_file, max_seq_len, max_predictions_per_seq, masked_lm_prob, out_as_tfrecord=True,
                 short_seq_prob=0.1, do_while_word_mask=False):
        # tokenier
        self.tokenier = CommentTokenizerChinese(vocab_file)
        self.cls_id = self.tokenier.cls_id
        self.mask_id = self.tokenier.mask_id
        self.sep_id = self.tokenier.sep_id

        # parameter
        self.max_seq_len = max_seq_len
        self.max_predictions_per_seq = max_predictions_per_seq
        self.masked_lm_prob = masked_lm_prob
        self.short_seq_prob = short_seq_prob
        self.do_while_word_mask = do_while_word_mask
        self.out_as_tfrecord = out_as_tfrecord

        # init
        self.rng = random.Random(1234)

    def parse_document(self, line):
        all_documents = []
        line = line.strip("\n\r")
        para = (json.loads(line)).get("content", "")
        if not para: return
        sentences = cut_sentence(para)
        sent_tokens = [self.tokenier.encode_pieces(sent) for sent in sentences]
        all_documents.append(sent_tokens)
        return all_documents

    def create_instances(self, all_documents):
        for doc_idx in range(len(all_documents)):
            yield self.create_instances_from_document(doc_idx, all_documents)

    def get_random_doc_idx(self, cur_idx, all_documents):
        for _ in range(10):
            random_doc_idx = self.rng.randint(0, len(all_documents)-1)
            if random_doc_idx != cur_idx: return random_doc_idx

    def get_random_text_b(self, target_b_length, doc_idx, all_documents):
        tokens_b = []
        random_document_index = self.get_random_doc_idx(doc_idx, all_documents)
        random_document = all_documents[random_document_index]
        random_start = self.rng.randint(0, len(random_document)-1)
        for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length: break
        return tokens_b

    def truncate_seq_pair(self, token_a, token_b, max_num_tokens):
        while True:
            total_length = len(token_a) + len(token_b)
            if total_length <= max_num_tokens: break
            trunc_tokens = token_a if len(token_a) > len(token_b) else token_b
            if self.rng.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pip()

    def create_masked_lm_predictions(self, tokens):
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token in [self.cls_id, self.sep_id]: continue
            if self.do_while_word_mask and len(cand_indexes) >= 1 and token.startswiith("##"):
                cand_indexes[-1].append(i)
            else: cand_indexes.append([i])
        self.rng.shuffle(cand_indexes)

        num_to_predict = min(self.max_predictions_per_seq, max(1, int(round(len(token) * self.masked_lm_prob))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict: break
            if len(masked_lms) + len(index_set) > num_to_predict: continue
            for index in index_set:
                covered_indexes.add(index)
                if self.rng.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    if self.rng.random() < 0.5:
                        masked_token = tokens[index]
                    else:
                        masked_token = self.tokenier.random_token()
                masked_lms.append([index, tokens[index]])
                tokens[index] = masked_token
        masked_lms = sorted(masked_lms, key=lambda x:x[0])
        masked_lm_positions = [item[0] for item in masked_lms]
        masked_lm_tokens = [item[1] for item in masked_lms]
        return tokens, masked_lm_positions, masked_lm_tokens

    def generate_instance(self, tokens_a, tokens_b, is_random_next):
        tokens = [self.cls_id] + tokens_a + [self.sep_id] + tokens_b + [self.sep_id]
        input_ids = self.tokenier.full_tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_mask = [1] * len(input_ids)
        if len(input_ids) < self.max_seq_len:
            paddings = [0] * (self.max_seq_len - len(input_ids))
            input_ids.extend(paddings)
            segment_ids.extend(paddings)
            input_mask.extend(paddings)

        tokens, masked_lm_positions, masked_lm_tokens = self.create_masked_lm_predictions(tokens)
        masked_lm_ids = self.tokenier.full_tokenizer.convert_tokens_to_ids(masked_lm_tokens)

        instance = dict()
        instance["tokens"] = tokens # debug
        instance["input_ids"] = input_ids
        instance["input_mask"] = input_mask
        instance["segment_ids"] = segment_ids
        instance["next_sentence_label"] = 1 if is_random_next else 0
        instance["masked_lm_positions"] = masked_lm_positions
        instance["masked_lm_tokens"] = masked_lm_tokens # debug
        instance["masked_lm_ids"] = masked_lm_ids
        return instance

    def create_instances_from_document(self, doc_idx, all_documents):
        instances = []
        document = all_documents[doc_idx]
        max_num_tokens = self.max_seq_len - 3
        target_seq_length = max_num_tokens
        if self.rng.random() < self.short_seq_prob:
            target_seq_length = self.rng.randint(2, max_num_tokens)
        chunk = []; i = 0; chunk_act_len = 0
        while i < len(document):
            sentence = document[i]
            chunk.append(sentence)
            chunk_act_len += len(sentence)
            if i == len(document)-1 or chunk_act_len >= target_seq_length:
                if chunk:
                    a_end = 1
                    if len(chunk) >= 2: a_end = self.rng.randint(1, len(chunk)-1)
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(chunk[j])

                    is_random_next = False
                    if len(chunk) == 1 or self.rng.random() < 0.5:
                        is_random_next = True
                        tokens_b = self.get_random_text_b(target_seq_length - len(tokens_a), doc_idx, all_documents)
                        num_unused_sentences = len(chunk) - a_end
                        i -= num_unused_sentences
                    else:
                        for j in range(a_end, len(chunk)):
                            tokens_b.extend(chunk[j])
                    self.truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                    if tokens_a and tokens_b:
                        instances.append(self.generate_instance(tokens_a, tokens_b, is_random_next))
                chunk = []; chunk_act_len = 0
            i += 1
        return instances


    def write_to_file(self, fout, instances):
        for instance in instances:
            if not self.out_as_tfrecord:
                fout.write(json.dumps(instance) + "\n")
            else:
                features = collections.OrderedDict()
                features["input_ids"] = create_int_feature(instance["input_ids"])
                features["input_mask"] = create_int_feature(instance["input_mask"])
                features["segment_ids"] = create_int_feature(instance["segment_ids"])
                features["masked_lm_positions"] = create_int_feature(instance["masked_lm_positions"])
                features["masked_lm_ids"] = create_int_feature(instance["masked_lm_ids"])
                features["next_sentence_labels"] = create_int_feature(instance["is_random_next"])
                tf_example = tf.train.Example(features=tf.train.Features(features=features))
                fout.write(tf_example.SerializaToString())

    def process_task(self, infiles, outfile):
        if self.out_as_tfrecord:
            fout = tf.python_io.TFRecordWriter(outfile)
        else:
            fout = open(outfile, "w")
        try:
            for infile in infiles:
                with open(infile, "r") as fin:
                    for line in fin:
                        try:
                            all_documents = self.parse_document(line)
                            for instances in self.create_instances(all_documents):
                                self.write_to_file(fout, instances)
                        except Exception as e:
                            import traceback
                            err_msg = traceback.format_exc()
                            logging.error("err:{}, line:{}".format(err_msg, line))
        finally:
            fout.close()


def main(args):
    processor = MaskedProcesser(vocab_file=args.vocab_file,
                                max_seq_len=args.max_seq_len,
                                max_predictions_per_seq=args.max_predictions_per_seq,
                                masked_lm_prob=args.masked_lm_prob,
                                out_as_tfrecord=args.out_as_tfrecord,
                                short_seq_prob=args.short_seq_prob,
                                do_while_word_mask=args.do_while_word_mask
                                )

    all_files = ["{}/{}".format(args.corpus_path, f) for f in io.listdir(args.corpus_path)]

    tasks = []
    for i in args.num_workers:
        infiles = [all_files[f_i] for f_i in range(i, len(all_files), args.num_workers)]
        outfile = "{}/{}".format(args.out_path, i)
        logging.info("Start process files:{}, outfile:{}".format(infiles, outfile))
        task = Process(tagert=processor.process_task, args=(infiles, outfile))
        tasks.append(task)
        task.start()

    for task in tasks: task.join()

    logging.info("Finished task.")
    return



if __name__ == "__main__":
    now_time = str(int(time.time()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, default="/Users/aodandan/data/tfrecord/bert_data/toutiao_article_1104/", required=True)
    parser.add_argument('--output_path', type=str, default="/Users/aodandan/data/tfrecord/bert_data/models/" + now_time, required=True)
    parser.add_argument('--vocab_file', type=str, default="/Users/aodandan/data/tfrecord/bert_data/vocab/vocab.txt", required=True)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--masked_lm_prob', type=float, default=0.15)
    parser.add_argument('--max_predictions_per_seq', type=int, default=20)
    parser.add_argument('--short_seq_prob', type=float, default=0.1, help="Probability of creating sequence")
    parser.add_argument('--num_workers', type=int, default=1, help="num of processor")
    parser.add_argument('--dupe_factor', type=int, default=1, help="num of times to duplicate the corpus")

    args = parser.parse_args()
    logging.info(f"args: {args}")
    main(args)