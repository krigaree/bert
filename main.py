import sys
sys.path.append("../")

import bert
import bert.modeling as tfm
import numpy as np
import tensorflow as tf
import utils.imports as imp
imp.del_all_flags(tf.flags.FLAGS)
import bert.tokenization as tft
imp.del_all_flags(tf.flags.FLAGS)

from model.net_architecture import pretraining_convert_examples_to_features
from model.net_architecture import model_fn_builder
from utils.reader import read_examples
from utils.preprocessing import input_fn_builder

path_to_colab = "/home/krigaree/Documents/Uczelnia/Dutkiewicz/"
path_to_bert = path_to_colab + "BERT/"

model_dir = path_to_bert + "uncased_L-24_H-1024_A-16/"

vocab_file = model_dir + "vocab.txt"
bert_config_file = model_dir + "bert_config.json"
init_checkpoint = model_dir + "bert_model.ckpt"

input_path = path_to_bert + "inputs/"
output_path = path_to_bert + "outputs/"
input_file = input_path + "input_3.txt"
max_seq_length = 128

# We are masking only one word in the sentence at chosen position
max_predictions_per_seq = 1
masked_lm_positions = [3]

bert_config = tfm.BertConfig.from_json_file(bert_config_file)
tokenizer = tft.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)
examples, masked_words = read_examples(input_file, tokenizer, masked_lm_positions=masked_lm_positions)

features = pretraining_convert_examples_to_features(
    instances=examples, max_seq_length=max_seq_length, 
    max_predictions_per_seq=max_predictions_per_seq, tokenizer=tokenizer)

is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
    master=None,
    tpu_config=tf.contrib.tpu.TPUConfig(
        num_shards=1,
        per_host_input_for_training=is_per_host))

model_fn = model_fn_builder(
    bert_config=bert_config,
    init_checkpoint=init_checkpoint,
    learning_rate=0,
    num_train_steps=1,
    num_warmup_steps=1,
    use_tpu=False,
    use_one_hot_embeddings=False)

# If TPU is not available, this will fall back to normal Estimator on CPU
# or GPU.
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    predict_batch_size=1)

input_fn = input_fn_builder(
    features=features, seq_length=max_seq_length, max_predictions_per_seq=max_predictions_per_seq,
tokenizer=tokenizer)

tensorflow_all_out = []
for result in estimator.predict(input_fn, yield_single_examples=True):
    tensorflow_all_out.append(result)

print("Number of sentences: {}".format(len(tensorflow_all_out)))
predicted_tokens = []
for i in range(len(tensorflow_all_out)):
    print("\nSentence_{}:".format(i+1))
    print(" ".join(examples[i].tokens))
    print("masked_word: {}".format(masked_words[i]))
    predicted_tokens.append(tokenizer.convert_ids_to_tokens(tensorflow_all_out[i]['masked_lm_predictions']))
    probs = np.exp(tensorflow_all_out[i]['masked_lm_probs']) 
    print("predicted tokens and probs: {}".format(tuple(zip(predicted_tokens[i],probs))))