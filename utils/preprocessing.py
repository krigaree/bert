import tensorflow as tf

class InputExample:
    """A single instance example."""

    def __init__(self, tokens, segment_ids, masked_lm_positions,
                 masked_lm_labels, is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.is_random_next = is_random_next
    def __repr__(self):
        return '\n'.join(k + ":" + str(v) for k, v in self.__dict__.items())

class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, masked_lm_positions,
                 masked_lm_ids, masked_lm_weights, next_sentence_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids
        self.masked_lm_weights = masked_lm_weights
        self.next_sentence_labels = next_sentence_label

    def __repr__(self):
        return '\n'.join(k + ":" + str(v) for k, v in self.__dict__.items())

def input_fn_builder(features, seq_length, max_predictions_per_seq, tokenizer): # pylint: disable=unused-argument
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_masked_lm_positions = []
    all_masked_lm_ids = []
    all_masked_lm_weights = []
    all_next_sentence_labels = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_masked_lm_positions.append(feature.masked_lm_positions)
        all_masked_lm_ids.append(feature.masked_lm_ids)
        all_masked_lm_weights.append(feature.masked_lm_weights)
        all_next_sentence_labels.append(feature.next_sentence_labels)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        data = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "masked_lm_positions":
                tf.constant(
                    all_masked_lm_positions,
                    shape=[num_examples, max_predictions_per_seq],
                    dtype=tf.int32),
            "masked_lm_ids":
                tf.constant(
                    all_masked_lm_ids,
                    shape=[num_examples, max_predictions_per_seq],
                    dtype=tf.int32),
            "masked_lm_weights":
                tf.constant(
                    all_masked_lm_weights,
                    shape=[num_examples, max_predictions_per_seq],
                    dtype=tf.float32),
            "next_sentence_labels":
                tf.constant(
                    all_next_sentence_labels,
                    shape=[num_examples, 1],
                    dtype=tf.int32),
        })

        data = data.batch(batch_size=batch_size, drop_remainder=False)
        return data

    return input_fn