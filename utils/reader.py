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


def read_examples(input_file, tokenizer, masked_lm_positions):
    """Read a list of `InputExample`s from an input file.    
    Ones in segemnt_ids stands for words in second sentence. Does not matter for us. Could be zeros."""
    examples = []
    unique_id = 0
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            text = line.strip()
            words = text.split()
            masked_lm_labels = []
            masked_words = []
            for m_pos in masked_lm_positions:                
                masked_words.append(words[m_pos])
                words[m_pos] = 'MASK'
            text = " ".join(words)
            tokens = tokenizer.tokenize(text)
            for m_pos in masked_lm_positions:                
                masked_lm_labels.append(tokens[m_pos])
                tokens[m_pos] = '[MASK]'
            examples.append(
                InputExample(
                    tokens=tokens,
                    segment_ids=[1] * len(tokens),
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                    is_random_next=False))
            unique_id += 1
    return examples, masked_words
