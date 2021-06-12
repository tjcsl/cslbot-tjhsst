#!/usr/local/virtualenvs/cslbot/bin/python
# Based on https://www.tensorflow.org/tutorials/text/text_generation
# Input built via:
# psql ircbot -c 'select msg from log' | head -n -2 | tail -n +3 > irc.txt

import argparse
import os

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

# Read in the corpus.
with open('irc.txt', 'rb') as f:
    # Strip non-ascii chars.
    text = f.read().decode('ascii', 'ignore')

# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')

# The unique characters in the file
vocab = sorted(set(text))

ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)

chars_from_ids = preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# Directory where the checkpoints will be saved
checkpoint_dir = './checkpoint'


def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


class MyModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    @tf.function
    def train_step(self, inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(labels, predictions)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {'loss': loss}

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


def build_model():
    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    model = MyModel(
        # Be sure the vocabulary size matches the `StringLookup` layers.
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    return model


def build_dataset():
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    seq_length = 100

    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    # Batch size
    BATCH_SIZE = 64

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    return dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


def train():
    model = build_model()

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

    # To keep training time reasonable, use 10 epochs to train the model.
    EPOCHS = 10

    dataset = build_dataset()

    # Train the model.
    model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


class OneStep(tf.keras.Model):

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


# Evaluation step (generating text using the learned model)
def generate_text(model, start_string):
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    states = None
    next_char = tf.constant([start_string])
    result = [next_char]

    # Number of characters to generate
    num_generate = 1000

    for n in range(num_generate):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    return result[0].numpy().decode('utf-8')


def sample(seed):
    # Load the trained model.
    model = build_model()

    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if not ckpt:
        raise Exception('No checkpoint found in %s' % checkpoint_dir)
    model.load_weights(ckpt).assert_consumed()

    start = ' '.join(seed)
    if not start:
        raise Exception('No seed specified.')

    print(generate_text(model, start_string=start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', help='The action to take.', choices=['train', 'sample'])
    parser.add_argument('seed', help='The seed to use.', nargs='*')
    args = parser.parse_args()

    if args.action == 'train':
        train()
    else:
        sample(args.seed)


if __name__ == '__main__':
    main()
