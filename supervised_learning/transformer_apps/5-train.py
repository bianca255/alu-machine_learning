#!/usr/bin/env python3
"""Train a Transformer for Portuguese-to-English translation."""

import tensorflow.compat.v2 as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Implement the Transformer learning-rate warmup schedule."""

    def __init__(self, dm, warmup_steps=4000):
        """Store model depth and warmup duration."""
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        """Return the learning rate for optimization step ``step``."""
        step = tf.cast(step, tf.float32)
        inverse_sqrt = tf.math.rsqrt(step)
        warmup = step * tf.math.pow(self.warmup_steps, -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(inverse_sqrt, warmup)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Train and return a Transformer on the TED translation dataset."""
    data = Dataset(batch_size, max_len)
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2
    transformer = Transformer(
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_len,
        max_len,
    )
    optimizer = tf.keras.optimizers.Adam(
        CustomSchedule(dm),
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none",
    )

    def loss_function(real, prediction):
        """Calculate masked sparse categorical cross-entropy."""
        loss = loss_object(real, prediction)
        mask = tf.cast(tf.not_equal(real, 0), loss.dtype)
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

    def accuracy_function(real, prediction):
        """Calculate token accuracy while ignoring padding tokens."""
        predictions = tf.argmax(prediction, axis=2, output_type=real.dtype)
        matches = tf.cast(tf.equal(real, predictions), tf.float32)
        mask = tf.cast(tf.not_equal(real, 0), tf.float32)
        return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)

    for epoch in range(epochs):
        total_loss = tf.keras.metrics.Mean()
        total_accuracy = tf.keras.metrics.Mean()
        for batch, (inputs, target) in enumerate(data.data_train):
            target_input = target[:, :-1]
            target_real = target[:, 1:]
            masks = create_masks(inputs, target_input)
            with tf.GradientTape() as tape:
                predictions = transformer(
                    inputs,
                    target_input,
                    True,
                    masks[0],
                    masks[1],
                    masks[2],
                )
                batch_loss = loss_function(target_real, predictions)
            gradients = tape.gradient(
                batch_loss,
                transformer.trainable_variables,
            )
            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables)
            )
            batch_accuracy = accuracy_function(target_real, predictions)
            total_loss.update_state(batch_loss)
            total_accuracy.update_state(batch_accuracy)
            if batch % 50 == 0:
                print(
                    "Epoch {}, batch {}: loss {} accuracy {}".format(
                        epoch + 1,
                        batch,
                        batch_loss,
                        batch_accuracy,
                    )
                )
        print(
            "Epoch {}: loss {} accuracy {}".format(
                epoch + 1,
                total_loss.result(),
                total_accuracy.result(),
            )
        )
    return transformer
