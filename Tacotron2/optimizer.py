import tensorflow as tf
from tensorflow import keras


class Tacotron2Loss(keras.layers.Layer):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def call(self, model_outputs, targets):
        mel_target, mel_length, gate_target = targets
        mel_output_before, mel_output_after, gate_output, _ = model_outputs

        mel_loss_before = tf.losses.MSE(mel_target, mel_output_before)
        mel_loss_after = tf.losses.MSE(mel_target, mel_output_after)

        mask = tf.cast(tf.sequence_mask(mel_length), dtype=mel_loss_after.dtype)
        mel_loss = tf.reduce_mean(mel_loss_before * mask) + tf.reduce_mean(mel_loss_after * mask)
        gate_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(gate_target, gate_output) * mask)

        return mel_loss + gate_loss


# learning_rate = d_model**(-0.5) * min(stem_num**(-0.5), step_num * warmup_steps**(-1.5))
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**(-1.5))

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
