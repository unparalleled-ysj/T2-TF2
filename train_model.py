import argparse
import os
import sys
import time
import audio
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Tacotron2.model_loader import parse_Tacotron2_args, get_Tacotron2_model
from Tacotron2.optimizer import Tacotron2Loss, CustomSchedule
from plot import plot_alignment
from train_config import Logger, parse_args, get_train_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUS
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

parser = argparse.ArgumentParser(description='Tensorflow2 Tacotron2 Training')
parser = parse_args(parser)
# args, _ = parser.parse_known_args()
parser = parse_Tacotron2_args(parser)
args = parser.parse_args()

os.makedirs(args.output_directory, exist_ok=True)
sys.stdout = Logger(os.path.join(args.output_directory, 'train.log'), sys.stdout)

train_dataset = get_train_dataset(args)
tacotron2_model = get_Tacotron2_model(args, training=True)
learning_rate = CustomSchedule(args.d_model)
optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# optimizer = keras.optimizers.Adam(1e-3)
criterion = Tacotron2Loss()
train_loss = keras.metrics.Mean(name='train_loss')
checkpoint = tf.train.Checkpoint(Tacotron2=tacotron2_model, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.output_directory, max_to_keep=5)
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print("restore lastest checkpoint from {}".format(checkpoint_manager.latest_checkpoint))
else:
    print("train new tacotorn2 model")


train_step_signature = [
    tf.TensorSpec(shape=(4, None), dtype=tf.int32),
    tf.TensorSpec(shape=(4, None, 80), dtype=tf.float32),
    tf.TensorSpec(shape=(4,), dtype=tf.int32),
    tf.TensorSpec(shape=(4, None), dtype=tf.float32),
    tf.TensorSpec(shape=(4,), dtype=tf.int32)
]


@tf.function(input_signature=train_step_signature)
def train_step(seq, mel_tar, mel_len, gate_tar, s_id):
    inp = [seq, mel_tar, s_id]
    tar = [mel_tar, mel_len, gate_tar]
    with tf.GradientTape() as tape:
        outputs = tacotron2_model(inp, training=True)
        loss = criterion(outputs, tar)

    gradients = tape.gradient(loss, tacotron2_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, tacotron2_model.trainable_variables))

    train_loss(loss)
    return outputs


print("start to train")
for epoch in range(1, args.epochs + 1):
    train_loss.reset_states()
    for batch, [seqs, mel_targets, mel_lengths, gate_targets, speaker_id] in enumerate(train_dataset):
        start_time = time.time()
        model_outputs = train_step(seqs, mel_targets, mel_lengths, gate_targets, speaker_id)
        print("Epoch {} Batch {} Loss {:.4f} ".format(epoch, batch, train_loss.result()))
        print("Time taken for 1 batch: {} secs\n".format(time.time() - start_time))
        if epoch % args.epochs_per_alignment == 0:
            alignments = model_outputs[-1].numpy()
            mel_outputs = model_outputs[1].numpy()
            index = np.random.randint(len(alignments))
            plot_alignment(alignments[index].transpose(0, 1),
                           os.path.join(args.output_directory, f"align_{epoch:04d}.png"),
                           info=f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} Epoch={epoch:04d} loss={train_loss.result():.4f}")
            wav = audio.inv_mel_spectrogram(mel_outputs[index].transpose())
            audio.save_wav(wav, os.path.join(args.output_directory, f"train_{epoch:04d}.wav"))
        if epoch % args.epochs_per_checkpoint == 0:
            checkpoint_save_path = checkpoint_manager.save(checkpoint_number=epoch)
            print("Saving checkpoint for epoch {} at {}".format(epoch, checkpoint_save_path))

