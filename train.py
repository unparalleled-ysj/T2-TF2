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
from Tacotron2.data_loader import LoadData
from plot import plot_alignment



def parse_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_directory', type=str, default='logs', help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='training_data', help='Path to dataset')
    parser.add_argument('--latest-checkpoint-file', type=str, default='checkpoint_latest.pt', help='Store the latest checkpoint in each epoch')
    parser.add_argument('--tacotron2-checkpoint', type=str, default=None, help='Path to pre-trained Tacotron2 checkpoint for sample generation')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=200, help='Number of total epochs to run')
    training.add_argument('--epochs-per-alignment', type=int, default=1, help='Number of epochs per alignment')
    training.add_argument('--epochs-per-checkpoint', type=int, default=10, help='Number of epochs per checkpoint')
    training.add_argument('--seed', type=int, default=1234, help='Seed for PyTorch random number generators')
    training.add_argument('--dynamic-loss-scaling', type=bool, default=True, help='Enable dynamic loss scaling')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument('--use-saved-learning-rate', default=False, type=bool)
    optimization.add_argument('--d_model', default=512, type=int, help='for setup learing rate')
    optimization.add_argument('-bs', '--batch-size', default=8, type=int, help='Batch size per GPU')

    # dataset parameters
    dataset = parser.add_argument_group('dataset parameters')
    dataset.add_argument('--training-anchor-dirs', default=['biaobei'], type=str, nargs='*', help='Path to training filelist')
    dataset.add_argument('--validation-anchor-dirs', default=['no'], type=str, nargs='*', help='Path to validation filelist')
    dataset.add_argument('--text-cleaners', nargs='*', default=['basic_cleaners'], type=str, help='Type of text cleaners for input text')

    # audio parameters
    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--max-wav-value', default=32768.0, type=float, help='Maximum audiowave value')
    audio.add_argument('--sampling-rate', default=16000, type=int, help='Sampling rate')
    audio.add_argument('--filter-length', default=1024, type=int, help='Filter length')
    audio.add_argument('--hop-length', default=200, type=int, help='Hop (stride) length')
    audio.add_argument('--win-length', default=800, type=int, help='Window length')
    audio.add_argument('--mel-fmin', default=50.0, type=float, help='Minimum mel frequency')
    audio.add_argument('--mel-fmax', default=7600.0, type=float, help='Maximum mel frequency')

    return parser


#@tf.function
def train_step(model, criterion, optimizer, train_loss, inputs, targets):

    with tf.GradientTape() as tape:
        model_outputs = model(inputs, training=True)
        loss = criterion(model_outputs, targets)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    return model_outputs


def ragged_tensor_to_tensor(seqs, mels_path, speaker_ids):
    return seqs.to_tensor(), mels_path, speaker_ids


def prepocess_data(seqs, mels_path, speaker_ids):
    mel_targets = []
    mel_lengths = []
    gate_targets = []
    for mel_path in mels_path:
        mel_target = np.load(mel_path.numpy()).transpose()
        mel_length = mel_target.shape[0]
        gate_target = np.zeros(mel_length, dtype=np.float32)
        gate_target[-1] = 1.

        mel_targets.append(mel_target)
        mel_lengths.append(mel_length)
        gate_targets.append(gate_target)

    mel_targets = keras.preprocessing.sequence.pad_sequences(sequences=mel_targets,
                                                             dtype='float32',
                                                             padding='post',
                                                             value=-5.)
    gate_targets = keras.preprocessing.sequence.pad_sequences(sequences=gate_targets,
                                                              dtype='float32',
                                                              padding='post',
                                                              value=1.)
    return seqs, mel_targets, mel_lengths, gate_targets, speaker_ids


def tf_prepocess_data(seqs, mels_path, speaker_ids):
    return tf.py_function(func=prepocess_data,
                          inp=[seqs, mels_path, speaker_ids],
                          Tout=[tf.int32, tf.float32, tf.int32, tf.float32, tf.int32])


def get_train_dataset(args):
    seqs, mels_path, speaker_ids = LoadData(args).get_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((tf.ragged.constant(seqs),
                                                        tf.constant(mels_path),
                                                        tf.constant(speaker_ids)))
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size=args.batch_size, drop_remainder=True)
    train_dataset = train_dataset.map(map_func=ragged_tensor_to_tensor,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(map_func=tf_prepocess_data,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
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
    args, _ = parser.parse_known_args()

    parser = parse_Tacotron2_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.output_directory, 'train.log'), sys.stdout)


    print("prepare training dataset")
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

    print("start to train")
    for epoch in range(1, args.epochs + 1):
        train_loss.reset_states()
        for batch, [seqs, mel_targets, mel_lengths, gate_targets, speaker_id] in enumerate(train_dataset):
            start_time = time.time()
            inputs = [seqs, mel_targets, speaker_id]
            targets = [mel_targets, mel_lengths, gate_targets]
            model_outputs = train_step(model=tacotron2_model, criterion=criterion, optimizer=optimizer,
                                   train_loss=train_loss, inputs=inputs, targets=targets)
            print("Epoch {} Batch {} Loss {:.4f} ".format(epoch, batch, train_loss.result()))
            print("Time taken for 1 epoch: {} secs\n".format(time.time() - start_time))
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


if __name__ == '__main__':
    main()
