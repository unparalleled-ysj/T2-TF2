import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import sys
from Tacotron2.data_loader import LoadData


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def parse_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_directory', type=str, default='logs-tf-function', help='Directory to save checkpoints')
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
    optimization.add_argument('-bs', '--batch-size', default=4, type=int, help='Batch size per GPU')

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
