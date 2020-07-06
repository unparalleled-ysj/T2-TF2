# import tensorflow as tf
# import audio
# import numpy as np
#
# # dataset = tf.data.Dataset.from_tensor_slices(([[1, 2], [2, 1], [0, -1], [1, 0]],
# #                                               [[3, 4, 5], [4, 5, 6], [1, 2, 3], [3, 2, 1]],
# #                                               [[5], [6], [7], [8]]))
#
# seq = [[1, 3, 2, 4], [1], [1, 2], [7, 5, 4], [1, 2, 3, 4, 5]]
# mel = [[[1.1, 2.9, 0.8], [1.0, -1.2, -0.9], [0.8, 1.3, 0.9]],
#                           [[6.8, 5.6, 7.9]],
#                           [[1.2, 3.1, 3.2], [9.9, 9.8, 9.7], [8.3, 8.9, 8.2], [2.1, 2.2, 2.3], [3.4, 3.5, 3.7], [5.1, 5.2, 5.3]],
#                           [[-1.2, -1.1, -1.], [0.1, 0.2, 0.3]],
#                           [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6], [4.7, 4.8, 4.9]]]
# id = tf.constant([0, 0, 0, 0, 0])
#
# x = tf.zeros(5)
# x[-1] = 1
# print(x)

# id1 = tf.constant([1, 0])
# print(tf.keras.preprocessing.sequence.pad_sequences([id, id1]))

# t = tf.ragged.constant([[1, 3, 2, 4], [1], [1, 2], [7, 5, 4], [1, 2, 3, 4, 5]])
# x = tf.keras.preprocessing.sequence.pad_sequences(sequences=t.to_list(), dtype='float32',
#                                                   padding='post', value=0.)
# print(x)

# mel_paths = ["training_data/biaobei/mels/000001.wav.npy",
#              "training_data/biaobei/mels/000002.wav.npy",
#              "training_data/biaobei/mels/000003.wav.npy"]
#
# mel_targets = []
# for mel_path in mel_paths:
#     mel_target = np.load(mel_path).transpose()
#     print(mel_target.shape)
#     mel_targets.append(mel_target)

# print(mel_targets)
# x = tf.keras.preprocessing.sequence.pad_sequences(sequences=mel_targets, dtype='float32',
#                                                   padding='post', value=0.)
# print(x)



# seqs = tf.data.Dataset.from_tensor_slices(seq)
# mel_targets = tf.data.Dataset.from_tensor_slices(mel)
# speaker_id = tf.data.Dataset.from_tensor_slices(id)


# seqs = seqs.map(lambda x: x).padded_batch(2, padded_shapes=[None])
# mel_targets = mel_targets.map(lambda x: x.to_tensor()).padded_batch(2, padded_shapes=[None, None])
# speaker_id = speaker_id.batch(2)

# dataset = tf.data.Dataset.from_tensor_slices((seqs, mel_targets, speaker_id))


# def ragged_tensor_to_tensor(seq, mel, id):
#     return seq, mel.to_tensor(), id
#
#
# dataset = tf.data.Dataset.from_tensor_slices((seq, mel, id))
# dataset = dataset.map(ragged_tensor_to_tensor).padded_batch(batch_size=2,
#                                                             padded_shapes=([None], [None, None], []),
#                                                             padding_values=(0, -4., 0))
#
# for x in dataset:
#     print(x)


# class embedding(tf.keras.layers.Layer):
#     def __init__(self):
#         super(embedding, self).__init__()
#         self.embedding = tf.keras.layers.Embedding(10, 10, mask_zero=True)
#         self.lstm0 = tf.keras.layers.LSTMCell(20, recurrent_dropout=0.1)
#         self.lstm1 = tf.keras.layers.LSTMCell(5, recurrent_dropout=0.1)
#
#         # bs = 3
#         #
#         # self.h0 = tf.zeros([bs, 20])
#         # self.c0 = tf.zeros([bs, 20])
#         #
#         # self.h1 = tf.zeros([bs, 5])
#         # self.c1 = tf.zeros([bs, 5])
#
#     # def init_state(self, memory, mask):
#     #     bs = tf.shape(memory)[0]
#     #
#     #     self.h0 = tf.zeros([bs, 20])
#     #     self.c0 = tf.zeros([bs, 20])
#     #
#     #     self.h1 = tf.zeros([bs, 5])
#     #     self.c1 = tf.zeros([bs, 5])
#     #
#     #     self.mask = mask
#
#     def build(self, input_shape):
#         bs = input_shape[0]
#
#         self.h0 = tf.zeros([bs, 20])
#         self.c0 = tf.zeros([bs, 20])
#
#         self.h1 = tf.zeros([bs, 5])
#         self.c1 = tf.zeros([bs, 5])
#
#         # self.mask = mask
#
#     def call(self, inputs, training=None):
#         embedding_output = self.embedding(inputs)
#         mask = embedding_output._keras_mask
#
#         # self.init_state(embedding_output, mask)
#         outputs = []
#         for frame in tf.unstack(embedding_output, axis=1):
#             out1, [self.h0, self.c0] = self.lstm0(frame, [self.h0, self.c0])
#             output, [self.h1, self.c1] = self.lstm1(out1, [self.h1, self.c1])
#
#
#             outputs.append([output])
#
#         return outputs
#
#
# @tf.function
# def test(model, seqs):
#     output = model(seqs)
#     return output
#
#
#
# if __name__ == '__main__':
#     # seqs = tf.constant([[1, 2, 3, 0], [3, 4, 5, 7], [1, 0, 0, 0]])
#     # model = embedding()
#     # output = test(model, seqs)
#     # print(tf.shape(output))
#     # print(output)


dict = {}
dict.keys()





