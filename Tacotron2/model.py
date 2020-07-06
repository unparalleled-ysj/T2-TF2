import tensorflow as tf
from tensorflow import keras


class PreNet(keras.layers.Layer):
    def __init__(self, out_dim):
        super(PreNet, self).__init__()

        self.fc1 = keras.layers.Dense(out_dim, activation='relu', bias_initializer='glorot_uniform')
        self.fc2 = keras.layers.Dense(out_dim, activation='relu', bias_initializer='glorot_uniform')

    def call(self, inputs, training=None):
        x = tf.nn.dropout(self.fc1(inputs), rate=0.5)
        outputs = tf.nn.dropout(self.fc2(x), rate=0.5)
        return outputs


class Tanh_layer(keras.layers.Layer):
    def __init__(self):
        super(Tanh_layer, self).__init__()

    def call(self, inputs, **kwargs):
        return tf.nn.tanh(inputs)


class PostNet(keras.layers.Layer):
    def __init__(self, n_mel_channels, postnet_n_convolutions,
                 postnet_embedding_dim, postnet_kernel_size):
        super(PostNet, self).__init__()
        self.convolutions_without_last_layer = keras.Sequential()
        for i in range(postnet_n_convolutions - 1):
            self.convolutions_without_last_layer.add(
                keras.layers.Conv1D(filters=postnet_embedding_dim,
                                    kernel_size=postnet_kernel_size,
                                    padding='same')
            )
            self.convolutions_without_last_layer.add(
                keras.layers.BatchNormalization()
            )
            self.convolutions_without_last_layer.add(
                Tanh_layer()
            )
        self.convolutions_last_layer = keras.layers.Conv1D(filters=n_mel_channels,
                                                           kernel_size=postnet_kernel_size,
                                                           padding='same')

    def call(self, inputs, training=None):
        x = self.convolutions_without_last_layer(inputs)
        outputs = self.convolutions_last_layer(x)
        return outputs


class Encoder(keras.layers.Layer):
    def __init__(self, encoder_n_convolutions, encoder_embedding_dim, encoder_kernel_size):
        super(Encoder, self).__init__()

        self.convolutions = keras.Sequential()
        for i in range(encoder_n_convolutions):
            self.convolutions.add(keras.layers.Conv1D(filters=encoder_embedding_dim,
                                          kernel_size=encoder_kernel_size,
                                          padding='same'))
            self.convolutions.add(keras.layers.BatchNormalization())
            self.convolutions.add(keras.layers.ReLU())

        self.encoder_lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(units=int(encoder_embedding_dim/2),
                                 recurrent_dropout=0.1,
                                 return_sequences=True),
            merge_mode='concat')

    def call(self, inputs, training=None):
        x = self.convolutions(inputs)
        outputs = self.encoder_lstm(x, training=training)
        return outputs


class Attention(keras.layers.Layer):
    def __init__(self, attention_dim, attention_n_filters, attention_kernel_size):
        super(Attention, self).__init__()

        self.query_layer = keras.layers.Dense(attention_dim)
        self.memory_layer = keras.layers.Dense(attention_dim)
        self.V = keras.layers.Dense(1)
        self.location_layer = LocationLayer(attention_n_filters,
                                            attention_kernel_size,
                                            attention_dim)

    def get_alignment_energies(self, query, memory, attention_weight_cum):
        '''
        :param query: decoder output [B, decoder_dim]
        :param memory: encoder output [B, T_in, embed_dim]
        :param attention_weight_cum: cumnlativa attention weight [B, max_time, 1]
        :return: aligment [B, T_in]
        '''

        # [B, 1, attenion_dim]
        query = self.query_layer(tf.expand_dims(query, 1))
        # [B, T_in, attenion_dim]
        key = self.memory_layer(memory)
        # [B, max_time, attenion_dim]
        location_sensitive_weight = self.location_layer(attention_weight_cum)

        # score function
        # [B, T_in, 1]
        energies = self.V(tf.nn.tanh(query + key + location_sensitive_weight))
        # [B, T_in]
        energies = tf.squeeze(energies, -1)

        return energies

    def call(self, query, memory, attention_weight_cum, mask=None, training=None):
        '''
        :param query: decoder rnn last output [B, decoder_dim]
        :param memory: encoder outputs [B, T_in, embed_dim]
        :param attention_weight_cum: cumulative attention weight [B, 1, max_time]
        :param mask: binary mask for padded data
        :param training: is training
        :return:
        '''
        # [B, T_in]
        alignment = self.get_alignment_energies(query, memory, attention_weight_cum)

        if mask is not None:
            alignment += (mask * -1e9)

        # [B, T_in]
        attention_weight = tf.nn.softmax(alignment, axis=-1)
        # [B, 1, embed_dim] = [B, 1, T_in] * [B, T_in, embed_dim]
        attention_context = tf.matmul(tf.expand_dims(attention_weight, axis=1), memory)
        # [B, embed_dim]
        attention_context = tf.squeeze(attention_context, axis=1)

        return attention_context, attention_weight


class LocationLayer(keras.layers.Layer):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        self.location_conv = keras.layers.Conv1D(filters=attention_n_filters,
                                                 kernel_size=attention_kernel_size,
                                                 padding='same',
                                                 bias_initializer='glorot_uniform')
        self.location_dense = keras.layers.Dense(units=attention_dim)

    def call(self, attention_weights_cum, training=None):
        '''
        convolution for attention_weights_cum
        :param attention_weights_cum: [B, max_time, 1]
        :param training: if training
        :return: processed_attention_weights [B, max_time, attention_dim]
        '''
        # [B, max_time, attention_n_filters]
        processed_attention_weights = self.location_conv(attention_weights_cum)
        # [B, max_time, attention_dim]
        processed_attention_weights = self.location_dense(processed_attention_weights)
        return processed_attention_weights


class Decoder(keras.layers.Layer):
    def __init__(self, n_mel_channels, n_frames_per_step,
                 encoder_embedding_dim, attention_dim,
                 attention_location_n_filters,
                 attention_location_kernel_size,
                 prenet_dim, decoder_rnn_dim,
                 max_decoder_steps, gate_threshold,
                 decoder_n_lstms, p_decoder_dropout):
        super(Decoder, self).__init__()

        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.decoder_n_lstms = decoder_n_lstms
        self.p_decoder_dropout = p_decoder_dropout

        self.prenet = PreNet(prenet_dim)

        self.h0 = None
        self.c0 = None
        self.h1 = None
        self.c1 = None
        self.lstm0 = keras.layers.LSTMCell(decoder_rnn_dim)
        self.lstm1 = keras.layers.LSTMCell(decoder_rnn_dim)

        self.attention_weights = None
        self.attention_weights_cum = None
        self.attention_context = None
        self.memory = None
        self.mask = None
        self.attention_layer = Attention(attention_dim,
                                         attention_location_n_filters,
                                         attention_location_kernel_size)

        self.linear_projection = keras.layers.Dense(n_mel_channels*n_frames_per_step)
        self.gate_layer = keras.layers.Dense(n_frames_per_step)

    def call(self, memory, targets, mask, training=None):
        '''
        Decoder forward pass for training
        :param memory: Encoder outputs [B, T_in, Embed_dim]
        :param targets: Decoder inputs for teacher forcing. i.e. mel-specs [B, len, n_mel_channels]
        :param mask: for attention masking [B, T_in]
        :return: mel_outputs, gate_outputs, alignments
        '''
        # [B, 1, n_mel_channels]
        go_frame = tf.expand_dims(tf.zeros([tf.shape(memory)[0], self.n_mel_channels]), axis=1)
        # [B, len+1, n_mel_channels]
        decoder_inputs = tf.concat((go_frame, targets), axis=1)
        # [B, len+1, prenet_dim]
        prenet_outputs = self.prenet(decoder_inputs, training=training)

        self.initialize_decoder_states(memory, mask)

        mel_outputs, gate_outputs, alignments = [], [], []

        # size - 1 for ignoring EOS symbol
        for prenet_output in tf.unstack(prenet_outputs, axis=1)[:-1]:
            mel_output, gate_output, attention_weights = self.decoder(prenet_output=prenet_output,
                                                                      training=training)

            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
            alignments += [attention_weights]

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

    def infer(self, memory, mask, training=False):
        '''
        Docoder inference
        :param memory: Encoder outputs
        :param mask: for attention masking
        :return: mel outputs: mel outputs from the decoder
                 gate_outputs: gate outputs from the decoder
                 alignment: sequence of attention weights from the decoder
        '''
        # [B, n_mel_channels]
        frame = tf.zeros([tf.shape(memory)[0], self.n_mel_channels])
        self.initialize_decoder_states(memory, mask)
        mel_lengths = tf.zeros([tf.shape(memory)[0]], dtype=tf.int32)

        mel_outputs, gate_outputs, alignments = [], [], []

        while True:
            prenet_output = self.prenet(frame, training=training)
            mel_output, gate_output, alignment = self.decoder(prenet_output=prenet_output,
                                                              training=training)
            gate_output = tf.sigmoid(gate_output)
            finished = tf.reduce_all(tf.greater(gate_output, self.gate_threshold), axis=-1)
            mel_lengths += tf.cast(tf.logical_not(finished), tf.int32)
            if tf.reduce_all(finished):
                break
            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if len(mel_outputs) == self.max_decoder_steps:
                print("warning! Reached max decoder steps")
                break

            frame = mel_output[:, -self.n_mel_channels:]

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments, mel_lengths)

    def initialize_decoder_states(self, memory, mask=None):
        '''
        initializes attention rnn states, decoder rnn states, attention weights,
        attention cumulative weights, attention context, stores memory
        :param memory: Encoder outputs [B, T_in, embed_dim]
        :param mask: Mask for padded data if training, expect None for inference
        '''

        batch_size = tf.shape(memory)[0]
        max_time = tf.shape(memory)[1]

        self.h0 = tf.zeros([batch_size, self.decoder_rnn_dim])
        self.c0 = tf.zeros([batch_size, self.decoder_rnn_dim])

        self.h1 = tf.zeros([batch_size, self.decoder_rnn_dim])
        self.c1 = tf.zeros([batch_size, self.decoder_rnn_dim])

        self.attention_weights = tf.zeros([batch_size, max_time])
        self.attention_weights_cum = tf.zeros([batch_size, max_time])
        self.attention_context = tf.zeros([batch_size, self.encoder_embedding_dim])

        self.memory = memory
        self.mask = tf.cast(tf.logical_not(mask), dtype=tf.float32)

    def decoder(self, prenet_output, training=None):
        '''
        Decoder step using stored states, attention and memory
        :param prenet_output: previous mel output [B, prenet_dim]
        :return: mel_output [B, n_mel_channels*n_frames_per_step]
                 gata output energies [B, n_frams_per_step]
                 attention_weights [B, max_time]
        '''
        # [B, prenet_dim + encoder_embedding_dim]
        x = tf.concat((prenet_output, self.attention_context), axis=-1)
        out0, [self.h0, self.c0] = self.lstm0(x, [self.h0, self.c0], training=training)
        x = tf.nn.dropout(out0, self.p_decoder_dropout)
        # [B, decoder_rnn_dim + encoder_embedding_dim]
        x = tf.concat((x, self.attention_context), axis=-1)
        out1, [self.h1, self.c1] = self.lstm1(x, [self.h1, self.c1], training=training)
        self.query = tf.nn.dropout(out1, self.p_decoder_dropout)

        # [B, max_time, 1]
        attention_weights_cumulative = tf.expand_dims(self.attention_weights_cum, axis=-1)
        # [B, embed_dim], [B, max_time]
        self.attention_context, self.attention_weights = self.attention_layer(self.query,
                                                                              self.memory,
                                                                              attention_weights_cumulative,
                                                                              self.mask)

        self.attention_weights_cum = self.attention_weights_cum + self.attention_weights
        x = tf.concat((self.query, self.attention_context), axis=-1)
        # [B, n_mel_channels*n_frames_per_step]
        mel_output = self.linear_projection(x)
        # [B, n_frams_per_step]
        gate_output = self.gate_layer(x)

        return mel_output, gate_output, self.attention_weights

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments, mel_lengths=None):
        '''
        Prepares decoder outputs for output
        :param mel_outputs:
        :param gate_outputs:
        :param alignments:
        :param mel_lengths:
        :return: mel_outputs
                 gate_outputs: gate output energies
                 alignments
        '''
        B = tf.shape(mel_outputs)[1]
        # [T_out, B, T_in] -> [B, T_out, T_in]
        alignments = tf.transpose(tf.stack(alignments), perm=[1, 0, 2])
        # [T_out, B, n_frame_per_step] -> [B, T_out, n_frame_per_step]
        gate_outputs = tf.transpose(tf.stack(gate_outputs), perm=[1, 0, 2])
        # [B, T_out, n_frame_per_step] -> [B, T_out*n_frame_per_step]
        gate_outputs = tf.reshape(gate_outputs, [B, -1])
        # [T_out, B, n_channels*n_frame_per_step] -> [B, T_out, n_channels*n_frame_per_step]
        mel_outputs = tf.transpose(tf.stack(mel_outputs), perm=[1, 0, 2])
        # decouple frames per step -> [B, T_out, n_mel_channels]
        mel_outputs = tf.reshape(mel_outputs, [B, -1, self.n_mel_channels])
        # mel lengths scale to the target length
        if mel_lengths is not None:
            mel_lengths *= self.n_frames_per_step

        return mel_outputs, gate_outputs, alignments, mel_lengths


class Tacotron2(keras.Model):
    def __init__(self, mask_padding, n_mel_channels,
                 n_symbols, symbols_embedding_dim, encoder_kernel_size,
                 encoder_n_convolutions, encoder_embedding_dim,
                 attention_dim, attention_location_n_filters,
                 attention_location_kernel_size, n_frames_per_step,
                 prenet_dim, decoder_rnn_dim, max_decoder_steps, gate_threshold,
                 decoder_n_lstms, p_decoder_dropout,
                 postnet_embedding_dim, postnet_kernel_size,
                 postnet_n_convolutions):
        super(Tacotron2, self).__init__()
        self.embedding = keras.layers.Embedding(n_symbols, symbols_embedding_dim, mask_zero=True)
        self.encoder = Encoder(encoder_n_convolutions=encoder_n_convolutions,
                               encoder_embedding_dim=encoder_embedding_dim,
                               encoder_kernel_size=encoder_kernel_size)
        self.decoder = Decoder(n_mel_channels=n_mel_channels,
                               n_frames_per_step=n_frames_per_step,
                               encoder_embedding_dim=encoder_embedding_dim,
                               attention_dim=attention_dim,
                               attention_location_n_filters=attention_location_n_filters,
                               attention_location_kernel_size=attention_location_kernel_size,
                               prenet_dim=prenet_dim,
                               decoder_rnn_dim=decoder_rnn_dim,
                               max_decoder_steps=max_decoder_steps,
                               gate_threshold=gate_threshold,
                               decoder_n_lstms=decoder_n_lstms,
                               p_decoder_dropout=p_decoder_dropout)
        self.postnet = PostNet(n_mel_channels=n_mel_channels,
                               postnet_n_convolutions=postnet_n_convolutions,
                               postnet_embedding_dim=postnet_embedding_dim,
                               postnet_kernel_size=postnet_kernel_size,
                               )

    def call(self, inputs, training=None):
        texts, targets, speaker_id = inputs

        # [B, T_in, embed_dim]
        embedded_inputs = self.embedding(texts)
        mask = embedded_inputs._keras_mask

        # [B, T_in, encoder_dim]
        encoder_outputs = self.encoder(embedded_inputs, training=training)

        # [B, T_out, n_mel_channels], [B, T_out*n_frames_per_step], [B, T_out, T_in]
        mel_outputs_before, gate_outputs, alignments, _ = self.decoder(encoder_outputs,
                                                                       targets,
                                                                       mask,
                                                                       training)

        mel_outputs_after = mel_outputs_before + self.postnet(mel_outputs_before)

        return self.parse_outputs([mel_outputs_before, mel_outputs_after,
                                   gate_outputs, alignments])

    def parse_outputs(self, outputs, target_lengths=None):
        if target_lengths is not None:
            # [B, max_len]
            mask = tf.sequence_mask(target_lengths)

            outputs[0] = tf.ragged.boolean_mask(outputs[0], tf.expand_dims(mask, axis=-1))
            outputs[1] = tf.ragged.boolean_mask(outputs[1], tf.expand_dims(mask, axis=-1))
            outputs[2] = tf.ragged.boolean_mask(outputs[2], mask)

        return [output for output in outputs]

    def infer(self, texts, training=False):
        # [B, T_in, embed_dim]
        embedded_inputs = self.embedding(texts)
        mask = embedded_inputs._keras_mask
        encoder_outputs = self.encoder(embedded_inputs, training=training)
        mel_outputs_before, gate_outputs, \
        alignments, mel_lengths = self.decoder.infer(encoder_outputs,
                                                     mask,
                                                     training)
        mel_outputs_after = mel_outputs_before + self.postnet(mel_outputs_before)
        return self.parse_outputs([mel_outputs_before, mel_outputs_after,
                                   gate_outputs, alignments, mel_lengths])