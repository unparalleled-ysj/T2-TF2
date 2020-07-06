import argparse
from Tacotron2.model import Tacotron2
from Tacotron2.text.symbols import symbols



def parse_Tacotron2_args(parent, add_help=False):
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)
    # misc parameters
    parser.add_argument('--mask_padding', default=False, type=bool, help='Use mask padding')
    parser.add_argument('--n_mel_channels', default=80, type=int, help='Number of bins in mel-spec')
    parser.add_argument('--mel_pad_val', default=-5, type=int, help='Corresponding to silence')

    # symbols parameters
    len_symbols = len(symbols)
    symbol = parser.add_argument_group('symbol parameters')
    symbol.add_argument('--n_symbols', default=len_symbols, type=int,
                        help='Number of symbols in dictionary')
    symbol.add_argument('--symbols_embedding_dim', default=512, type=int,
                        help='input embedding dimension')

    # encoder parameters
    encoder = parser.add_argument_group('encoder parameters')
    encoder.add_argument('--encoder_kernel_size', default=5, type=int, help='Encoder kernel size')
    encoder.add_argument('--encoder_n_convolutions', default=3, type=int,
                         help='Number of encoder convolution layers')
    encoder.add_argument('--encoder_embedding_dim', default=512, type=int,
                         help='Encoder embedding dimension')

    # decoder parameters
    decoder = parser.add_argument_group('decoder parameters')
    decoder.add_argument('--n_frames_per_step', default=1, type=int,
                         help='Number of frames processed per step')
    decoder.add_argument('--decoder_rnn_dim', default=1024, type=int,
                         help='Number of unit in decoder LSTM layers')
    decoder.add_argument('--decoder_n_lstms', default=2, type=int,
                         help='Number of decoder LSTM layers')
    decoder.add_argument('--prenet_dim', default=256, type=int,
                         help='Number of ReLU units in prenet layers')
    decoder.add_argument('--max_decoder_steps', default=1000, type=int,
                         help='Maximum number of output mel-spec')
    decoder.add_argument('--gate_threshold', default=0.5, type=float,
                         help='Probability threshold for stop token')
    decoder.add_argument('--p_decoder_dropout', default=0.1, type=float,
                         help='Dropout probability for decoder LSTM')

    # attention parameters
    attention = parser.add_argument_group('attention parameters')
    attention.add_argument('--attention_dim', default=128, type=int,
                           help='Dimension of attention hidden represetation')

    # location layer parameters
    location = parser.add_argument_group('location parameters')
    location.add_argument('--attention_location_n_filters', default=32, type=int,
                          help='Number of filters for location-sensitive-attention')
    location.add_argument('--attention_location_kernel_size', default=31, type=int,
                          help='Kernel size for location-sensitive-attention')

    # Mel-post processing net work parameters
    postnet = parser.add_argument_group('postnet parameters')
    postnet.add_argument('--postnet_embedding_dim', default=512, type=int,
                         help='Postnet embedding dimension')
    postnet.add_argument('--postnet_kernel_size', default=5, type=int,
                         help='Postnet kernel size')
    postnet.add_argument('--postnet_n_convolutions', default=5, type=int,
                         help='Number of postnet convolutions')

    return parser


def get_Tacotron2_model(args, training=None):
    config = dict(
        # optimization
        mask_padding=args.mask_padding,
        # audio
        n_mel_channels=args.n_mel_channels,
        # symbols
        n_symbols=args.n_symbols,
        symbols_embedding_dim=args.symbols_embedding_dim,
        # encoder
        encoder_kernel_size=args.encoder_kernel_size,
        encoder_n_convolutions=args.encoder_n_convolutions,
        encoder_embedding_dim=args.encoder_embedding_dim,
        # attention
        attention_dim=args.attention_dim,
        # attention location
        attention_location_n_filters=args.attention_location_n_filters,
        attention_location_kernel_size=args.attention_location_kernel_size,
        # decoder
        n_frames_per_step=args.n_frames_per_step,
        prenet_dim=args.prenet_dim,
        decoder_rnn_dim=args.decoder_rnn_dim,
        max_decoder_steps=args.max_decoder_steps,
        gate_threshold=args.gate_threshold,
        decoder_n_lstms=args.decoder_n_lstms,
        p_decoder_dropout=args.p_decoder_dropout,
        # postnet
        postnet_embedding_dim=args.postnet_embedding_dim,
        postnet_kernel_size=args.postnet_kernel_size,
        postnet_n_convolutions=args.postnet_n_convolutions
    )

    model = Tacotron2(**config)

    return model






