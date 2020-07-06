import tensorflow as tf
import argparse
import os
import audio
from Tacotron2.model_loader import parse_Tacotron2_args, get_Tacotron2_model
from Tacotron2.text import text_to_sequence
from Tacotron2.text import cmudict, thchsdict
from Text_Norm.text_normalization import text_normalize, text_segment


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input-file', type=str, default="text.txt", help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', type=str, default="outputs", help='output folder to save audio (file per phrase)')
    parser.add_argument('--checkpoint-path', type=str, default="logs", help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('-id', '--speaker-id', default=0, type=int, help='Speaker identity')
    parser.add_argument('-sn', '--speaker-num', default=1, type=int, help='Speaker number')
    parser.add_argument('-sr', '--sampling-rate', default=16000, type=int, help='Sampling rate')

    return parser


def main():
    parser = argparse.ArgumentParser(description='Tensorflow2 Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
    parser = parse_Tacotron2_args(parser)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    tacotron2_model = get_Tacotron2_model(args, training=False)
    checkpoint = tf.train.Checkpoint(Tacotron2=tacotron2_model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint_path))

    thchsdict_path = os.path.abspath(r'dictionary/thchs_tonebeep')
    g_thchsdict = thchsdict.THCHSDict(thchsdict_path)
    cmudict_path = os.path.abspath(r'dictionary/cmudict-0.7b')
    g_cmudict = cmudict.CMUDict(cmudict_path, keep_ambiguous=True)
    g_text_cleaners = ['basic_cleaners']

    text = [
        "明天早上就是NBA总决赛开打的时间啦，50%的人因为上班看不了。",
        "这是新版本，支持中英文混合语音合成哦。",
        "这是你新买的iphone11 pro max版本吗？用着感觉怎么样",
        "车牌号码是沪B79533，另一辆是闽D63363，其它的你可以打12580查询，罚金500元。",
        "can you speak chinese?"
        "今天天气很好，今天天气很好，今天天气很好，今天天气很好，今天天气很好，今天天气很好，今天天气很好，今天天气很好，今天天气很好。"
    ]

    sentences = list(map(text_normalize, text))
    for i, sentence in enumerate(sentences):
        sequence = text_to_sequence(sentence, g_text_cleaners, g_cmudict, g_thchsdict)
        model_outputs = tacotron2_model.infer(sequence)
        wav = audio.inv_mel_spectrogram(tf.squeeze(model_outputs[1]).numpy().transpose())
        audio.save_wav(wav, os.path.join(args.output, 'test_{}.wav'.format(i)))









if __name__ == '__main__':
    main()