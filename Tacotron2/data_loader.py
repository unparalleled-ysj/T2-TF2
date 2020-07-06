import os
from Tacotron2.text import text_to_sequence
from Tacotron2.text import cmudict, thchsdict


class LoadData(object):
    def __init__(self, args):
        anchor_dirs = args.training_anchor_dirs
        self.speaker_num = len(anchor_dirs)
        self.meta_dirs = [os.path.join(args.dataset_path, anchor_dirs[i])
                          for i in range(self.speaker_num)]
        self.text_cleaners = args.text_cleaners

        thchsdict_path = os.path.abspath(r'dictionary/thchs_tonebeep')
        self._thchsdict = thchsdict.THCHSDict(thchsdict_path)

        cmudict_path = os.path.abspath(r'dictionary/cmudict-0.7b')
        self._cmudict = cmudict.CMUDict(cmudict_path, keep_ambiguous=True)

    def get_data(self):
        seqs = []
        mels_path = []
        speaker_ids = []
        for speaker_id, speaker in enumerate(self.meta_dirs):
            with open(os.path.join(speaker, 'train.txt'), encoding='utf-8')as f:
                for line in f:
                    content = line.strip().split('|')
                    text = content[-1]
                    seq = self.get_sequence(text)
                    mel_path = os.path.join(speaker, 'mels', content[0])
                    seqs.append(seq)
                    mels_path.append(mel_path)
                    speaker_ids.append(speaker_id)
        return seqs, mels_path, speaker_ids

    def get_sequence(self, text):
        return text_to_sequence(text, self.text_cleaners, self._cmudict, self._thchsdict)

















