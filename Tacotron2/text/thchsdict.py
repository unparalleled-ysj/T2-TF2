class THCHSDict:
    '''Thin wrapper around THCHSDict data to convert pinyin to phoneme'''
    def __init__(self, file):
        with open(file, encoding='utf-8') as f:
            entries = _parse_thchsdict(f)
        self._entries = entries


    def __len__(self):
        return len(self._entries)


    def lookup(self, pinyin):
        '''Returns list of  phoneme of the given pinyin'''
        return self._entries.get(pinyin)


def _parse_thchsdict(file):
    thchsdict = {}
    for line in file:
        parts = line.strip('\n').split(' ')
        pinyin = parts[0]
        phoneme = ' '.join(parts[1:])
        thchsdict[pinyin] = phoneme
    return thchsdict

