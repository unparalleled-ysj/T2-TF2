""" from https://github.com/keithito/tacotron """
from Tacotron2.text import cleaners
from Tacotron2.text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names, cmudict, thchsdict):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  text = [get_phoneme(thchsdict, word) for word in text.split(' ')]
  text = ' '.join([get_arpabet(cmudict, word) for word in text])

  sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
  # Append EOS token
  sequence.append(_symbol_to_id['~'])
  return sequence



def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]
  # result = []
  # space = _symbol_to_id[' ']
  # for s in symbols.split(' '):
  #   if _should_keep_symbol(s):
  #     result.append(_symbol_to_id[s])
  #     result.append(space)
  # return result


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'


def get_arpabet(cmudict, word):
  arpabet = cmudict.lookup(word)
  return '%s' % arpabet[0] if arpabet is not None else word


def get_phoneme(thchsdict, pinyin):
  phoneme = thchsdict.lookup(pinyin)
  return '%s' % phoneme if phoneme is not None else pinyin
