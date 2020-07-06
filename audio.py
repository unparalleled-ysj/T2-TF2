import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile


num_mels = 80
n_fft = 1024
sample_rate = 16000
hop_size = 200
win_size = 800
preemphasis_value = 0.97
min_level_db = -120
ref_level_db = 20
power = 1.2
griffin_lim_iters = 60
fmax = 7600
fmin = 50
max_abs_value = 4.


def dc_notch_filter(wav):
	# code from speex
	notch_radius = 0.982
	den = notch_radius ** 2 + 0.7 * (1 - notch_radius) ** 2
	b = np.array([1, -2, 1]) * notch_radius
	a = np.array([1, -2 * notch_radius, den])
	return signal.lfilter(b, a, wav)


def load_wav(path, sr):
	return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path):
	wav = dc_notch_filter(wav)
	wav = wav / np.abs(wav).max() * 0.999
	f1 = 0.5 * 32767 / max(0.01, np.max(np.abs(wav)))
	f2 = np.sign(wav) * np.power(np.abs(wav), 0.95)
	wav = f1 * f2
	#proposed by @dsmiller
	wavfile.write(path, sample_rate, wav.astype(np.int16))


def preemphasis(wav, k):
	return signal.lfilter([1, -k], [1], wav)


def inv_preemphasis(wav, k):
	return signal.lfilter([1], [1, -k], wav)


def get_hop_size():
	return hop_size

def linearspectrogram(wav):
	D = _stft(preemphasis(wav, preemphasis_value))
	S = _amp_to_db(np.abs(D)) - ref_level_db
	return _normalize(S)


def melspectrogram(wav):
	D = _stft(preemphasis(wav, preemphasis_value))
	S = _amp_to_db(_linear_to_mel(np.abs(D))) - ref_level_db
	return _normalize(S)


def inv_linear_spectrogram(linear_spectrogram):
	'''Converts linear spectrogram to waveform using librosa'''
	D = _denormalize(linear_spectrogram)
	S = _db_to_amp(D + ref_level_db) #Convert back to linear
	return inv_preemphasis(_griffin_lim(S ** power), preemphasis_value)


def inv_mel_spectrogram(mel_spectrogram):
	'''Converts mel spectrogram to waveform using librosa'''
	D = _denormalize(mel_spectrogram)
	S = _mel_to_linear(_db_to_amp(D + ref_level_db))  # Convert back to linear
	return inv_preemphasis(_griffin_lim(S ** power), preemphasis_value)


def _griffin_lim(S):
	'''librosa implementation of Griffin-Lim
	Based on https://github.com/librosa/librosa/issues/434
	'''
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles)
	for i in range(griffin_lim_iters):
		angles = np.exp(1j * np.angle(_stft(y)))
		y = _istft(S_complex * angles)
	return y


def _stft(y):
	return librosa.stft(y=y, n_fft=n_fft, hop_length=get_hop_size(), win_length=win_size)


def _istft(y):
	return librosa.istft(y, hop_length=get_hop_size(), win_length=win_size)


# Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram):
	global _inv_mel_basis
	if _inv_mel_basis is None:
		_inv_mel_basis = np.linalg.pinv(_build_mel_basis())
	return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis():
	assert fmax <= sample_rate // 2
	return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels,
							   fmin=fmin, fmax=fmax)


def _amp_to_db(x):
	min_level = np.exp(min_level_db / 20 * np.log(10))
	return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
	return np.power(10.0, (x) * 0.05)

def _normalize(S):
	return (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value


def _denormalize(D):
	return (((D + max_abs_value) * -min_level_db / (2 * max_abs_value)) + min_level_db)
