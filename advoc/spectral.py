from functools import lru_cache

import librosa
import lws
import numpy as np


def stft(x, nfft, nhop):
  """Performs the short-time Fourier transform on a waveform.

  Args:
    x: nd-array dtype float32 of shape [?, 1, 1].
    nfft: FFT size.
    nhop: Window size.

  Returns:
    nd-array dtype complex128 of shape [?, (nfft // 2) + 1, 1] containing the features.
  """

  return lws.lws(nfft, nhop).stft(x[:, 0, 0])[:, :, np.newaxis]


@lru_cache(maxsize=4)
def create_mel_filterbank(*args, **kwargs):
  return librosa.filters.mel(*args, **kwargs)


# NOTE: nfft and hop are configured for fs=20480
def waveform_to_melspec(
    x,
    fs,
    nfft,
    nhop,
    mel_min=125,
    mel_max=7600,
    mel_num_bins=80,
    norm_allow_clipping=True,
    norm_min_level_db=-100,
    norm_ref_level_db=20):
  """Transforms waveform into mel spectrogram feature representation.

  References:
    - https://github.com/r9y9/wavenet_vocoder
    - https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
    - https://github.com/r9y9/wavenet_vocoder/blob/master/hparams.py

  Args:
    x: nd-array dtype float32 of shape [?, 1, 1].
    fs: Sample rate of x.
    nfft: FFT size.
    nhop: Window size.
    mel_min: Minimum frequency for mel transform.
    mel_max: Maximum frequency for mel transform.
    mel_num_bins: Number of mel bins.
    norm_allow_clipping: If False, throws error if data is clipped during norm.
    norm_min_level_db: Minimum dB level.
    norm_ref_level_db: Maximum dB level (clips between this and 0).

  Returns:
    nd-array dtype float32 of shape [?, nmels, 1] containing the features.
  """
  nsamps, nfeats, nch = x.shape
  if nfeats != 1:
    raise ValueError()
  if nch != 1:
    raise NotImplementedError('Can only extract features from monaural signals')

  # TODO: figure out centering
  X = stft(x, nfft, nhop)[:, :, 0]
  X_mag = np.abs(X)

  mel_filterbank = librosa.filters.mel(
      fs, nfft, fmin=mel_min, fmax=mel_max, n_mels=mel_num_bins)
  X_mel = np.swapaxes(np.dot(mel_filterbank, X_mag.T), 0, 1)

  min_level = np.exp(norm_min_level_db / 20 * np.log(10))
  X_mel_db = 20 * np.log10(np.maximum(min_level, X_mel)) - norm_ref_level_db

  if not norm_allow_clipping:
    assert X_mel_db.max() <= 0 and X_mel_db.min() - norm_min_level_db >= 0
  X_mel_dbnorm = np.clip((X_mel_db - norm_min_level_db) / -norm_min_level_db, 0, 1)

  return X_mel_dbnorm[:, :, np.newaxis]


def waveform_to_tacotron2_feats(x):
  """Transforms waveform into mel spectrogram feature representation.

  Transforms waveform into feature representation for as described in original Tacotron 2 paper. No open source implementation so cannot gaurantee correctness. Reference:
    - https://arxiv.org/pdf/1712.05884.pdf

  Args:
    x: nd-array dtype float32 of shape [?, 1, 1] at 24000Hz.

  Returns:
    nd-array dtype float32 of shape [?, 80, 1] at 80Hz.
  """
  return waveform_to_melspec(
      x,
      fs=24000,
      nfft=1200,
      nhop=300,
      norm_min_level_db=-40)


def waveform_to_r9y9_feats(x):
  """Transforms waveform into unofficial mel spectrogram feature representation.

  Transforms waveform into feature representation for unofficial reimplementation of WaveNet vocoder. Unit tests guaranteeing parity with implementation. References:
    - https://github.com/r9y9/wavenet_vocoder
    - https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
    - https://github.com/r9y9/wavenet_vocoder/blob/master/hparams.py

  Args:
    x: nd-array dtype float32 of shape [?, 1, 1] at 22050Hz.

  Returns:
    nd-array dtype float32 of shape [?, 80, 1] at 86.13Hz.
  """
  return waveform_to_melspec(
      x,
      fs=22050,
      nfft=1024,
      nhop=256)
