import glob
import os
import shutil
import sys

import lws
import numpy as np
from tqdm import tqdm

from advoc.audioio import decode_audio, save_as_wav
from advoc.spectral import stft, magspec_to_waveform_griffin_lim, magspec_to_waveform_lws, create_mel_filterbank, create_inverse_mel_filterbank

wav_dir = sys.argv[1]
heuristic_dir = sys.argv[2]

if os.path.isdir(heuristic_dir):
  shutil.rmtree(heuristic_dir)
os.makedirs(heuristic_dir)

melbasis = create_mel_filterbank(
    22050,
    1024,
    fmin=125.,
    fmax=7600.,
    n_mels=80)
invmelbasis = create_inverse_mel_filterbank(
    22050,
    1024,
    fmin=125.,
    fmax=7600.,
    n_mels=80)

wav_fps = glob.glob(os.path.join(wav_dir, '*.wav'))
for wav_fp in tqdm(wav_fps):
  wav_fn = os.path.split(wav_fp)[1].split('.')[0]

  fs, w = decode_audio(wav_fp)
  wlen = w.shape[0]
  assert fs == 22050

  lws_proc = lws.lws(1024, 256, mode='speech', perfectrec=False)

  W = lws_proc.stft(w[:, 0, 0])
  magspec = np.abs(W)
  phsspec = np.angle(W)

  melspec = np.matmul(magspec, np.transpose(melbasis))
  magspec = np.matmul(melspec, np.transpose(invmelbasis))

  W = magspec * np.exp(1j*phsspec)
  w = lws_proc.istft(W)
  w = w[:, np.newaxis, np.newaxis]

  w = w[:wlen]
  assert w.shape[0] == wlen

  wav_fp = os.path.join(heuristic_dir, wav_fn + '.wav')
  save_as_wav(wav_fp, int(fs), w)
  os.chmod(wav_fp, 0o555)

os.chmod(heuristic_dir, 0o555)
