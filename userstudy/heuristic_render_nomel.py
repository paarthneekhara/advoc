import glob
import os
import shutil
import sys

import numpy as np
from tqdm import tqdm

from advoc.audioio import decode_audio, save_as_wav
from advoc.spectral import stft, magspec_to_waveform_griffin_lim, magspec_to_waveform_lws

wav_dir = sys.argv[1]
heuristic_dir = sys.argv[2]
phase_estimation = sys.argv[3]

if os.path.isdir(heuristic_dir):
  shutil.rmtree(heuristic_dir)
os.makedirs(heuristic_dir)

wav_fps = glob.glob(os.path.join(wav_dir, '*.wav'))
for wav_fp in tqdm(wav_fps):
  wav_fn = os.path.split(wav_fp)[1].split('.')[0]

  fs, w = decode_audio(wav_fp)
  wlen = w.shape[0]
  assert fs == 22050

  magspec = np.abs(stft(w, 1024, 256))

  if phase_estimation == 'lws':
    w = magspec_to_waveform_lws(magspec, 1024, 256)
  elif phase_estimation[:2] == 'gl':
    try:
      ngl = int(phase_estimation[2:])
    except:
      raise ValueError()
    w = magspec_to_waveform_griffin_lim(magspec, 1024, 256, ngl)
  else:
    raise ValueError()

  w = w[:wlen]
  assert w.shape[0] == wlen

  wav_fp = os.path.join(heuristic_dir, wav_fn + '.wav')
  save_as_wav(wav_fp, int(fs), w)
  os.chmod(wav_fp, 0o555)

os.chmod(heuristic_dir, 0o555)
