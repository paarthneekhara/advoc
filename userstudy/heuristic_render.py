import glob
import os
import shutil
import sys

import numpy as np
from tqdm import tqdm

from advoc.audioio import save_as_wav
from advoc.spectral import melspec_to_waveform

spec_dir = sys.argv[1]
phase_estimation = sys.argv[2]
fs = float(sys.argv[3])

heuristic_dir = spec_dir.replace('Spectrogram', 'Waveform')
if heuristic_dir[-1] == '/':
  heuristic_dir = heuristic_dir[:-1]
heuristic_dir += '_' + phase_estimation.upper()

if os.path.isdir(heuristic_dir):
  shutil.rmtree(heuristic_dir)
os.makedirs(heuristic_dir)

spec_fps = glob.glob(os.path.join(spec_dir, '*.npy'))
for spec_fp in tqdm(spec_fps):
  spec_fn = os.path.split(spec_fp)[1].split('.')[0]
  m = np.load(spec_fp).astype(np.float64)
  w = melspec_to_waveform(m, fs, 1024, 256, phase_estimation=phase_estimation)
  wav_fp = os.path.join(heuristic_dir, spec_fn + '.wav')
  save_as_wav(wav_fp, int(fs), w)
