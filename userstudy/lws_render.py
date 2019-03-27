import glob
import os
import shutil
import sys

import numpy as np
from tqdm import tqdm

from advoc.audioio import save_as_wav
from advoc.spectral import melspec_to_waveform

spec_dir = sys.argv[1]
fs = float(sys.argv[2])
wavlen = int(sys.argv[3])

lws_dir = spec_dir.replace('Spectrogram', 'Waveform')
if lws_dir[-1] == '/':
  lws_dir = lws_dir[:-1]
lws_dir += '_LWS'

if os.path.isdir(lws_dir):
  shutil.rmtree(lws_dir)
os.makedirs(lws_dir)

spec_fps = glob.glob(os.path.join(spec_dir, '*.npy'))
for spec_fp in tqdm(spec_fps):
  spec_fn = os.path.split(spec_fp)[1].split('.')[0]
  m = np.load(spec_fp).astype(np.float64)
  w = melspec_to_waveform(m, fs, 1024, 256, waveform_len=wavlen)
  assert w.shape[0] == wavlen
  wav_fp = os.path.join(lws_dir, spec_fn + '.wav')
  save_as_wav(wav_fp, int(fs), w)
