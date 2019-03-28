import glob
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from advoc.audioio import save_as_wav
from advoc.spectral import melspec_to_waveform

spec_dir = sys.argv[1]
fs = float(sys.argv[2])

heuristic_dir = spec_dir.replace('Spectrogram', 'Waveform')
if heuristic_dir[-1] == '/':
  heuristic_dir = heuristic_dir[:-1]
heuristic_dir += '_GANWavenetVocoder'

if os.path.isdir(heuristic_dir):
  shutil.rmtree(heuristic_dir)
os.makedirs(heuristic_dir)

infer_fp = './wavenet_model/infer.meta'
ckpt_fp = './wavenet_model/model.ckpt-56921'

saver = tf.train.import_meta_graph(infer_fp)
with tf.Session() as sess:
  saver.restore(sess, ckpt_fp)

  graph = tf.get_default_graph()
  step = graph.get_tensor_by_name('global_step:0')
  print('Restored from {}'.format(sess.run(step)))

  spec = graph.get_tensor_by_name('long_spec:0')
  spec_chopped = graph.get_tensor_by_name('spec_chopped:0')

  input_spec = graph.get_tensor_by_name('input_spec:0')
  vocoded_wave = graph.get_tensor_by_name('vocoded_wave:0')

  spec_fps = glob.glob(os.path.join(spec_dir, '*.npy'))
  for spec_fp in tqdm(spec_fps):
    spec_fn = os.path.split(spec_fp)[1].split('.')[0]
    m = np.load(spec_fp).astype(np.float64)

    m_chopped = sess.run(spec_chopped, {spec: m})
    frame_wave = []
    for frame in m_chopped:
      frame_wave.append(sess.run(vocoded_wave, {input_spec: frame}))
    frame_wave = np.concatenate(frame_wave, axis=1)
    w = np.transpose(frame_wave, [1, 0, 2])
    w = w[:m.shape[0] * 256]

    wav_fp = os.path.join(heuristic_dir, spec_fn + '.wav')
    save_as_wav(wav_fp, int(fs), w)
    os.chmod(wav_fp, 0o555)

  os.chmod(heuristic_dir, 0o555)
