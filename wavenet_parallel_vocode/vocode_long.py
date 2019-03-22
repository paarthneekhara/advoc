WAVELEN = 65536

import sys

import numpy as np

from advoc.audioio import decode_audio, save_as_wav
from advoc.spectral import waveform_to_r9y9_melspec, r9y9_melspec_to_waveform

wav_fp, ckpt_fp = sys.argv[1:]

fs, wav = decode_audio(wav_fp, fastwav=True)
assert wav.shape[0] >= WAVELEN
wav = wav[:WAVELEN]

save_as_wav('vocode_orig.wav', fs, wav)

spec = waveform_to_r9y9_melspec(wav)
heuristic = r9y9_melspec_to_waveform(spec)

save_as_wav('vocode_heuristic.wav', fs, heuristic)

import tensorflow as tf

from wavenet import build_nsynth_wavenet_decoder

input_wave = tf.random.normal([1, spec.shape[0] * 256, 1], dtype=tf.float32)
input_spec = tf.constant(spec[np.newaxis, :, :, 0], dtype=tf.float32)

with tf.variable_scope('vocoder'):
  vocoded_wave = build_nsynth_wavenet_decoder(
      input_wave,
      input_spec,
      causal=False,
      output_width=1,
      num_stages=10,
      num_layers=20,
      filter_length=3,
      width=128,
      skip_width=128)[0, :, :, tf.newaxis]
trainable_vars = tf.trainable_variables(scope='vocoder')

saver = tf.train.Saver(trainable_vars)

with tf.Session() as sess:
  saver.restore(sess, ckpt_fp)

  _vocoded_wave = sess.run(vocoded_wave)

  save_as_wav('vocode_wavenet.wav', fs, _vocoded_wave)
