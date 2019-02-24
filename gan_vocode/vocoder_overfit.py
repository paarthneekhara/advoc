import sys

import librosa
import librosa.filters
import numpy as np
import lws

from wgpp.conv1d import WaveDecoderFactor256

fn = 'real_sc09.wav'
nsteps = int(sys.argv[1])
spec = 'wavenet_vocoder_mel'
use_noise = True
learn_noise = False
opt = 'l1'


import numpy as np
from scipy.io.wavfile import read as wavread, write as wavwrite
import tensorflow as tf

fs, _x = wavread(fn)
_x = np.reshape(_x, [50, -1])
_x = _x[0, :16384]
_x = _x.astype(np.float32)
_x /= 32767.

print ("Shape", _x.shape)

x = tf.constant(_x, dtype=tf.float32)

dec_input = []

# STFT
if spec == 'mag':
  X = tf.contrib.signal.stft(x, 128, 256, pad_end=True)
  X_mag = tf.abs(X)
  X_spec = X_mag[:, :-1]
elif spec == 'wavenet_vocoder_mel':
  # Adapted from https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
  # Only difference from out-of-box is that they use 22kHz
  nfft = 1024
  nhop = 256
  
  # TODO: Figure out what to do about center-vs-left-padding (lws uses center padding, decoder probably should as well)
  _X = lws.lws(nfft, nhop, mode='speech').stft(_x)[3:]
  _X_mag = np.abs(_X)

  _mel = librosa.filters.mel(16000, nfft, fmin=125, fmax=7600, n_mels=80)
  _X_mel = np.dot(_X_mag, _mel.T)

  min_level_db = -100
  ref_level_db = 20
  min_level = np.exp(min_level_db / 20. * np.log(10))
  _X_mel_db = 20. * np.log10(np.maximum(min_level, _X_mel)) - ref_level_db

  #assert _X_mel_db.max() <= 0 and _X_mel_db.min() - min_level_db >= 0
  _X_mel_dbnorm = np.clip((_X_mel_db - min_level_db) / -min_level_db, 0, 1)

  X_spec = tf.constant(_X_mel_dbnorm, dtype=tf.float32)
else:
  raise ValueError()
X_spec = tf.stop_gradient(X_spec)
dec_input.append(X_spec)

# Noise
opt_vars = []
if use_noise:
  if learn_noise:
    noise = tf.get_variable('noise', [64, 64])
    opt_vars.append(noise)
  else:
    noise = tf.constant(np.random.uniform(-1, 1, [64, 64]), dtype=tf.float32)
  dec_input.append(noise)

# dec_input is list of [64, 64]
# transform to [1, 64, 1, 64*len(dec_input)]
dec_input = tf.concat(dec_input, axis=1)[tf.newaxis, :, tf.newaxis, :]
print(dec_input)

with tf.variable_scope('Dec'):
  Dec = WaveDecoderFactor256()
  Dec_x = Dec(dec_input, training=True)[0, :, 0, 0]
opt_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dec'))

if opt == 'l1':
  loss = tf.reduce_mean(tf.abs(Dec_x - x))
elif opt == 'l2':
  loss = tf.reduce_mean(tf.square(Dec_x - x))
else:
  raise ValueError()

opt = tf.train.AdamOptimizer()
step = tf.train.get_or_create_global_step()
train = opt.minimize(loss, var_list=opt_vars, global_step=step)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(nsteps):
    sess.run(train)

  _Dec_x = sess.run(Dec_x)

  _Dec_x *= 32767.
  _Dec_x = np.clip(_Dec_x, -32768., 32767.)
  _Dec_x = _Dec_x.astype(np.int16)

  wavwrite('overfit_{}.wav'.format(nsteps), fs, _Dec_x)