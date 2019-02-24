import numpy as np
import tensorflow as tf

import advoc.spectral as spectral


def feats_to_uint8_img(x):
  x = tf.image.rot90(x)
  x *= 255.
  x = tf.clip_by_value(x, 0., 255.)
  x = tf.cast(x, tf.uint8)
  return x


def feats_to_approx_audio(x, fs, waveform_len, n=None):
  if n is not None:
    x = x[:n]

  inv_closure = lambda _x: spectral.r9y9_melspec_to_waveform(
      _x.astype(np.float64), fs=fs, waveform_len=waveform_len)

  inv_pyfn = lambda x_item: tf.py_func(
      inv_closure, [x_item], tf.float32, stateful=False)

  return tf.map_fn(inv_pyfn, x)
