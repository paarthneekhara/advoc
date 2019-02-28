import numpy as np
import tensorflow as tf

import advoc.spectral


def best_shape(t, axis=None):
  """Gets static shape if available, otherwise dynamic.

  Args:
    t: Tensor in question.
    axis: None if requesting entire shape, otherwise the axis in question.

  Returns:
    Python list containing (possibly a mixture of) ints or tf.Tensor.
  """
  if axis is None:
    ndims = t.get_shape().ndims
    if ndims is None:
      raise ValueError('Cannot run on tensor with dynamic ndims')
    dims = []
    for i in range(ndims):
      try:
        dim = int(t.get_shape()[i])
      except:
        dim = tf.shape(t)[i]
      dims.append(dim)
    return dims
  else:
    try:
      dim = int(t.get_shape()[axis])
    except:
      dim = tf.shape(t)[i]
    return dim


def r9y9_melspec_norm(x):
  return (x * 2.) - 1.


def r9y9_melspec_denorm(x):
  return (x + 1.) * 0.5


def r9y9_melspec_to_uint8_img(x):
  x = tf.image.rot90(x)
  x *= 255.
  x = tf.clip_by_value(x, 0., 255.)
  x = tf.cast(x, tf.uint8)
  return x


def r9y9_melspec_to_approx_audio(x, fs, waveform_len, n=None):
  if n is not None:
    x = x[:n]

  inv_closure = lambda _x: spectral.r9y9_melspec_to_waveform(
      _x.astype(np.float64), fs=fs, waveform_len=waveform_len)

  inv_pyfn = lambda x_item: tf.py_func(
      inv_closure, [x_item], tf.float32, stateful=False)

  return tf.map_fn(inv_pyfn, x)
