import tensorflow as tf


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  if rad == 0:
    return x

  b, x_len, _, nch = x.get_shape().as_list()

  phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, 1, nch])

  return x


"""
  Input: [None, slice_len, nch]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(
    x,
    spec=None,
    patched=False,
    kernel_len=25,
    dim=64,
    phaseshuffle_rad=0):
  conv1d = lambda x, n: tf.layers.conv2d(
      x,
      n,
      (kernel_len, 1),
      strides=(4, 1),
      padding='same')

  conv1x1d = lambda x, n: tf.layers.conv2d(
      x,
      n,
      (1, 1),
      strides=(1, 1),
      padding='same')

  batch_size = tf.shape(x)[0]
  slice_len = int(x.get_shape()[1])

  if spec is not None:
    nmels = int(spec.get_shape()[2])
    spec = tf.image.resize_nearest_neighbor(spec, [slice_len, nmels])

    spec = tf.transpose(spec, [0, 1, 3, 2])

    x = tf.concat([x, spec], axis=3)

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  # Layer 0
  # [16384, 1] -> [4096, 64]
  output = x
  with tf.variable_scope('downconv_0'):
    output = conv1d(output, dim)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('downconv_1'):
    output = conv1d(output, dim * 2)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('downconv_2'):
    output = conv1d(output, dim * 4)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('downconv_3'):
    output = conv1d(output, dim * 8)
  output = lrelu(output)

  if patched:
    with tf.variable_scope('output'):
      output = conv1x1d(output, 1)
  else:
    output = phaseshuffle(output)
    with tf.variable_scope('downconv_4'):
      output = conv1d(output, dim * 16)
    output = lrelu(output)

    output = tf.reshape(output, [batch_size, -1])

    with tf.variable_scope('output'):
      output = tf.layers.dense(output, 1)

    output = output[:, :, tf.newaxis, tf.newaxis]

  return output
