import tensorflow as tf


def dense_layer(x, out_dim, stddev=0.02, dtype=tf.float32):
  _, in_dim = x.get_shape().as_list()

  W = tf.get_variable('W', [in_dim, out_dim], dtype=dtype,
      initializer=tf.initializers.random_normal(stddev=stddev))
  x = tf.matmul(x, W)
  b = tf.get_variable('b', [out_dim], dtype=dtype,
      initializer=tf.initializers.zeros())
  x = tf.nn.bias_add(x, b)

  return x


def conv2d_transpose_layer(
    x,
    out_shape_or_num_ch=None,
    kernel_h=5,
    kernel_w=5,
    stride_h=2,
    stride_w=2,
    stddev=0.02,
    dtype=tf.float32):
  try:
    batch_size = int(x.get_shape()[0])
  except:
    batch_size = tf.shape(x)[0]

  _, in_h, in_w, in_ch = x.get_shape().as_list()

  try:
    out_h, out_w, out_ch = out_shape_or_num_ch
  except TypeError:
    out_h = in_h * stride_h
    out_w = in_w * stride_w
    out_ch = out_shape_or_num_ch

  W = tf.get_variable('W', [kernel_h, kernel_w, out_ch, in_ch], dtype=dtype,
      initializer=tf.initializers.random_normal(stddev=stddev))
  x = tf.nn.conv2d_transpose(
      x, W,
      output_shape=[batch_size, out_h, out_w, out_ch],
      strides=[1, stride_h, stride_w, 1],
      padding='SAME')

  b = tf.get_variable('b', [out_ch], dtype=dtype,
      initializer=tf.initializers.zeros())
  x = tf.nn.bias_add(x, b)

  return x


def conv2d_layer(
    x,
    out_ch=None,
    kernel_h=5,
    kernel_w=5,
    stride_h=2,
    stride_w=2,
    stddev=0.02,
    dtype=tf.float32):
  try:
    batch_size = int(x.get_shape()[0])
  except:
    batch_size = tf.shape(x)[0]

  _, in_h, in_w, in_ch = x.get_shape().as_list()

  W = tf.get_variable('W', [kernel_h, kernel_w, in_ch, out_ch], dtype=dtype,
      initializer=tf.initializers.random_normal(stddev=stddev))
  x = tf.nn.conv2d(x, W, strides=[1, stride_h, stride_w, 1], padding='SAME')

  b = tf.get_variable('b', [out_ch], dtype=dtype,
      initializer=tf.initializers.zeros())
  x = tf.nn.bias_add(x, b)

  return x


class MelspecGANGenerator(object):
  def __init__(
      self,
      dim=64,
      kernel_len=5,
      batchnorm=True,
      nonlin=tf.nn.tanh):
    self.dim = dim
    self.kernel_len = kernel_len
    self.stride = 2
    self.batchnorm = batchnorm
    self.nonlin = nonlin

  def __call__(self, z, training=False):
    # input z is [batch_size, 100]
    # returns [batch_size, 64, 80, 1]

    std_conv2d_transpose_layer = lambda x, out_shape: conv2d_transpose_layer(
        x,
        out_shape,
        kernel_h=self.kernel_len,
        kernel_w=self.kernel_len,
        stride_h=self.stride,
        stride_w=self.stride)

    if self.batchnorm:
      batchnorm = lambda x: tf.layers.batch_normalization(x, training=training)
    else:
      batchnorm = lambda x: x

    # project z to [batch_size, 4, 4, dim * 8]
    with tf.variable_scope('z_proj'):
      x = dense_layer(z, 4 * 4 * self.dim * 8)
    x = tf.reshape(x, [-1, 4, 4, self.dim * 8])
    x = batchnorm(x)
    x = tf.nn.relu(x)

    # [4, 4, 8d] -> [8, 8, 4d]
    with tf.variable_scope('upconv_1'):
      x = std_conv2d_transpose_layer(x, [8, 8, self.dim * 4])
    x = batchnorm(x)
    x = tf.nn.relu(x)

    # [8, 8, 4d] -> [16, 16, 2d]
    with tf.variable_scope('upconv_2'):
      x = std_conv2d_transpose_layer(x, [16, 16, self.dim * 2])
    x = batchnorm(x)
    x = tf.nn.relu(x)

    # [16, 16, 2d] -> [32, 32, d]
    with tf.variable_scope('upconv_3'):
      x = std_conv2d_transpose_layer(x, [32, 32, self.dim])
    x = batchnorm(x)
    x = tf.nn.relu(x)

    # [32, 32, d] -> [64, 64, d]
    with tf.variable_scope('upconv_4'):
      x = std_conv2d_transpose_layer(x, [64, 64, self.dim])
    x = batchnorm(x)
    x = tf.nn.relu(x)

    # [64, 64, d] -> [64, 80, 1]
    with tf.variable_scope('upconv_5'):
      x = conv2d_transpose_layer(x, [64, 80, 1],
          kernel_h=1, kernel_w=self.kernel_len,
          stride_h=1, stride_w=self.stride)

    x = self.nonlin(x)

    if training and self.batchnorm:
      update_ops = tf.get_collection(
          tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
      assert len(update_ops) == 10
      with tf.control_dependencies(update_ops):
        x = tf.identity(x)

    return x


class MelspecGANDiscriminator(object):
  def __init__(
      self,
      dim=64,
      kernel_len=5,
      batchnorm=True,
      nonlin=tf.nn.tanh):
    self.dim = dim
    self.kernel_len = kernel_len
    self.stride = 2
    self.batchnorm = batchnorm
    self.nonlin = nonlin

  def __call__(self, x, training=False):
    # input x is [batch_size, 64, 80, 1]
    # returns [batch_size, 1]
    std_conv2d_layer = lambda x, out_ch: conv2d_layer(
        x,
        out_ch,
        kernel_h=self.kernel_len,
        kernel_w=self.kernel_len,
        stride_h=self.stride,
        stride_w=self.stride)

    if self.batchnorm:
      batchnorm = lambda x: tf.layers.batch_normalization(x, training=training)
    else:
      batchnorm = lambda x: x

    # [64, 80, 1] -> [32, 40, d]
    with tf.variable_scope('conv_0'):
      x = std_conv2d_layer(x, self.dim)
    x = tf.nn.leaky_relu(x, 0.2)

    # [32, 40, d] -> [16, 20, 2d]
    with tf.variable_scope('conv_1'):
      x = std_conv2d_layer(x, self.dim * 2)
    x = batchnorm(x)
    x = tf.nn.leaky_relu(x, 0.2)

    # [16, 20, 2d] -> [8, 10, 4d]
    with tf.variable_scope('conv_2'):
      x = std_conv2d_layer(x, self.dim * 4)
    x = batchnorm(x)
    x = tf.nn.leaky_relu(x, 0.2)

    # [8, 10, 4d] -> [4, 5, 8d]
    with tf.variable_scope('conv_3'):
      x = std_conv2d_layer(x, self.dim * 8)
    x = batchnorm(x)
    x = tf.nn.leaky_relu(x, 0.2)

    x = tf.reshape(x, [-1, 4 * 5 * self.dim * 8])
    with tf.variable_scope('out'):
      x = dense_layer(x, 1)[:, 0]

    # TODO
    """
    if training and self.batchnorm:
      update_ops = tf.get_collection(
          tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
      assert len(update_ops) == 10
      with tf.control_dependencies(update_ops):
        x = tf.identity(x)
    """

    return x
