import tensorflow as tf


def build_nsynth_wavenet_encoder(
    input_wave,
    num_stages=10,
    num_layers=30,
    filter_length=3,
    width=128,
    hop_length=512,
    bottleneck_width=16):
  x_scaled = input_wave
  ae_num_stages = num_stages
  ae_num_layers = num_layers
  ae_filter_length = filter_length
  ae_width = width
  ae_hop_length = hop_length
  ae_bottleneck_width = bottleneck_width

  ###
  # The Non-Causal Temporal Encoder.
  ###
  en = masked_conv1d(
      x_scaled,
      causal=False,
      num_filters=ae_width,
      filter_length=ae_filter_length,
      name='ae_startconv')

  for num_layer in range(ae_num_layers):
    dilation = 2**(num_layer % ae_num_stages)
    d = tf.nn.relu(en)
    d = masked_conv1d(
	d,
	causal=False,
	num_filters=ae_width,
	filter_length=ae_filter_length,
	dilation=dilation,
	name='ae_dilatedconv_%d' % (num_layer + 1))
    d = tf.nn.relu(d)
    en += masked_conv1d(
	d,
	num_filters=ae_width,
	filter_length=1,
	name='ae_res_%d' % (num_layer + 1))

  en = masked_conv1d(
      en,
      num_filters=ae_bottleneck_width,
      filter_length=1,
      name='ae_bottleneck')
  en = masked.pool1d(en, ae_hop_length, name='ae_pool', mode='avg')

  return en


def build_nsynth_wavenet_decoder(
    input_wave,
    conditioning_info,
    output_width=256,
    num_stages=10,
    num_layers=30,
    filter_length=3,
    width=256,
    skip_width=128):
  l = input_wave # Make sure this is shifted if we're doing autoregression!
  en = conditioning_info

  ###
  # The WaveNet Decoder.
  ###
  l = masked_conv1d(
      l, num_filters=width, filter_length=filter_length, name='startconv')

  # Set up skip connections.
  s = masked_conv1d(
      l, num_filters=skip_width, filter_length=1, name='skip_start')

  # Residual blocks with skip connections.
  for i in range(num_layers):
    dilation = 2**(i % num_stages)
    d = masked_conv1d(
	l,
	num_filters=2 * width,
	filter_length=filter_length,
	dilation=dilation,
	name='dilatedconv_%d' % (i + 1))
    d = self_condition(d,
			masked_conv1d(
			    en,
			    num_filters=2 * width,
			    filter_length=1,
			    name='cond_map_%d' % (i + 1)))

    assert d.get_shape().as_list()[2] % 2 == 0
    m = d.get_shape().as_list()[2] // 2
    d_sigmoid = tf.sigmoid(d[:, :, :m])
    d_tanh = tf.tanh(d[:, :, m:])
    d = d_sigmoid * d_tanh

    l += masked_conv1d(
	d, num_filters=width, filter_length=1, name='res_%d' % (i + 1))
    s += masked_conv1d(
	d, num_filters=skip_width, filter_length=1, name='skip_%d' % (i + 1))

  s = tf.nn.relu(s)
  s = masked_conv1d(s, num_filters=skip_width, filter_length=1, name='out1')
  s = self_condition(s,
		      masked_conv1d(
			  en,
			  num_filters=skip_width,
			  filter_length=1,
			  name='cond_map_out1'))
  s = tf.nn.relu(s)

  ###
  # Compute the logits and get the loss.
  ###
  logits = masked_conv1d(s, num_filters=output_width, filter_length=1, name='logits')

  return logits


def self_condition(x, encoding):
  """Condition the input on the encoding.
  Args:
    x: The [mb, length, channels] float tensor input.
    encoding: The [mb, encoding_length, channels] float tensor encoding.
  Returns:
    The output after broadcasting the encoding to x's shape and adding them.
  """
  mb, length, channels = x.get_shape().as_list()
  enc_mb, enc_length, enc_channels = encoding.get_shape().as_list()
  assert enc_mb == mb
  assert enc_channels == channels

  encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
  x = tf.reshape(x, [mb, enc_length, -1, channels])
  x += encoding
  x = tf.reshape(x, [mb, length, channels])
  x.set_shape([mb, length, channels])
  return x


def masked_conv1d(x,
           num_filters,
           filter_length,
           name,
           dilation=1,
           causal=True,
           kernel_initializer=tf.uniform_unit_scaling_initializer(1.0),
           biases_initializer=tf.constant_initializer(0.0)):
  """Fast 1D convolution that supports causal padding and dilation.
  Args:
    x: The [mb, time, channels] float tensor that we convolve.
    num_filters: The number of filter maps in the convolution.
    filter_length: The integer length of the filter.
    name: The name of the scope for the variables.
    dilation: The amount of dilation.
    causal: Whether or not this is a causal convolution.
    kernel_initializer: The kernel initialization function.
    biases_initializer: The biases initialization function.
  Returns:
    y: The output of the 1D convolution.
  """
  batch_size, length, num_input_channels = x.get_shape().as_list()
  assert length % dilation == 0

  kernel_shape = [1, filter_length, num_input_channels, num_filters]
  strides = [1, 1, 1, 1]
  biases_shape = [num_filters]
  padding = 'VALID' if causal else 'SAME'

  with tf.variable_scope(name):
    weights = tf.get_variable(
        'W', shape=kernel_shape, initializer=kernel_initializer)
    biases = tf.get_variable(
        'biases', shape=biases_shape, initializer=biases_initializer)

  x_ttb = time_to_batch(x, dilation)
  if filter_length > 1 and causal:
    x_ttb = tf.pad(x_ttb, [[0, 0], [filter_length - 1, 0], [0, 0]])

  x_ttb_shape = x_ttb.get_shape().as_list()
  x_4d = tf.reshape(x_ttb, [x_ttb_shape[0], 1,
                            x_ttb_shape[1], num_input_channels])
  y = tf.nn.conv2d(x_4d, weights, strides, padding=padding)
  y = tf.nn.bias_add(y, biases)
  y_shape = y.get_shape().as_list()
  y = tf.reshape(y, [y_shape[0], y_shape[2], num_filters])
  y = batch_to_time(y, dilation)
  y.set_shape([batch_size, length, num_filters])
  return y


def time_to_batch(x, block_size):
  """Splits time dimension (i.e. dimension 1) of `x` into batches.
  Within each batch element, the `k*block_size` time steps are transposed,
  so that the `k` time steps in each output batch element are offset by
  `block_size` from each other.
  The number of input time steps must be a multiple of `block_size`.
  Args:
    x: Tensor of shape [nb, k*block_size, n] for some natural number k.
    block_size: number of time steps (i.e. size of dimension 1) in the output
      tensor.
  Returns:
    Tensor of shape [nb*block_size, k, n]
  """
  shape = x.get_shape().as_list()
  y = tf.reshape(x, [
      shape[0], shape[1] // block_size, block_size, shape[2]
  ])
  y = tf.transpose(y, [0, 2, 1, 3])
  y = tf.reshape(y, [
      shape[0] * block_size, shape[1] // block_size, shape[2]
  ])
  y.set_shape([
      mul_or_none(shape[0], block_size), mul_or_none(shape[1], 1. / block_size),
      shape[2]
  ])
  return y


def batch_to_time(x, block_size):
  """Inverse of `time_to_batch(x, block_size)`.
  Args:
    x: Tensor of shape [nb*block_size, k, n] for some natural number k.
    block_size: number of time steps (i.e. size of dimension 1) in the output
      tensor.
  Returns:
    Tensor of shape [nb, k*block_size, n].
  """
  shape = x.get_shape().as_list()
  y = tf.reshape(x, [shape[0] // block_size, block_size, shape[1], shape[2]])
  y = tf.transpose(y, [0, 2, 1, 3])
  y = tf.reshape(y, [shape[0] // block_size, shape[1] * block_size, shape[2]])
  y.set_shape([mul_or_none(shape[0], 1. / block_size),
               mul_or_none(shape[1], block_size),
               shape[2]])
  return y


def mul_or_none(a, b):
  """Return the element wise multiplicative of the inputs.
  If either input is None, we return None.
  Args:
    a: A tensor input.
    b: Another tensor input with the same type as a.
  Returns:
    None if either input is None. Otherwise returns a * b.
  """
  if a is None or b is None:
    return None
  return a * b
