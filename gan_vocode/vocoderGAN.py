import tensorflow as tf

from model import Model, Modes

class VocoderGAN(Model):
  audio_fs = 22050
  subseq_len = 64
  zdim = 100
  dim = 64
  stride = 4
  kernel_len = 25
  phaseshuffle_rad = 0
  wgangp_lambda = 10
  wgangp_nupdates = 5
  gen_nonlin = 'relu'
  recon_loss_type = 'wav' # wav, spec
  recon_objective = 'l1' # l1, l2
  recon_regularizer = 1. 
  train_batch_size = 64

  def build_generator(self, x_spec, z):
    x_spec = tf.transpose(x_spec, [0, 1, 3, 2])
    conv1d_transpose = lambda x, n: tf.layers.conv2d_transpose(
        x,
        n,
        (self.kernel_len, 1),
        strides=(self.stride, 1),
        padding='same')
    

    conv1x1d_transpose = lambda x, n: tf.layers.conv2d_transpose(
        x,
        n,
        (1, 1),
        strides=(1, 1),
        padding='same')

    if self.gen_nonlin == 'relu':
      nonlin = lambda x: tf.nn.relu(x)
    elif self.gen_nonlin == 'linear':
      nonlin = lambda x: x
    else:
      raise ValueError()

    # FC and reshape for convolution
    # [100] -> [64, 64]
    x = z
    batch_size = tf.shape(z)[0]
    with tf.variable_scope('z_project'):
      x = tf.layers.dense(x, 8 * 8 * self.dim)
      x = tf.reshape(x, [batch_size, 64, 1, self.dim])

    x = tf.nn.tanh(x)
    x = tf.concat([x, x_spec], axis=3)

    # [64, 128] -> [64, 512]
    with tf.variable_scope('upconv_1x1'):
      x = conv1x1d_transpose(x, self.dim * 8)
    x = nonlin(x)
    
    
    # [64, 512] -> [256, 256]
    with tf.variable_scope('upconv_1'):
      x = conv1d_transpose(x, self.dim * 4)
    x = nonlin(x)

    # Layer 2
    # [256, 256] -> [1024, 128]
    with tf.variable_scope('upconv_2'):
      x = conv1d_transpose(x, self.dim * 2)
    x = nonlin(x)

    # Layer 3
    # [1024, 128] -> [4096, 64]
    with tf.variable_scope('upconv_3'):
      x = conv1d_transpose(x, self.dim)
    x = nonlin(x)

    # Layer 4
    # [4096, 64] -> [16384, 1]
    with tf.variable_scope('upconv_4'):
      x = conv1d_transpose(x, 1)
    x = tf.nn.tanh(x)

    return x

  def build_discriminator(self, x):
    conv1d = lambda x, n: tf.layers.conv2d(
        x,
        n,
        (self.kernel_len, 1),
        strides=(self.stride, 1),
        padding='same')

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

    batch_size = tf.shape(x)[0]

    phaseshuffle = lambda x: apply_phaseshuffle(x, self.phaseshuffle_rad)

    # Layer 0
    # [16384, 1] -> [4096, 64]
    output = x
    with tf.variable_scope('downconv_0'):
      output = conv1d(output, self.dim)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 1
    # [4096, 64] -> [1024, 128]
    with tf.variable_scope('downconv_1'):
      output = conv1d(output, self.dim * 2)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 2
    # [1024, 128] -> [256, 256]
    with tf.variable_scope('downconv_2'):
      output = conv1d(output, self.dim * 4)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 3
    # [256, 256] -> [64, 512]
    with tf.variable_scope('downconv_3'):
      output = conv1d(output, self.dim * 8)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 4
    # [64, 512] -> [16, 1024]
    with tf.variable_scope('downconv_4'):
      output = conv1d(output, self.dim * 16)
    output = lrelu(output)

    # Flatten
    output = tf.reshape(output, [batch_size, 4 * 4 * self.dim * 16])

    # Connect to single logit
    with tf.variable_scope('output'):
      output = tf.layers.dense(output, 1)[:, 0]

    return output 

  def __call__(self, x_wav, x_spec):
    try:
      batch_size = int(x_wav.get_shape()[0])
    except:
      batch_size = tf.shape(x_wav)[0]

    # Noise var
    z = tf.random_uniform([batch_size, self.zdim], -1, 1, dtype=tf.float32)

    # Generator
    with tf.variable_scope('G'):
      G_z = self.build_generator(x_spec, z)
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

    # Discriminators
    with tf.name_scope('D_x'), tf.variable_scope('D'):
      D_x = self.build_discriminator(x_wav)
    with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
      D_G_z = self.build_discriminator(G_z)
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

    self.wav_l1 = wav_l1 = tf.reduce_mean(tf.abs(x_wav - G_z))
    self.wav_l2 = wav_l2 = tf.reduce_mean(tf.square(x_wav - G_z))

    gen_spec = tf.contrib.signal.stft(G_z[:,:,0,0], 128, 256, pad_end=True)
    gen_spec_mag = tf.abs(gen_spec)

    target_spec = tf.contrib.signal.stft(x_wav[:,:,0,0], 128, 256, pad_end=True)
    target_spec_mag = tf.abs(target_spec)

    self.spec_l1 = spec_l1 = tf.reduce_mean(tf.abs(target_spec_mag - gen_spec_mag))
    self.spec_l2 = spec_l2 = tf.reduce_mean(tf.square(target_spec_mag - gen_spec_mag))

    
    if self.recon_objective == 'l1':
      if self.recon_loss_type == 'wav':
        self.recon_loss = wav_l1
      elif self.recon_loss_type == 'spec':
        self.recon_loss = spec_l1
    elif self.recon_objective == 'l2':
      if self.recon_loss_type == 'wav':
        self.recon_loss = wav_l2
      elif self.recon_loss_type == 'spec':
        self.recon_loss = spec_l2

    # WGAN-GP loss
    G_loss = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

    alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    differences = G_z - x_wav
    interpolates = x_wav + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
      D_interp = self.build_discriminator(interpolates)

    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    D_loss += self.wgangp_lambda * gradient_penalty

    G_loss += self.recon_regularizer * self.recon_loss
    # Optimizers
    G_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9)

    D_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9)

    # Training ops
    self.G_train_op = G_opt.minimize(G_loss, var_list=G_vars,
	global_step=tf.train.get_or_create_global_step())
    self.D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

    # Summarize
    tf.summary.audio('x_wav', x_wav[:, :, 0, :], self.audio_fs)
    tf.summary.audio('G_z', G_z[:, :, 0, :], self.audio_fs)
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('Recon_loss', self.recon_loss)
    tf.summary.scalar('D_loss', D_loss)

  def train_loop(self, sess):
    for i in range(self.wgangp_nupdates):
      sess.run(self.D_train_op)

    sess.run(self.G_train_op)
