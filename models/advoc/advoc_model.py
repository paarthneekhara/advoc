import tensorflow as tf

from model import Model, Modes
import advoc.spectral
import lws
from spectral_util import SpectralUtil
import numpy as np
EPS = 1e-12

class Advoc(Model):
  audio_fs = 22050
  subseq_len = 256
  n_mels = 80
  ngf = 64
  ndf = 64
  gan_weight = 1. 
  l1_weight = 100.
  train_batch_size = 32
  eval_batch_size = 1
  use_adversarial = True #Train as a GAN or not
  separable_conv = False
  use_batchnorm = True
  generator_type = "pix2pix" #pix2pix, linear, linear+pix2pix


  def _discrim_conv(self, x, out_channels, stride):
    padded_input = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, 
      out_channels, 
      kernel_size=4, 
      strides=(stride, stride), 
      padding="valid", 
      kernel_initializer=tf.random_normal_initializer(0, 0.02))

  def _gen_conv(self, x, out_channels, strides = (2, 2)):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if self.separable_conv:
      return tf.layers.separable_conv2d(x, 
        out_channels, 
        kernel_size=4, 
        strides=strides, 
        padding="same", 
        depthwise_initializer=initializer, 
        pointwise_initializer=initializer)
    else:
      return tf.layers.conv2d(x, 
        out_channels, 
        kernel_size=4, 
        strides=strides, 
        padding="same", 
        kernel_initializer=initializer)

  def _gen_deconv(self, x, out_channels, strides = (2, 2)):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if self.separable_conv:
        _b, h, w, _c = x.shape
        resized_input = tf.image.resize_images(x, [h * strides[0], w * strides[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, 
          out_channels, kernel_size=4, 
          strides=(1, 1), padding="same", 
          depthwise_initializer=initializer, 
          pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(x, 
          out_channels, kernel_size=4, 
          strides=strides, 
          padding="same", 
          kernel_initializer=initializer)

  def build_linear_generator(self, x):
    gen = tf.layers.dense(x[:,:,:,0], 513)
    return tf.expand_dims(gen, -1)

  def build_generator(self, x):
    
    if self.use_batchnorm:
      batchnorm = lambda x: tf.layers.batch_normalization(x, 
        axis=3, 
        epsilon=1e-5, 
        momentum=0.1, 
        training = True)
    else:
      batchnorm = lambda x: x

    def lrelu(inputs, alpha=0.2):
      return tf.maximum(alpha * inputs, inputs)

    layers = []
    n_time = self.subseq_len
    with tf.variable_scope("encoder_1"):
      output = self._gen_conv(x[:,:,:,:], self.ngf)
      n_time /= 2
      layers.append(output)

    layer_specs = [
      self.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
      self.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
      self.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
      self.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
      self.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
      self.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
      self.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    n_stride1_layers = 0 # number of decoder layer
    for out_channels in layer_specs:
      with tf.variable_scope("encoder_{}".format(len(layers) + 1)):
        rectified = lrelu(layers[-1], 0.2)
        # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
        if n_time > 1:
          convolved = self._gen_conv(rectified, out_channels, strides = (2, 2))
          n_time /= 2
        else:
          n_stride1_layers += 1
          convolved = self._gen_conv(rectified, out_channels, strides = (1, 2))
        output = batchnorm(convolved)
        layers.append(output)

    layer_specs = [
        (self.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (self.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (self.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (self.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (self.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (self.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (self.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
      skip_layer = num_encoder_layers - decoder_layer - 1
      with tf.variable_scope("decoder_{}".format(skip_layer + 1)):
        if decoder_layer == 0:
          input = layers[-1]
        else:
          input = tf.concat([layers[-1][:,:,:-1,:], layers[skip_layer]], axis=3)
        rectified = tf.nn.relu(input)
        if decoder_layer < n_stride1_layers:
          output = self._gen_deconv(rectified, out_channels, strides = (1, 2))
        else:
          output = self._gen_deconv(rectified, out_channels, strides = (2, 2))
        output = batchnorm(output)
        if dropout > 0.0:
          if self.mode == Modes.TRAIN:
            output = tf.nn.dropout(output, keep_prob= 1 - dropout)
          else:
            # use dropout in inference as well
            output = tf.nn.dropout(output, keep_prob= 1 - dropout)
        layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1][:,:,:-1,:], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = self._gen_deconv(rectified, 1)[:,:,:-1,:]
        # output = tf.tanh(output)
        layers.append(output)

    # some batch norm thing that i dont understand
    if self.mode == Modes.TRAIN and self.use_batchnorm:
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
      with tf.control_dependencies(update_ops):
        output = tf.identity(output)

    return layers[-1]

  def build_discriminator(self, discrim_inputs, discrim_targets):
    def lrelu(inputs, alpha=0.2):
      return tf.maximum(alpha * inputs, inputs)

    if self.use_batchnorm:
      batchnorm = lambda x: tf.layers.batch_normalization(x, 
        axis=3, 
        epsilon=1e-5, 
        momentum=0.1,
        training = True)
    else:
      batchnorm = lambda x: x

    n_layers = 3
    layers = []
    
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)
    with tf.variable_scope("layer_1"):
      convolved = self._discrim_conv(input, self.ndf, stride=2)
      rectified = lrelu(convolved, 0.2)
      layers.append(rectified)

    for i in range(n_layers):
      with tf.variable_scope("layer_{}".format(len(layers) + 1)):
        out_channels = self.ndf * min(2**(i+1), 8)
        stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
        convolved = self._discrim_conv(layers[-1], out_channels, stride=stride)
        normalized = batchnorm(convolved)
        rectified = lrelu(normalized, 0.2)
        layers.append(rectified)

    with tf.variable_scope("layer_{}".format(len(layers) + 1)):
      convolved = self._discrim_conv(rectified, out_channels=1, stride=1)
      output = tf.sigmoid(convolved)
      layers.append(output)

    return layers[-1]

  def __call__(self, x, target, x_wav, x_mel_spec):
    
    self.spectral = SpectralUtil(n_mels = self.n_mels, fs = self.audio_fs)

    try:
      batch_size = int(x.get_shape()[0])
    except:
      batch_size = tf.shape(x)[0]

    with tf.variable_scope("generator"):
      if self.generator_type == "pix2pix":
        gen_mag_spec = self.build_generator(x)
      elif self.generator_type == "linear":
        gen_mag_spec = self.build_linear_generator(x)
      elif self.generator_type == "linear+pix2pix":
        temp_spec = self.build_linear_generator(x_mel_spec)
        gen_mag_spec = self.build_linear_generator(temp_spec)
      elif self.generator_type == "interp+pix2pix":
        _temp_spec = tf.image.resize_images(x_mel_spec, 
          [self.subseq_len, 513])
        gen_mag_spec = self.build_linear_generator(_temp_spec)
      else:
        raise NotImplementedError()

    with tf.name_scope("real_discriminator"):
      with tf.variable_scope("discriminator"):
        predict_real = self.build_discriminator(x, target)

    with tf.name_scope("fake_discriminator"):
      with tf.variable_scope("discriminator", reuse=True):
        predict_fake = self.build_discriminator(x, gen_mag_spec)

    discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
    gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
    gen_loss_L1 = tf.reduce_mean(tf.abs(target - gen_mag_spec))
    
    if self.gan_weight > 0:
      gen_loss = gen_loss_GAN * self.gan_weight + gen_loss_L1 * self.l1_weight
    else:
      gen_loss = gen_loss_L1 * self.l1_weight

    self.D_vars = D_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
    self.G_vars = G_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

    D_opt = tf.train.AdamOptimizer(0.0002, 0.5)
    G_opt = tf.train.AdamOptimizer(0.0002, 0.5)
    
    self.step = step = tf.train.get_or_create_global_step()
    self.G_train_op = G_opt.minimize(gen_loss, var_list=G_vars,
  global_step=self.step)

    self.D_train_op = D_opt.minimize(discrim_loss, var_list=D_vars)

    input_audio = tf.py_func( self.spectral.audio_from_mag_spec, [x[0]], tf.float32, stateful=False)
    target_audio = tf.py_func( self.spectral.audio_from_mag_spec, [target[0]], tf.float32, stateful=False)
    gen_audio = tf.py_func( self.spectral.audio_from_mag_spec, [gen_mag_spec[0]], tf.float32, stateful=False)

    input_audio = tf.reshape(input_audio, [1, -1, 1, 1] )
    target_audio = tf.reshape(target_audio, [1, -1, 1, 1] )
    gen_audio = tf.reshape(gen_audio, [1, -1, 1, 1] )
    
    
    tf.summary.audio('input_audio', input_audio[:, :, 0, :], self.audio_fs)
    tf.summary.audio('target_audio', target_audio[:, :, 0, :], self.audio_fs)
    tf.summary.audio('target_x_wav', x_wav[:, :, 0, :], self.audio_fs)
    tf.summary.audio('gen_audio', gen_audio[:, :, 0, :], self.audio_fs)
    tf.summary.scalar('gen_loss_total', gen_loss)
    tf.summary.scalar('gen_loss_L1', gen_loss_L1)
    tf.summary.scalar('gen_loss_GAN', gen_loss_GAN)
    tf.summary.scalar('disc_loss', discrim_loss)

    #image summaries
    tf.summary.image('input_melspec', tf.image.rot90(x_mel_spec))
    tf.summary.image('input_magspec', tf.image.rot90(x))
    tf.summary.image('generated_magspec', tf.image.rot90(gen_mag_spec))
    tf.summary.image('target_magspec', tf.image.rot90(target))
    


  def train_loop(self, sess):
    if self.gan_weight > 0:
      sess.run(self.D_train_op)
    _, _step = sess.run([self.G_train_op, self.step])
    return _step
    

