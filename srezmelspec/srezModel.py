import tensorflow as tf

from model import Model, Modes
import advoc.spectral


class SrezMelSpec(Model):
  audio_fs = 22050
  subseq_len = 256
  zdim = 100
  ngf = 64
  stride = 4
  kernel_len = 25
  phaseshuffle_rad = 0
  wgangp_lambda = 10
  wgangp_nupdates = 5
  gen_nonlin = 'relu'
  gan_strategy = 'wgangp'
  
  recon_objective = 'l2' # l1, l2
  discriminator_type = "patched" # patched, regular
  recon_regularizer = 1. 
  train_batch_size = 32
  eval_batch_size = 1
  use_adversarial = True #Train as a GAN or not
  separable_conv = False
  use_batchnorm = True


  def _gen_conv(self, x, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if self.separable_conv:
      return tf.layers.separable_conv2d(x, 
        out_channels, 
        kernel_size=4, 
        strides=(2, 2), 
        padding="same", 
        depthwise_initializer=initializer, 
        pointwise_initializer=initializer)
    else:
      return tf.layers.conv2d(x, 
        out_channels, 
        kernel_size=4, 
        strides=(2, 2), 
        padding="same", 
        kernel_initializer=initializer)

  def _gen_deconv(self, x, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if self.separable_conv:
        _b, h, w, _c = x.shape
        resized_input = tf.image.resize_images(x, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, 
          out_channels, kernel_size=4, 
          strides=(1, 1), padding="same", 
          depthwise_initializer=initializer, 
          pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(x, 
          out_channels, kernel_size=4, 
          strides=(2, 2), 
          padding="same", 
          kernel_initializer=initializer)

  def build_generator(self, x):
    
    if self.use_batchnorm:
      batchnorm = lambda x: tf.layers.batch_normalization(x, 
        axis=3, 
        epsilon=1e-5, 
        momentum=0.1, 
        training = self.mode == Modes.TRAIN)
    else:
      batchnorm = lambda x: x

    def lrelu(inputs, alpha=0.2):
      return tf.maximum(alpha * inputs, inputs)

    layers = []
    with tf.variable_scope("encoder_1"):
      output = self._gen_conv(x[:,:,:,:], self.ngf)
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
    for out_channels in layer_specs:
      with tf.variable_scope("encoder_{}".format(len(layers) + 1)):
        rectified = lrelu(layers[-1], 0.2)
        # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
        convolved = self._gen_conv(rectified, out_channels)
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
        output = self._gen_deconv(rectified, out_channels)
        output = batchnorm(output)
        if dropout > 0.0:
          output = tf.nn.dropout(output, keep_prob=1 - dropout)
        layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1][:,:,:-1,:], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = self._gen_deconv(rectified, 1)
        layers.append(output)

    return layers[-1]

  def build_discriminator(self, x):
    pass


  def __call__(self, x, target):
    try:
      batch_size = int(x.get_shape()[0])
    except:
      batch_size = tf.shape(x)[0]

    self.build_generator(x)

    


  def train_loop(self, sess):
    if self.use_adversarial:
      # Run Discriminator update only in adversarial scenario
      num_disc_updates = self.wgangp_nupdates if self.gan_strategy == 'wgangp' else 1
      for i in range(num_disc_updates):
        sess.run(self.D_train_op)
    sess.run(self.G_train_op)
    

