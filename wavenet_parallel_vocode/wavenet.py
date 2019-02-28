import numpy as np
import tensorflow as tf

import advoc.util
import advoc.spectral

from model import Model, Modes
from wavenet_util import *


class WavenetVocoder(Model):
  # Data params
  audio_fs = 22050
  subseq_len = 250
  subseq_nsamps = 64000

  # NSynth decoder params
  num_stages = 10
  num_layers = 30
  filter_length = 3
  width = 512
  skip_width = 256

  # NSynth encoder params
  ae_num_stages = 10
  ae_num_layers = 30
  ae_filter_length = 3
  ae_width = 128
  ae_hop_length = 512
  ae_bottleneck_width = 16

  # Other model params
  input_noise = 'uniform'

  # Loss
  recon_loss_type = 'wave' #'spec' # wave, spec
  recon_objective = 'l1' #'l2' # l1, l2

  # Training
  train_batch_size = 1
  train_lr = 1e-4


  def build_wavenet_decoder(self, z_wave, x_spec):
    width = self.width
    filter_length = self.filter_length
    skip_width = self.skip_width
    num_layers = self.num_layers
    num_stages = self.num_stages

    z_wave = z_wave[:, :, 0]
    x_spec = x_spec[:, :, :, 0]

    l = z_wave
    en = x_spec

    # x_scaled is [None, 64000, 1]
    # convolve to expand channel depth to [None, 64000, 512] (initial input)
    l = masked_conv1d(
	l, num_filters=width, filter_length=filter_length, name='startconv')

    # Set up skip connections.
    # this is first skip connection of [None, 64000, 256]
    # why not just use l or x_scaled here?
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
    #logits = masked_conv1d(s, num_filters=256, filter_length=1, name='logits')
    output = masked_conv1d(s, num_filters=1, filter_length=1, name='logits')
    #output = tf.tanh(output)

    return output[:, :, tf.newaxis, :]


  def __call__(self, x_spec, x_wave):
    batch_size = advoc.util.best_shape(x_wave, axis=0)

    # TODO: normalize spec??

    # Noise var (phase)
    if self.input_noise == 'uniform':
      z_wave = tf.random.uniform([batch_size, self.subseq_nsamps, 1, 1], minval=-1, maxval=1, dtype=tf.float32)
    elif self.input_noise == 'gaussian':
      z_wave = tf.random.normal([batch_size, self.subseq_nsamps, 1, 1], dtype=tf.float32)
    else:
      raise ValueError()

    with tf.variable_scope('vocoder'):
      x_hat = self.build_wavenet_decoder(z_wave, x_spec)
    vocoder_vars = tf.trainable_variables(scope='vocoder')

    num_params = 0
    for v in vocoder_vars:
      num_params += np.prod(v.shape.as_list())
    print('Model size: {:.4f} GB'.format(float(num_params) * 4 / 1024 / 1024 / 1024))

    tf.summary.audio('x_wave', x_wave[:, :, 0, 0], self.audio_fs)
    tf.summary.audio('x_hat', x_hat[:, :, 0, 0], self.audio_fs)
    
    x_hat_spec = advoc.spectral.waveform_to_r9y9_melspec_tf(x_hat, fs=self.audio_fs)

    tf.summary.image('x_spec', advoc.util.r9y9_melspec_to_uint8_img(x_spec))
    tf.summary.image('x_hat_spec', advoc.util.r9y9_melspec_to_uint8_img(x_hat_spec))

    wav_l1 = tf.reduce_mean(tf.abs(x_hat - x_wave))
    wav_l2 = tf.reduce_mean(tf.square(x_hat - x_wave))
    spec_l1 = tf.reduce_mean(tf.abs(x_hat_spec - x_spec))
    spec_l2 = tf.reduce_mean(tf.square(x_hat_spec - x_spec))

    tf.summary.scalar('wav_l1', wav_l1)
    tf.summary.scalar('wav_l2', wav_l2)
    tf.summary.scalar('spec_l1', spec_l1)
    tf.summary.scalar('spec_l2', spec_l2)

    if self.recon_loss_type == 'wave' and self.recon_objective == 'l1':
      loss = wav_l1
    elif self.recon_loss_type == 'wave' and self.recon_objective == 'l2':
      loss = wav_l2
    elif self.recon_loss_type == 'spec' and self.recon_objective == 'l1':
      loss = spec_l1
    elif self.recon_loss_type == 'spec' and self.recon_objective == 'l2':
      loss = spec_l2
    else:
      raise ValueError()

    opt = tf.train.AdamOptimizer(learning_rate=self.train_lr)

    self.train_op = opt.minimize(
        loss,
        global_step=tf.train.get_or_create_global_step(),
        var_list=vocoder_vars)


  def train_loop(self, sess):
    sess.run(self.train_op)
