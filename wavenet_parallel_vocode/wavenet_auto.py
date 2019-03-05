import numpy as np
import tensorflow as tf

import advoc.util
import advoc.spectral

from model import AudioModel, Modes
from wavenet import build_nsynth_wavenet_decoder, shift_right, mu_law, inv_mu_law


class Wavenet(AudioModel):
  # Data params
  audio_fs = 22050
  subseq_len = 6144
  subseq_nsamps = 6144

  # NSynth decoder params
  num_stages = 10
  num_layers = 30
  filter_length = 3
  width = 256
  skip_width = 128

  # Training
  train_batch_size = 32
  train_lr = 2e-4

  def __init__(self, mode):
    super().__init__(
        mode,
        spectral=False,
        train_batch_size=self.train_batch_size,
        subseq_len=self.subseq_len,
        audio_fs=self.audio_fs)


  def __call__(self, x_spec, x_wave):
    batch_size = advoc.util.best_shape(x_wave, axis=0)

    tf.summary.audio('x_wave', x_wave[:, :, 0, 0], self.audio_fs)

    x_quantized = mu_law(x_wave[:,  :, 0])
    x_scaled = tf.cast(x_quantized, tf.float32) / 128.0

    tf.summary.audio('x_mulaw', inv_mu_law(x_quantized)[:, :, 0], self.audio_fs)

    x_shift = tf.stop_gradient(shift_right(x_scaled))
    x_indices = tf.stop_gradient(tf.cast(x_quantized[:, :, 0], tf.int32) + 128)

    with tf.variable_scope('decoder'):
      logits = build_nsynth_wavenet_decoder(
          x_shift,
          None,
          output_width=256,
          num_stages=self.num_stages,
          num_layers=self.num_layers,
          filter_length=self.filter_length,
          width=self.width,
          skip_width=self.skip_width)
    decoder_vars = tf.trainable_variables(scope='decoder')

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=x_indices,
        logits=logits)
    loss = tf.reduce_mean(loss)

    tf.summary.scalar('loss', loss)

    opt = tf.train.AdamOptimizer(learning_rate=self.train_lr)
    self.train_op = opt.minimize(
        loss,
        global_step=tf.train.get_or_create_global_step(),
        var_list=decoder_vars)


  def train_loop(self, sess):
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    sess.run(self.train_op, options=run_options)
