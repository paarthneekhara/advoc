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

  # Evaluation
  eval_batch_size = 16
  eval_batch_num = 16

  def __init__(self, mode):
    super().__init__(
        mode,
        spectral=False,
        train_batch_size=self.train_batch_size,
        subseq_len=self.subseq_len,
        audio_fs=self.audio_fs)


  def get_global_variables(self):
    return self.global_vars


  def __call__(self, x_spec, x_wave, train=False):
    batch_size = advoc.util.best_shape(x_wave, axis=0)

    x_quantized = mu_law(x_wave[:,  :, 0])
    x_scaled = tf.cast(x_quantized, tf.float32) / 128.0

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
    self.global_vars = tf.global_variables(scope='decoder')
    trainable_vars = tf.trainable_variables(scope='decoder')

    self.nll = nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=x_indices,
        logits=logits)

    if self.mode == Modes.TRAIN:
      avg_nll = tf.reduce_mean(nll)
      avg_nll_last = tf.reduce_mean(self.nll[:, -1])

      tf.summary.scalar('nll', avg_nll)
      tf.summary.scalar('nll_last', avg_nll)
      tf.summary.scalar('ppl', tf.exp(avg_nll))
      tf.summary.scalar('ppl_last', tf.exp(avg_nll_last))
      tf.summary.audio('x_wave', x_wave[:, :, 0, 0], self.audio_fs)
      tf.summary.audio('x_mulaw', inv_mu_law(x_quantized)[:, :, 0], self.audio_fs)

      loss = avg_nll
      opt = tf.train.AdamOptimizer(learning_rate=self.train_lr)
      self.train_op = opt.minimize(
          loss,
          global_step=tf.train.get_or_create_global_step(),
          var_list=trainable_vars)
    elif self.mode == Modes.EVAL:
      self.all_nll = tf.placeholder(tf.float32, [None, self.subseq_len])
      avg_nll = tf.reduce_mean(self.all_nll)
      avg_nll_last = tf.reduce_mean(self.all_nll[:, -1])

      summaries = [
          tf.summary.scalar('nll', avg_nll),
          tf.summary.scalar('nll_last', avg_nll_last),
          tf.summary.scalar('ppl', tf.exp(avg_nll)),
          tf.summary.scalar('ppl_last', tf.exp(avg_nll_last)),
      ]
      self.summaries = tf.summary.merge(summaries)
      self.best_avg_nll = None


  def train_loop(self, sess):
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    sess.run(self.train_op, options=run_options)


  def eval_ckpt(self, sess):
    _all_nll = []
    for i in range(self.eval_batch_num):
      try:
        _nll = sess.run(self.nll)
      except tf.errors.OutOfRangeError:
        break
      _all_nll.append(_nll)
    _all_nll = np.concatenate(_all_nll, axis=0)

    _summaries = sess.run(self.summaries, {self.all_nll: _all_nll})

    _avg_nll = np.mean(_all_nll)
    best = False
    if self.best_avg_nll is None or _avg_nll < self.best_avg_nll:
      best = True
      self.best_avg_nll = _avg_nll

    return best, _summaries
