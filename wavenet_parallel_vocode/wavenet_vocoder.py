import numpy as np
import tensorflow as tf

import advoc.util
import advoc.spectral

from model import AudioModel, Modes
from wavenet import build_nsynth_wavenet_decoder


class WavenetVocoder(AudioModel):
  # Data params
  audio_fs = 22050
  subseq_len = 24
  subseq_nsamps = 6144

  # NSynth decoder params
  num_stages = 10
  num_layers = 30
  filter_length = 3
  width = 256
  skip_width = 128
  causal = True # TODO: change this

  # Other model params
  input_type = 'gaussian_spec' #'gaussian_spec', 'uniform_spec', 'spec_none', 'spec_spec'

  # Training
  train_recon_domain = 'spec' #'spec' # wave, spec
  train_recon_norm = 'l2' #'l2' # l1, l2
  train_batch_size = 32
  train_lr = 2e-4

  # Evaluation
  eval_batch_size = 16
  eval_batch_num = 32


  def __init__(self, mode):
    super().__init__(
        mode,
        spectral=True,
        train_batch_size=self.train_batch_size,
        subseq_len=self.subseq_len,
        audio_fs=self.audio_fs)


  def get_global_variables(self):
    return self.global_vars


  def __call__(self, x_spec, x_wave):
    batch_size, _, nmels, _ = advoc.util.best_shape(x_spec)

    if self.input_type == 'uniform_spec':
      input_wave = tf.random.uniform([batch_size, self.subseq_nsamps, 1], minval=-1, maxval=1, dtype=tf.float32)
      input_spec = x_spec[:, :, :, 0]
    elif self.input_type == 'gaussian_spec':
      input_wave = tf.random.normal([batch_size, self.subseq_nsamps, 1], dtype=tf.float32)
      input_spec = x_spec[:, :, :, 0]
    elif self.input_type == 'spec_none':
      input_wave = x_spec[:, :, :, 0]

      # Upsample [32, 24, 80] to [32, 6144, 80]
      input_wave = input_wave[:, :, tf.newaxis, :]
      compression = self.subseq_nsamps // self.subseq_len
      input_wave = tf.tile(input_wave, [1, 1, compression, 1])
      input_wave = tf.reshape(input_wave, [batch_size, -1, nmels])

      input_spec = None
    elif self.input_type == 'spec_spec':
      input_wave = x_spec[:, :, :, 0]

      # Upsample [32, 24, 80] to [32, 6144, 80]
      input_wave = input_wave[:, :, tf.newaxis, :]
      compression = self.subseq_nsamps // self.subseq_len
      input_wave = tf.tile(input_wave, [1, 1, compression, 1])
      input_wave = tf.reshape(input_wave, [batch_size, -1, nmels])

      input_spec = x_spec[:, :, :, 0]
    else:
      raise ValueError()

    with tf.variable_scope('decoder'):
      vocoded_wave = build_nsynth_wavenet_decoder(
          input_wave,
          input_spec,
          output_width=1,
          num_stages=self.num_stages,
          num_layers=self.num_layers,
          filter_length=self.filter_length,
          width=self.width,
          skip_width=self.skip_width)[:, :, tf.newaxis, :]
    self.global_vars = tf.global_variables(scope='decoder')
    trainable_vars = tf.trainable_variables(scope='decoder')
    assert len(self.global_vars) == len(trainable_vars)

    num_params = 0
    for v in trainable_vars:
      num_params += np.prod(v.shape.as_list())
    print('Model size: {:.4f} GB'.format(float(num_params) * 4 / 1024 / 1024 / 1024))

    vocoded_wave_spec = advoc.spectral.waveform_to_r9y9_melspec_tf(vocoded_wave, fs=self.audio_fs)

    wav_l1 = tf.reduce_mean(tf.abs(vocoded_wave - x_wave))
    wav_l2 = tf.reduce_mean(tf.square(vocoded_wave - x_wave))
    spec_l1 = tf.reduce_mean(tf.abs(vocoded_wave_spec - x_spec))
    spec_l2 = tf.reduce_mean(tf.square(vocoded_wave_spec - x_spec))

    if self.mode == Modes.TRAIN:
      if self.train_recon_domain == 'wave' and self.train_recon_norm == 'l1':
        loss = wav_l1
      elif self.train_recon_domain == 'wave' and self.train_recon_norm == 'l2':
        loss = wav_l2
      elif self.train_recon_domain == 'spec' and self.train_recon_norm == 'l1':
        loss = spec_l1
      elif self.train_recon_domain == 'spec' and self.train_recon_norm == 'l2':
        loss = spec_l2
      else:
        raise ValueError()

      tf.summary.audio('x_wave', x_wave[:, :, 0, 0], self.audio_fs)
      tf.summary.audio('x_vocoded', vocoded_wave[:, :, 0, 0], self.audio_fs)
      tf.summary.image('x_spec', advoc.util.r9y9_melspec_to_uint8_img(x_spec))
      tf.summary.image('x_vocoded_spec', advoc.util.r9y9_melspec_to_uint8_img(vocoded_wave_spec))
      tf.summary.scalar('loss', loss)
      tf.summary.scalar('wav_l1', wav_l1)
      tf.summary.scalar('wav_l2', wav_l2)
      tf.summary.scalar('spec_l1', spec_l1)
      tf.summary.scalar('spec_l2', spec_l2)

      opt = tf.train.AdamOptimizer(learning_rate=self.train_lr)

      self.train_op = opt.minimize(
          loss,
          global_step=tf.train.get_or_create_global_step(),
          var_list=trainable_vars)
    elif self.mode == Modes.EVAL:
      self.all_nll = tf.placeholder(tf.float32, [None])
      self.all_wav_l1 = tf.placeholder(tf.float32, [None])
      self.all_wav_l2 = tf.placeholder(tf.float32, [None])
      self.all_spec_l1 = tf.placeholder(tf.float32, [None])
      self.all_spec_l2 = tf.placeholder(tf.float32, [None])

      avg_nll = tf.reduce_mean(self.all_nll)
      summaries = [
          tf.summary.scalar('nll', avg_nll),
          tf.summary.scalar('ppl', tf.exp(avg_nll)),
          tf.summary.scalar('wav_l1', tf.reduce_mean(self.all_wav_l1)),
          tf.summary.scalar('wav_l2', tf.reduce_mean(self.all_wav_l2)),
          tf.summary.scalar('spec_l1', tf.reduce_mean(self.all_spec_l1)),
          tf.summary.scalar('spec_l2', tf.reduce_mean(self.all_spec_l2))
      ]
      self.summaries = tf.summary.merge(summaries)

      self.best_avg_nll = None
      self.best_wav_l1 = None
      self.best_spec_l2 = None


  def train_loop(self, sess):
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    sess.run(self.train_op, options=run_options)


  def eval_ckpt(self, sess, wavenet_sess):
    pass
