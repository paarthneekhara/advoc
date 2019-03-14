import numpy as np
import tensorflow as tf

import advoc.util
import advoc.spectral

from model import AudioModel, Modes
from wavenet import build_nsynth_wavenet_decoder, build_nsynth_wavenet_encoder


class WavenetVocoder(AudioModel):
  # Data params
  audio_fs = 22050
  subseq_len = 24
  subseq_nsamps = 6144

  # NSynth decoder params
  num_stages = 10
  num_layers = 20 # originally 30
  filter_length = 3
  width = 128 # originally 256
  skip_width = 128
  causal = False # originally True

  # NSynth encoder params
  ae_num_stages = 10
  ae_num_layers = 30
  ae_filter_length = 3
  ae_width = 128
  ae_hop_length = 512

  # Other model params
  input_type = 'gaussian_spec' #'gaussian_spec', 'uniform_spec', 'spec_none', 'spec_spec'
  input_spec_upsample = 'default' #'lin', 'learned'

  # Training
  train_recon_domain = 'r9y9_legacy' # wave, r9y9, linmagspec, logmagspec
  train_recon_norm = 'l2' #'l2' # l1, l2
  train_recon_multiplier = 1.
  train_gan = False
  train_gan_objective = 'wgangp'
  train_gan_multiplier = 1.
  train_wgangp_lambda = 10
  train_batch_size = 32
  train_lr = 2e-4

  # Evaluation
  eval_batch_size = 16
  eval_batch_num = 16
  eval_wavenet_metagraph_fp = ''
  eval_wavenet_ckpt_fp = ''


  def __init__(self, mode):
    super().__init__(
        mode,
        spectral=True,
        train_batch_size=self.train_batch_size,
        subseq_len=self.subseq_len,
        audio_fs=self.audio_fs)


  def get_global_variables(self):
    return self.global_vars


  def __call__(self, x_r9y9, x_wave):
    batch_size, _, nmels, _ = advoc.util.best_shape(x_r9y9)

    # Optionally upsample r9y9trogram
    if self.input_spec_upsample == 'default':
      x_r9y9_up = x_r9y9
    elif self.input_spec_upsample == 'nearest_neighbor':
      x_r9y9_up = tf.stop_gradient(tf.image.resize_nearest_neighbor(
          x_r9y9,
          [self.subseq_nsamps, nmels]))
    elif self.input_spec_upsample == 'linear':
      x_r9y9_up = tf.stop_gradient(tf.image.resize_bilinear(
          x_r9y9,
          [self.subseq_nsamps, nmels]))
    elif self.input_spec_upsample == 'learned':
      x_r9y9_up = x_r9y9
      with tf.variable_scope('vocoder'):
        while int(x_r9y9_up.get_shape()[1]) != self.subseq_nsamps:
          if int(x_r9y9_up.get_shape()[1]) > self.subseq_nsamps:
            raise ValueError()
          x_r9y9_up = tf.layers.conv2d_transpose(
              x_r9y9_up,
              nmels,
              (9, 1),
              strides=(4, 1),
              padding='same')
          x_r9y9_up = tf.nn.relu(x_r9y9_up)
    else:
      raise ValueError()

    # Create input structure
    # First part (e.g. uniform) represents waveform-rate input to WaveNet.
    # Second part (e.g. r9y9) represents r9y9-rate conditioning info (number of timesteps must perfectly divide number of audio samples)
    if self.input_type == 'uniform_spec':
      input_wave = tf.random.uniform([batch_size, self.subseq_nsamps, 1, 1], minval=-1, maxval=1, dtype=tf.float32)
      input_cond = x_r9y9_up
    elif self.input_type == 'gaussian_spec':
      input_wave = tf.random.normal([batch_size, self.subseq_nsamps, 1, 1], dtype=tf.float32)
      input_cond = x_r9y9_up
    elif self.input_type == 'spec_none':
      input_wave = x_r9y9_up
      input_cond = None
    elif self.input_type == 'spec_spec':
      input_wave = x_r9y9_up
      input_cond = x_r9y9_up
    elif self.input_type == 'spec_lospec':
      input_wave = x_r9y9_up
      input_cond = x_r9y9
    else:
      raise ValueError()

    with tf.variable_scope('vocoder'):
      self.vocoded_wave = vocoded_wave = build_nsynth_wavenet_decoder(
          input_wave[:, :, 0, :],
          input_cond[:, :, :, 0] if input_cond is not None else input_cond,
          causal=self.causal,
          output_width=1,
          num_stages=self.num_stages,
          num_layers=self.num_layers,
          filter_length=self.filter_length,
          width=self.width,
          skip_width=self.skip_width)[:, :, tf.newaxis, :]
    self.global_vars = tf.global_variables(scope='vocoder')
    trainable_vars = tf.trainable_variables(scope='vocoder')
    assert len(self.global_vars) == len(trainable_vars)

    num_params = 0
    for v in trainable_vars:
      num_params += np.prod(v.shape.as_list())
    print('Model size: {:.4f} GB'.format(float(num_params) * 4 / 1024 / 1024 / 1024))

    # TODO: remove stop gradient on these?
    x_wave_r9y9 = tf.stop_gradient(advoc.spectral.waveform_to_r9y9_melspec_tf(x_wave, fs=self.audio_fs))
    vocoded_wave_r9y9 = advoc.spectral.waveform_to_r9y9_melspec_tf(vocoded_wave, fs=self.audio_fs)

    # TODO: Reshape these to proper chris-like spectrogram
    x_wave_linmagspec = tf.stop_gradient(tf.abs(tf.contrib.signal.stft(x_wave[:, :, 0, 0], 1024, 256, pad_end=True)))
    vocoded_wave_linmagspec = tf.abs(tf.contrib.signal.stft(vocoded_wave[:, :, 0, 0], 1024, 256, pad_end=True))

    x_wave_logmagspec = tf.stop_gradient(tf.log(x_wave_linmagspec + 1e-10))
    vocoded_wave_logmagspec = tf.log(vocoded_wave_linmagspec + 1e-10)

    self.wav_l1 = wav_l1 = tf.reduce_mean(tf.abs(vocoded_wave - x_wave))
    self.wav_l2 = wav_l2 = tf.reduce_mean(tf.square(vocoded_wave - x_wave))
    self.r9y9_legacy_l1 = r9y9_legacy_l1 = tf.reduce_mean(tf.abs(vocoded_wave_r9y9 - x_r9y9))
    self.r9y9_legacy_l2 = r9y9_legacy_l2 = tf.reduce_mean(tf.square(vocoded_wave_r9y9 - x_r9y9))
    self.r9y9_l1 = r9y9_l1 = tf.reduce_mean(tf.abs(vocoded_wave_r9y9 - x_wave_r9y9))
    self.r9y9_l2 = r9y9_l2 = tf.reduce_mean(tf.square(vocoded_wave_r9y9 - x_wave_r9y9))
    self.linmagspec_l1 = linmagspec_l1 = tf.reduce_mean(tf.abs(vocoded_wave_linmagspec - x_wave_linmagspec))
    self.linmagspec_l2 = linmagspec_l2 = tf.reduce_mean(tf.square(vocoded_wave_linmagspec - x_wave_linmagspec))
    self.logmagspec_l1 = logmagspec_l1 = tf.reduce_mean(tf.abs(vocoded_wave_logmagspec - x_wave_logmagspec))
    self.logmagspec_l2 = logmagspec_l2 = tf.reduce_mean(tf.square(vocoded_wave_logmagspec - x_wave_logmagspec))

    if self.mode == Modes.TRAIN:
      if self.train_recon_domain == 'wave' and self.train_recon_norm == 'l1':
        loss = self.train_recon_multiplier * wav_l1
      elif self.train_recon_domain == 'wave' and self.train_recon_norm == 'l2':
        loss = self.train_recon_multiplier * wav_l2
      elif self.train_recon_domain == 'r9y9_legacy' and self.train_recon_norm == 'l1':
        loss = self.train_recon_multiplier * r9y9_legacy_l1
      elif self.train_recon_domain == 'r9y9_legacy' and self.train_recon_norm == 'l2':
        loss = self.train_recon_multiplier * r9y9_legacy_l2
      elif self.train_recon_domain == 'r9y9' and self.train_recon_norm == 'l1':
        loss = self.train_recon_multiplier * r9y9_l1
      elif self.train_recon_domain == 'r9y9' and self.train_recon_norm == 'l2':
        loss = self.train_recon_multiplier * r9y9_l2
      elif self.train_recon_domain == 'linmagspec' and self.train_recon_norm == 'l1':
        loss = self.train_recon_multiplier * linmagspec_l1
      elif self.train_recon_domain == 'linmagspec' and self.train_recon_norm == 'l2':
        loss = self.train_recon_multiplier * linmagspec_l2
      elif self.train_recon_domain == 'logmagspec' and self.train_recon_norm == 'l1':
        loss = self.train_recon_multiplier * logmagspec_l1
      elif self.train_recon_domain == 'logmagspec' and self.train_recon_norm == 'l2':
        loss = self.train_recon_multiplier * logmagspec_l2
      else:
        raise ValueError()

      if self.train_gan:
        with tf.name_scope('D_x'), tf.variable_scope('discriminator'):
          # TODO: get spec into encoder somehow
          D_x = build_nsynth_wavenet_encoder(
              x_wave[:, :, 0, :],
              num_stages=self.ae_num_stages,
              num_layers=self.ae_num_layers,
              filter_length=self.ae_filter_length,
              width=self.ae_width,
              hop_length=self.ae_hop_length,
              bottleneck_width=1)
        D_vars = tf.trainable_variables(scope='discriminator')
        assert len(D_vars) == len(tf.global_variables('discriminator'))

        with tf.name_scope('D_G_z'), tf.variable_scope('discriminator', reuse=True):
          D_G_z = build_nsynth_wavenet_encoder(
              vocoded_wave[:, :, 0, :],
              num_stages=self.ae_num_stages,
              num_layers=self.ae_num_layers,
              filter_length=self.ae_filter_length,
              width=self.ae_width,
              hop_length=self.ae_hop_length,
              bottleneck_width=1)

        if self.train_gan_objective == 'dcgan':
          fake = tf.zeros_like(D_G_z)
          real = tf.ones_like(D_x)

          G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_G_z,
            labels=real
          ))

          D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_G_z,
            labels=fake
          ))
          D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_x,
            labels=real
          ))

          D_loss /= 2.

          self.train_gan_num_disc_updates = 1
        elif self.train_gan_objective == 'wgangp':
          G_loss = -tf.reduce_mean(D_G_z)
          D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

          alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
          differences = vocoded_wave - x_wave
          interpolates = x_wave + (alpha * differences)
          with tf.name_scope('D_G_z'), tf.variable_scope('discriminator', reuse=True):
            D_interp = build_nsynth_wavenet_encoder(
                interpolates[:, :, 0, :],
                num_stages=self.ae_num_stages,
                num_layers=self.ae_num_layers,
                filter_length=self.ae_filter_length,
                width=self.ae_width,
                hop_length=self.ae_hop_length,
                bottleneck_width=1)

          gradients = tf.gradients(D_interp, [interpolates])[0]
          slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
          gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
          D_loss += self.train_wgangp_lambda * gradient_penalty

          self.train_gan_num_disc_updates = 5
        else:
          raise ValueError()

        loss += self.train_gan_multiplier * G_loss

        D_opt = tf.train.AdamOptimizer(learning_rate=self.train_lr)
        self.D_train_op = D_opt.minimize(
            D_loss,
            var_list=D_vars)

        tf.summary.scalar('D_loss', D_loss)
        tf.summary.scalar('G_loss', G_loss)

      x_r9y9_up_preview = tf.image.resize_nearest_neighbor(
          x_r9y9_up,
          [self.subseq_len * 8, nmels])

      tf.summary.image('x_r9y9', advoc.util.r9y9_melspec_to_uint8_img(x_r9y9))
      tf.summary.image('x_r9y9_up', advoc.util.r9y9_melspec_to_uint8_img(x_r9y9_up_preview))
      tf.summary.audio('x_wave', x_wave[:, :, 0, 0], self.audio_fs)
      tf.summary.image('x_wave_r9y9', advoc.util.r9y9_melspec_to_uint8_img(x_wave_r9y9))
      tf.summary.audio('x_vocoded', vocoded_wave[:, :, 0, 0], self.audio_fs)
      tf.summary.image('x_vocoded_r9y9', advoc.util.r9y9_melspec_to_uint8_img(vocoded_wave_r9y9))
      tf.summary.scalar('loss', loss)
      tf.summary.scalar('wav_l1', wav_l1)
      tf.summary.scalar('wav_l2', wav_l2)
      tf.summary.scalar('r9y9_legacy_l1', r9y9_legacy_l1)
      tf.summary.scalar('r9y9_legacy_l2', r9y9_legacy_l2)
      tf.summary.scalar('r9y9_l1', r9y9_l1)
      tf.summary.scalar('r9y9_l2', r9y9_l2)
      tf.summary.scalar('linmagspec_l1', linmagspec_l1)
      tf.summary.scalar('linmagspec_l2', linmagspec_l2)
      tf.summary.scalar('logmagspec_l1', logmagspec_l1)
      tf.summary.scalar('logmagspec_l2', logmagspec_l2)

      opt = tf.train.AdamOptimizer(learning_rate=self.train_lr)

      self.train_op = opt.minimize(
          loss,
          global_step=tf.train.get_or_create_global_step(),
          var_list=trainable_vars)
    elif self.mode == Modes.EVAL:
      self.wavenet_sess = None
      if len(self.eval_wavenet_metagraph_fp.strip()) > 0:
        wavenet_graph = tf.Graph()
        with wavenet_graph.as_default():
          wavenet_saver = tf.train.import_meta_graph(self.eval_wavenet_metagraph_fp)
        self.wavenet_sess = tf.Session(graph=wavenet_graph)
        wavenet_saver.restore(self.wavenet_sess, self.eval_wavenet_ckpt_fp)
        wavenet_step = wavenet_graph.get_tensor_by_name('global_step:0')
        self.wavenet_input_wave = wavenet_graph.get_tensor_by_name('input_wave:0')
        self.wavenet_avg_nll = wavenet_graph.get_tensor_by_name('avg_nll:0')
        print('Loaded WaveNet (step {})'.format(self.wavenet_sess.run(wavenet_step)))

      self.all_nll = tf.placeholder(tf.float32, [None])
      self.all_wav_l1 = tf.placeholder(tf.float32, [None])
      self.all_wav_l2 = tf.placeholder(tf.float32, [None])
      self.all_r9y9_legacy_l1 = tf.placeholder(tf.float32, [None])
      self.all_r9y9_legacy_l2 = tf.placeholder(tf.float32, [None])
      self.all_r9y9_l1 = tf.placeholder(tf.float32, [None])
      self.all_r9y9_l2 = tf.placeholder(tf.float32, [None])
      self.all_linmagspec_l1 = tf.placeholder(tf.float32, [None])
      self.all_linmagspec_l2 = tf.placeholder(tf.float32, [None])
      self.all_logmagspec_l1 = tf.placeholder(tf.float32, [None])
      self.all_logmagspec_l2 = tf.placeholder(tf.float32, [None])

      avg_nll = tf.reduce_mean(self.all_nll)
      summaries = [
          tf.summary.scalar('nll', avg_nll),
          tf.summary.scalar('ppl', tf.exp(avg_nll)),
          tf.summary.scalar('wav_l1', tf.reduce_mean(self.all_wav_l1)),
          tf.summary.scalar('wav_l2', tf.reduce_mean(self.all_wav_l2)),
          tf.summary.scalar('r9y9_legacy_l1', tf.reduce_mean(self.all_r9y9_legacy_l1)),
          tf.summary.scalar('r9y9_legacy_l2', tf.reduce_mean(self.all_r9y9_legacy_l2)),
          tf.summary.scalar('r9y9_l1', tf.reduce_mean(self.all_r9y9_l1)),
          tf.summary.scalar('r9y9_l2', tf.reduce_mean(self.all_r9y9_l2)),
          tf.summary.scalar('linmagspec_l1', tf.reduce_mean(self.all_linmagspec_l1)),
          tf.summary.scalar('linmagspec_l2', tf.reduce_mean(self.all_linmagspec_l2)),
          tf.summary.scalar('logmagspec_l1', tf.reduce_mean(self.all_logmagspec_l1)),
          tf.summary.scalar('logmagspec_l2', tf.reduce_mean(self.all_logmagspec_l2)),
      ]
      self.summaries = tf.summary.merge(summaries)

      self.best_nll = None
      self.best_wav_l1 = None
      self.best_r9y9_l2 = None
      self.best_linmagspec_l2 = None


  def train_loop(self, sess):
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    if self.train_gan:
      for i in range(self.train_gan_num_disc_updates):
        sess.run(self.D_train_op, options=run_options)
    sess.run(self.train_op, options=run_options)


  def eval_ckpt(self, sess):
    if self.wavenet_sess is not None and self.eval_batch_size != 16:
      raise NotImplementedError()

    _all_nll = []
    _all_wav_l1 = []
    _all_wav_l2 = []
    _all_r9y9_legacy_l1 = []
    _all_r9y9_legacy_l2 = []
    _all_r9y9_l1 = []
    _all_r9y9_l2 = []
    _all_linmagspec_l1 = []
    _all_linmagspec_l2 = []
    _all_logmagspec_l1 = []
    _all_logmagspec_l2 = []
    for i in range(self.eval_batch_num):
      try:
        _vocoded_wave, _wav_l1, _wav_l2, _r9y9_legacy_l1, _r9y9_legacy_l2, _r9y9_l1, _r9y9_l2, _linmagspec_l1, _linmagspec_l2, _logmagspec_l1, _logmagspec_l2 = sess.run([
          self.vocoded_wave,
          self.wav_l1,
          self.wav_l2,
          self.r9y9_legacy_l1,
          self.r9y9_legacy_l2,
          self.r9y9_l1,
          self.r9y9_l2,
          self.linmagspec_l1,
          self.linmagspec_l2,
          self.logmagspec_l1,
          self.logmagspec_l2
        ])
      except tf.errors.OutOfRangeError:
        break

      if self.wavenet_sess is not None:
        _avg_nll = self.wavenet_sess.run(self.wavenet_avg_nll, {self.wavenet_input_wave: _vocoded_wave})
        _all_nll.append(_avg_nll)

      _all_wav_l1.append(_wav_l1)
      _all_wav_l2.append(_wav_l2)
      _all_r9y9_legacy_l1.append(_r9y9_legacy_l1)
      _all_r9y9_legacy_l2.append(_r9y9_legacy_l2)
      _all_r9y9_l1.append(_r9y9_l1)
      _all_r9y9_l2.append(_r9y9_l2)
      _all_linmagspec_l1.append(_linmagspec_l1)
      _all_linmagspec_l2.append(_linmagspec_l2)
      _all_logmagspec_l1.append(_logmagspec_l1)
      _all_logmagspec_l2.append(_logmagspec_l2)

    _all_nll = np.concatenate(_all_nll, axis=0)

    _summaries = sess.run(self.summaries, {
      self.all_nll: _all_nll,
      self.all_wav_l1: _all_wav_l1,
      self.all_wav_l2: _all_wav_l2,
      self.all_r9y9_legacy_l1: _all_r9y9_legacy_l1,
      self.all_r9y9_legacy_l2: _all_r9y9_legacy_l2,
      self.all_r9y9_l1: _all_r9y9_l1,
      self.all_r9y9_l2: _all_r9y9_l2,
      self.all_linmagspec_l1: _all_linmagspec_l1,
      self.all_linmagspec_l2: _all_linmagspec_l2,
      self.all_logmagspec_l1: _all_logmagspec_l1,
      self.all_logmagspec_l2: _all_logmagspec_l2
    })

    best = []

    _avg_nll = np.mean(_all_nll)
    if self.best_nll is None or _avg_nll < self.best_nll:
      best.append('wavenet_nll')
      self.best_nll = _avg_nll

    _avg_wav_l1 = np.mean(_all_wav_l1)
    if self.best_wav_l1 is None or _avg_wav_l1 < self.best_wav_l1:
      best.append('wav')
      self.best_wav_l1 = _avg_wav_l1

    _avg_r9y9_l2 = np.mean(_all_r9y9_l2)
    if self.best_r9y9_l2 is None or _avg_r9y9_l2 < self.best_r9y9_l2:
      best.append('r9y9')
      self.best_r9y9_l2 = _avg_r9y9_l2

    _avg_linmagspec_l2 = np.mean(_all_linmagspec_l2)
    if self.best_linmagspec_l2 is None or _avg_linmagspec_l2 < self.best_linmagspec_l2:
      best.append('linmagspec')
      self.best_linmagspec_l2 = _avg_linmagspec_l2

    return best, _summaries
