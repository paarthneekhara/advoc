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
  train_recon_domain = 'r9y9_recomp' # wave, r9y9, linmagspec, logmagspec
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

    with tf.variable_scope('vocoder'):
      self.vocoded_wave = vocoded_wave = build_nsynth_wavenet_decoder(
          input_wave,
          input_spec,
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
    x_wave_spec = tf.stop_gradient(advoc.spectral.waveform_to_r9y9_melspec_tf(x_wave, fs=self.audio_fs))
    vocoded_wave_spec = advoc.spectral.waveform_to_r9y9_melspec_tf(vocoded_wave, fs=self.audio_fs)

    x_wave_magspec = tf.stop_gradient(tf.abs(tf.contrib.signal.stft(x_wave[:, :, 0, 0], 1024, 256, pad_end=True)))
    vocoded_wave_magspec = tf.abs(tf.contrib.signal.stft(vocoded_wave[:, :, 0, 0], 1024, 256, pad_end=True))

    self.wav_l1 = wav_l1 = tf.reduce_mean(tf.abs(vocoded_wave - x_wave))
    self.wav_l2 = wav_l2 = tf.reduce_mean(tf.square(vocoded_wave - x_wave))
    self.spec_l1 = spec_l1 = tf.reduce_mean(tf.abs(vocoded_wave_spec - x_wave_spec))
    self.spec_l2 = spec_l2 = tf.reduce_mean(tf.square(vocoded_wave_spec - x_wave_spec))
    self.magspec_l1 = magspec_l1 = tf.reduce_mean(tf.abs(vocoded_wave_magspec - x_wave_magspec))
    self.magspec_l2 = magspec_l2 = tf.reduce_mean(tf.square(vocoded_wave_magspec - x_wave_magspec))

    if self.mode == Modes.TRAIN:
      if self.train_recon_domain == 'wave' and self.train_recon_norm == 'l1':
        loss = self.train_recon_multiplier * wav_l1
      elif self.train_recon_domain == 'wave' and self.train_recon_norm == 'l2':
        loss = self.train_recon_multiplier * wav_l2
      elif self.train_recon_domain == 'spec' and self.train_recon_norm == 'l1':
        loss = self.train_recon_multiplier * spec_l1
      elif self.train_recon_domain == 'spec' and self.train_recon_norm == 'l2':
        loss = self.train_recon_multiplier * spec_l2
      elif self.train_recon_domain == 'magspec' and self.train_recon_norm == 'l1':
        loss = self.train_recon_multiplier * magspec_l1
      elif self.train_recon_domain == 'magspec' and self.train_recon_norm == 'l2':
        loss = self.train_recon_multiplier * magspec_l2
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

      tf.summary.audio('x_wave', x_wave[:, :, 0, 0], self.audio_fs)
      tf.summary.audio('x_vocoded', vocoded_wave[:, :, 0, 0], self.audio_fs)
      tf.summary.image('x_spec', advoc.util.r9y9_melspec_to_uint8_img(x_wave_spec))
      tf.summary.image('x_vocoded_spec', advoc.util.r9y9_melspec_to_uint8_img(vocoded_wave_spec))
      tf.summary.scalar('loss', loss)
      tf.summary.scalar('wav_l1', wav_l1)
      tf.summary.scalar('wav_l2', wav_l2)
      tf.summary.scalar('spec_l1', spec_l1)
      tf.summary.scalar('spec_l2', spec_l2)
      tf.summary.scalar('magspec_l1', magspec_l1)
      tf.summary.scalar('magspec_l2', magspec_l2)

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

      self.best_nll = None
      self.best_wav_l1 = None
      self.best_spec_l2 = None


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
    _all_spec_l1 = []
    _all_spec_l2 = []
    for i in range(self.eval_batch_num):
      try:
        _vocoded_wave, _wav_l1, _wav_l2, _spec_l1, _spec_l2 = sess.run([
          self.vocoded_wave,
          self.wav_l1,
          self.wav_l2,
          self.spec_l1,
          self.spec_l2
        ])
      except tf.errors.OutOfRangeError:
        break

      if self.wavenet_sess is not None:
        _avg_nll = self.wavenet_sess.run(self.wavenet_avg_nll, {self.wavenet_input_wave: _vocoded_wave})
        _all_nll.append(_avg_nll)

      _all_wav_l1.append(_wav_l1)
      _all_wav_l2.append(_wav_l2)
      _all_spec_l1.append(_spec_l1)
      _all_spec_l2.append(_spec_l2)

    _all_nll = np.concatenate(_all_nll, axis=0)

    _summaries = sess.run(self.summaries, {
      self.all_nll: _all_nll,
      self.all_wav_l1: _all_wav_l1,
      self.all_wav_l2: _all_wav_l2,
      self.all_spec_l1: _all_spec_l1,
      self.all_spec_l2: _all_spec_l2
    })

    best = []

    _avg_nll = np.mean(_all_nll)
    if self.best_nll is None or _avg_nll < self.best_nll:
      best.append('wavenet_nll')
      self.best_nll = _avg_nll

    _avg_wav_l1 = np.mean(_all_wav_l1)
    if self.best_wav_l1 is None or _avg_wav_l1 < self.best_wav_l1:
      best.append('wav_l1')
      self.best_wav_l1 = _avg_wav_l1

    _avg_spec_l2 = np.mean(_all_spec_l2)
    if self.best_spec_l2 is None or _avg_spec_l2 < self.best_spec_l2:
      best.append('spec_l2')
      self.best_spec_l2 = _avg_spec_l2

    return best, _summaries
