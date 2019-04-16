import pickle
import time

import numpy as np
import tensorflow as tf

from advoc.audioio import save_as_wav
from advoc.loader import decode_extract_and_batch
from advoc.spectral import r9y9_melspec_to_waveform
from conv2d import MelspecGANGenerator, MelspecGANDiscriminator
from util import feats_to_uint8_img, feats_to_approx_audio, feats_norm, feats_denorm

TRAIN_BATCH_SIZE = 64
TRAIN_LOSS = 'wgangp'
Z_DIM = 100

def train(fps, args):
  # Load data
  with tf.name_scope('loader'):
    x, x_audio = decode_extract_and_batch(
        fps=fps,
        batch_size=TRAIN_BATCH_SIZE,
        slice_len=64,
        audio_fs=args.data_sample_rate,
        audio_mono=True,
        audio_normalize=args.data_normalize,
        decode_fastwav=args.data_fastwav,
        decode_parallel_calls=8,
        extract_type='melspec',
        extract_nfft=1024,
        extract_nhop=256,
        extract_parallel_calls=8,
        repeat=True,
        shuffle=True,
        shuffle_buffer_size=512,
        slice_first_only=args.data_slice_first_only,
        slice_randomize_offset=args.data_slice_randomize_offset,
        slice_overlap_ratio=args.data_slice_overlap_ratio,
        slice_pad_end=args.data_slice_pad_end,
        prefetch_size=TRAIN_BATCH_SIZE * 8,
        prefetch_gpu_num=args.data_prefetch_gpu_num)
    x = feats_norm(x)

  # Data summaries
  tf.summary.audio('x_audio', x_audio[:, :, 0], args.data_sample_rate)
  tf.summary.image('x', feats_to_uint8_img(feats_denorm(x)))
  tf.summary.audio('x_inv_audio',
      feats_to_approx_audio(feats_denorm(x), args.data_sample_rate, 16384, n=3)[:, :, 0], args.data_sample_rate)

  # Make z vector
  z = tf.random.normal([TRAIN_BATCH_SIZE, Z_DIM], dtype=tf.float32)

  # Make generator
  with tf.variable_scope('G'):
    G = MelspecGANGenerator()
    G_z = G(z, training=True)
  G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

  # Summarize G_z
  tf.summary.image('G_z', feats_to_uint8_img(feats_denorm(G_z)))
  tf.summary.audio('G_z_inv_audio',
      feats_to_approx_audio(feats_denorm(G_z), args.data_sample_rate, 16384, n=3)[:, :, 0], args.data_sample_rate)

  # Make real discriminator
  D = MelspecGANDiscriminator()
  with tf.name_scope('D_x'), tf.variable_scope('D'):
    D_x = D(x, training=True)
  D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

  # Make fake discriminator
  with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
    D_G_z = D(G_z, training=True)

  # Create loss
  num_disc_updates_per_genr = 1
  if TRAIN_LOSS == 'dcgan':
    fake = tf.zeros([TRAIN_BATCH_SIZE], dtype=tf.float32)
    real = tf.ones([TRAIN_BATCH_SIZE], dtype=tf.float32)

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
  elif TRAIN_LOSS == 'wgangp':
    G_loss = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

    alpha = tf.random_uniform(shape=[TRAIN_BATCH_SIZE, 1, 1, 1], minval=0., maxval=1.)
    differences = G_z - x
    interpolates = x + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
      D_interp = D(interpolates, training=True)

    LAMBDA = 10
    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    D_loss += LAMBDA * gradient_penalty

    num_disc_updates_per_genr = 5
  else:
    raise ValueError()

  tf.summary.scalar('G_loss', G_loss)
  tf.summary.scalar('D_loss', D_loss)

  # Create opt
  if TRAIN_LOSS == 'dcgan':
    G_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5)
    D_opt = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5)
  elif TRAIN_LOSS == 'wgangp':
    # TODO: some igul code uses beta1=0.
    G_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9)
    D_opt = tf.train.AdamOptimizer(
        learning_rate=1e-4,
        beta1=0.5,
        beta2=0.9)
  else:
    raise ValueError()

  # Create training ops
  G_train_op = G_opt.minimize(G_loss, var_list=G_vars,
      global_step=tf.train.get_or_create_global_step())
  D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

  # Train
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_ckpt_every_nsecs,
      save_summaries_secs=args.train_summary_every_nsecs) as sess:
    while not sess.should_stop():
      for i in range(num_disc_updates_per_genr):
        sess.run(D_train_op)

      sess.run(G_train_op)


def infer(args):
  zgen_n = tf.placeholder(tf.int32, [], name='samp_z_n')
  zgen = tf.random.normal([zgen_n, Z_DIM], dtype=tf.float32, name='samp_z')

  z = tf.placeholder(tf.float32, [None, Z_DIM], name='z')
  with tf.variable_scope('G'):
    G = MelspecGANGenerator()
    G_z = G(z, training=False)
  G_z = feats_denorm(G_z)
  G_z = tf.identity(G_z, name='G_z')
  G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
  step = tf.train.get_or_create_global_step()
  saver = tf.train.Saver(var_list=G_vars + [step])

  tf.train.write_graph(tf.get_default_graph(), args.train_dir, 'infer.pbtxt')

  tf.train.export_meta_graph(
    filename=os.path.join(args.train_dir, 'infer.meta'),
    clear_devices=True,
    saver_def=saver.as_saver_def())

  tf.reset_default_graph()


"""
  Computes inception score every time a checkpoint is saved
"""
def incept(args):
  incept_dir = os.path.join(args.train_dir, 'incept')
  if not os.path.isdir(incept_dir):
    os.makedirs(incept_dir)

  # Create GAN graph
  z = tf.placeholder(tf.float32, [None, Z_DIM])
  with tf.variable_scope('G'):
    G = MelspecGANGenerator()
    G_z = G(z, training=False)
  G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
  step = tf.train.get_or_create_global_step()
  gan_saver = tf.train.Saver(var_list=G_vars + [step], max_to_keep=1)

  # Load or generate latents
  z_fp = os.path.join(incept_dir, 'z.pkl')
  if os.path.exists(z_fp):
    with open(z_fp, 'rb') as f:
      _zs = pickle.load(f)
  else:
    zs = tf.random.normal([args.incept_n, Z_DIM], dtype=tf.float32)
    with tf.Session() as sess:
      _zs = sess.run(zs)
    with open(z_fp, 'wb') as f:
      pickle.dump(_zs, f)

  # Load classifier graph
  incept_graph = tf.Graph()
  with incept_graph.as_default():
    incept_saver = tf.train.import_meta_graph(args.incept_metagraph_fp)
  incept_x = incept_graph.get_tensor_by_name('x:0')
  incept_preds = incept_graph.get_tensor_by_name('scores:0')
  incept_sess = tf.Session(graph=incept_graph)
  incept_saver.restore(incept_sess, args.incept_ckpt_fp)

  # Create summaries
  summary_graph = tf.Graph()
  with summary_graph.as_default():
    incept_mean = tf.placeholder(tf.float32, [])
    incept_std = tf.placeholder(tf.float32, [])
    summaries = [
        tf.summary.scalar('incept_mean', incept_mean),
        tf.summary.scalar('incept_std', incept_std)
    ]
    summaries = tf.summary.merge(summaries)
  summary_writer = tf.summary.FileWriter(incept_dir)

  # Loop, waiting for checkpoints
  ckpt_fp = None
  _best_score = 0.
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      print('Incept: {}'.format(latest_ckpt_fp))

      sess = tf.Session()

      gan_saver.restore(sess, latest_ckpt_fp)

      _step = sess.run(step)

      _G_z_feats = []
      for i in range(0, args.incept_n, 100):
        _G_z_feats.append(sess.run(G_z, {z: _zs[i:i+100]}))
      _G_z_feats = np.concatenate(_G_z_feats, axis=0)
      _G_zs = []
      for i, _G_z in enumerate(_G_z_feats):
        _G_z = feats_denorm(_G_z).astype(np.float64)
        _audio = r9y9_melspec_to_waveform(_G_z, fs=args.data_sample_rate, waveform_len=16384)
        if i == 0:
          out_fp = os.path.join(incept_dir, '{}.wav'.format(str(_step).zfill(9)))
          save_as_wav(out_fp, args.data_sample_rate, _audio)
        _G_zs.append(_audio[:, 0, 0])

      _preds = []
      for i in range(0, args.incept_n, 100):
        _preds.append(incept_sess.run(incept_preds, {incept_x: _G_zs[i:i+100]}))
      _preds = np.concatenate(_preds, axis=0)

      # Split into k groups
      _incept_scores = []
      split_size = args.incept_n // args.incept_k
      for i in range(args.incept_k):
        _split = _preds[i * split_size:(i + 1) * split_size]
        _kl = _split * (np.log(_split) - np.log(np.expand_dims(np.mean(_split, 0), 0)))
        _kl = np.mean(np.sum(_kl, 1))
        _incept_scores.append(np.exp(_kl))

      _incept_mean, _incept_std = np.mean(_incept_scores), np.std(_incept_scores)

      # Summarize
      with tf.Session(graph=summary_graph) as summary_sess:
        _summaries = summary_sess.run(summaries, {incept_mean: _incept_mean, incept_std: _incept_std})
      summary_writer.add_summary(_summaries, _step)

      # Save
      if _incept_mean > _best_score:
        gan_saver.save(sess, os.path.join(incept_dir, 'best_score'), _step)
        _best_score = _incept_mean

      sess.close()

      print('Done')

      ckpt_fp = latest_ckpt_fp

    time.sleep(1)

  incept_sess.close()


if __name__ == '__main__':
  from argparse import ArgumentParser
  import glob
  import os

  parser = ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'incept'])
  parser.add_argument('train_dir', type=str)

  data_args = parser.add_argument_group('Data')
  data_args.add_argument('--data_cfg', type=str,
          help='Path to dataset configuration')
  data_args.add_argument('--data_dir', type=str,
          help='Data directory containing *only* audio files to load')
  data_args.add_argument('--data_prefetch_gpu_num', type=int,
	  help='If nonnegative, prefetch examples to this GPU (Tensorflow device num)')

  train_args = parser.add_argument_group('Train')
  train_args.add_argument('--train_ckpt_every_nsecs', type=int)
  train_args.add_argument('--train_summary_every_nsecs', type=int)

  incept_args = parser.add_argument_group('Incept')
  incept_args.add_argument('--incept_metagraph_fp', type=str,
      help='Inference model for inception score')
  incept_args.add_argument('--incept_ckpt_fp', type=str,
      help='Checkpoint for inference model')
  incept_args.add_argument('--incept_n', type=int,
      help='Number of generated examples to test')
  incept_args.add_argument('--incept_k', type=int,
      help='Number of groups to test')

  parser.set_defaults(
      mode=None,
      train_dir=None,
      data_cfg='../../datacfg/sc09.txt',
      data_dir=None,
      data_prefetch_gpu_num=0,
      train_ckpt_every_nsecs=600,
      train_summary_every_nsecs=300,
      incept_metagraph_fp='./eval/inception/infer.meta',
      incept_ckpt_fp='./eval/inception/best_acc-103005',
      incept_n=5000,
      incept_k=10)

  args = parser.parse_args()

  with open(args.data_cfg, 'r') as f:
    for l in f.read().strip().splitlines():
      k, v = l.split(',')
      try:
        v = int(v)
      except:
        v = float(v)
      setattr(args, 'data_' + k, v)

  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  if args.mode == 'train':
    fps = glob.glob(os.path.join(args.data_dir, '*'))
    if len(fps) == 0:
      raise ValueError('Found no audio files in {}'.format(args.data_dir))
    print('Found {} audio files'.format(len(fps)))
    infer(args)
    train(fps, args)
  elif args.mode == 'incept':
    incept(args)
