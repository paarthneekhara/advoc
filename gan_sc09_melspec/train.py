import numpy as np
import tensorflow as tf

from advoc.loader import decode_extract_and_batch
from conv2d import MelspecGANGenerator, MelspecGANDiscriminator
from util import feats_to_uint8_img, feats_to_approx_audio


TRAIN_BATCH_SIZE = 64
TRAIN_LOSS = 'wgangp'
Z_DIM = 100
FS = 16000
NUM_DISC_UPDATES = 5

def train(fps, args):
  # Load data
  with tf.name_scope('loader'):
    x, x_audio = decode_extract_and_batch(
        fps=fps,
        batch_size=TRAIN_BATCH_SIZE,
        subseq_len=64,
        audio_fs=FS,
        audio_mono=True,
        audio_normalize=True,
        decode_fastwav=True,
        decode_parallel_calls=8,
        extract_type='r9y9_melspec',
        extract_parallel_calls=8,
        repeat=True,
        shuffle=True,
        shuffle_buffer_size=512,
        subseq_randomize_offset=False,
        subseq_overlap_ratio=0,
        subseq_pad_end=True,
        prefetch_size=TRAIN_BATCH_SIZE * 8,
        gpu_num=0)

  # Data summaries
  tf.summary.audio('x_audio', x_audio[:, :, 0], FS)
  tf.summary.image('x', feats_to_uint8_img(x))
  tf.summary.audio('x_inv_audio',
      feats_to_approx_audio(x, FS, 16384, n=3)[:, :, 0], FS)

  # Make z vector
  z = tf.random.normal([TRAIN_BATCH_SIZE, Z_DIM], dtype=tf.float32)

  # Make generator
  with tf.variable_scope('G'):
    G = MelspecGANGenerator()
    G_z = G(z, training=True)
  G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

  # Summarize G_z
  # TODO: approximate invert to audio
  tf.summary.image('G_z', feats_to_uint8_img(G_z))
  tf.summary.audio('G_z_inv_audio', feats_to_approx_audio(G_z, FS, 16384, n=3)[:, :, 0], FS)

  # Make real discriminator
  D = MelspecGANDiscriminator()
  with tf.name_scope('D_x'), tf.variable_scope('D'):
    D_x = D(x, training=True)

  # Make fake discriminator
  with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
    D_G_z = D(G_z, training=True)

  # Create loss
  if TRAIN_LOSS == 'wgangp':
    G_loss = -tf.reduce_mean(D_G_z)
    D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

    alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1, 1], minval=0., maxval=1.)
    differences = G_z - x
    interpolates = x + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
      D_interp = D(interpolates, training=True)

    LAMBDA = 10
    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    D_loss += LAMBDA * gradient_penalty
  else:
    raise ValueError()

  tf.summary.scalar('G_loss', G_loss)
  tf.summary.scalar('D_loss', D_loss)

  # Create opt
  if TRAIN_LOSS == 'wgangp':
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
      for i in range(NUM_DISC_UPDATES):
        sess.run(D_train_op)

      sess.run(G_train_op)


if __name__ == '__main__':
  from argparse import ArgumentParser
  import glob
  import os

  parser = ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train'])
  parser.add_argument('train_dir', type=str)

  parser.add_argument('--data_dir', type=str, required=True)

  parser.add_argument('--train_ckpt_every_nsecs', type=int)
  parser.add_argument('--train_summary_every_nsecs', type=int)

  parser.set_defaults(
      mode=None,
      train_dir=None,
      data_dir=None,
      train_ckpt_every_nsecs=600,
      train_summary_every_nsecs=300)

  args = parser.parse_args()

  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  fps = glob.glob(os.path.join(args.data_dir, '*.wav'))
  print('Found {} audio files'.format(len(fps)))

  if args.mode == 'train':
    train(fps, args)
