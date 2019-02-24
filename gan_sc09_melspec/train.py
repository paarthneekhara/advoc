import numpy as np
import tensorflow as tf

from advoc.loader import decode_extract_and_batch
from util import feats_to_uint8_img


TRAIN_BATCH_SIZE = 64
Z_DIM = 100

def train(fps, args):
  # Load data
  with tf.name_scope('loader'):
    x, x_audio = decode_extract_and_batch(
        fps=fps,
        batch_size=TRAIN_BATCH_SIZE,
        subseq_len=64,
        audio_fs=16000,
        audio_mono=True,
        audio_normalize=True,
        decode_fastwav=True,
        decode_parallel_calls=4,
        extract_type='r9y9_melspec',
        extract_parallel_calls=4,
        repeat=True,
        shuffle=True,
        shuffle_buffer_size=512,
        subseq_randomize_offset=False,
        subseq_overlap_ratio=0,
        subseq_pad_end=True,
        prefetch_size=TRAIN_BATCH_SIZE * 4,
        gpu_num=0)

  # Data summaries
  tf.summary.audio('x_audio', x_audio[:, :, 0], 16000)
  tf.summary.image('x', feats_to_uint8_img(x))

  # Make z vector
  z = tf.random.normal([TRAIN_BATCH_SIZE, Z_DIM], dtype=tf.float32)

  # Make generator
  with tf.variable_scope('G'):
    G_z = MelspecGANGenerator(z, train=True)
  G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

  # Summarize G_z
  # TODO: approximate invert to audio
  tf.summary.image(G_z, feats_to_uint8_img(G_z))

  # Make real discriminator
  with tf.name_scope('D_x'), tf.variable_scope('D'):
    D_x = MelspecGANDiscriminator(x, train=True)

  # Make fake discriminator
  with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
    D_G_z = MelspecGANDiscriminator(G_z, train=True)

  # Create loss
  tf.summary.scalar('G_loss', G_loss)
  tf.summary.scalar('D_loss', D_loss)

  # Create opt

  # Create training ops
  G_train_op = G_opt.minimize(G_loss, var_list=G_vars,
      global_step=tf.train.get_or_create_global_step())
  D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

  # Train
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=60,
      save_summaries_secs=5) as sess:
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
