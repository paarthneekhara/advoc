# This script creates a directory of spectrograms using a trained melspec GAN.

if __name__ == '__main__':
  from argparse import ArgumentParser
  import glob
  import os

  import numpy as np
  import tensorflow as tf
  from tqdm import tqdm

  parser = ArgumentParser()

  parser.add_argument('--out_dir', type=str, required=True,
      help='Directory for spectrograms')
  parser.add_argument('--ckpt_fp', type=str,
      help='Adversarial vocoder checkpoint')
  parser.add_argument('--meta_fp', type=str,
      help='Meta graph filepath')
  parser.add_argument('--n', type=int,
      help='Total number of spectrograms to generate')
  parser.add_argument('--b', type=int,
      help='Number of spectrograms to generate per batch')

  parser.set_defaults(
      out_dir=None,
      ckpt_fp=None,
      meta_fp='../models/melspecgan/infer.meta',
      n=1000,
      b=10)

  args = parser.parse_args()

  if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

  saver = tf.train.import_meta_graph(args.meta_fp)
  step = tf.train.get_or_create_global_step()

  g = tf.get_default_graph()
  samp_z_n = g.get_tensor_by_name('samp_z_n:0')
  samp_z = g.get_tensor_by_name('samp_z:0')
  z = g.get_tensor_by_name('z:0')
  G_z = g.get_tensor_by_name('G_z:0')

  with tf.Session() as sess:
    saver.restore(sess, args.ckpt_fp)
    print('Restored from step {}'.format(sess.run(step)))

    for i in tqdm(range(0, args.n, args.b)):
      _z = sess.run(samp_z, {samp_z_n: args.b})
      _G_z = sess.run(G_z, {z: _z})
      for j, _s in enumerate(_G_z):
        _s = _s.astype(np.float32)
        out_fp = os.path.join(args.out_dir, '{}.npy'.format(str(j+i).zfill(9)))
        np.save(out_fp, _s)
