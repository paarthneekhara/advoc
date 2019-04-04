import numpy as np
import tensorflow as tf

from conv2d import MelspecGANGenerator, MelspecGANDiscriminator

from util import feats_denorm

Z_DIM = 100
OUT_FP = 'infer.meta'

zgen_n = tf.placeholder(tf.int32, [], name='samp_z_n')
zgen = tf.random.normal([zgen_n, Z_DIM], dtype=tf.float32, name='samp_z')

z = tf.placeholder(tf.float32, [None, Z_DIM], name='z')
with tf.variable_scope('G'):
  G = MelspecGANGenerator()
  G_z = G(z, training=False)
G_z = feats_denorm(G_z)
G_z = tf.identity(G_z, name='G_z')
print(G_z)
G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
print(G_vars)
step = tf.train.get_or_create_global_step()
saver = tf.train.Saver(var_list=G_vars + [step])

tf.train.write_graph(tf.get_default_graph(), './', 'infer.pbtxt')

tf.train.export_meta_graph(
  filename=OUT_FP,
  clear_devices=True,
  saver_def=saver.as_saver_def())
