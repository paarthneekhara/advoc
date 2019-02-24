import tensorflow as tf

def feats_to_uint8_img(x):
  x = tf.image.rot90(x)
  x *= 255.
  x = tf.clip_by_value(x, 0., 255.)
  x = tf.cast(x, tf.uint8)
  return x
