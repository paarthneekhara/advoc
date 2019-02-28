import tensorflow as tf

def best_shape(t, axis=None):
  """Gets static shape if available, otherwise dynamic.

  Args:
    t: Tensor in question.
    axis: None if requesting entire shape, otherwise the axis in question.

  Returns:
    Python list containing (possibly a mixture of) ints or tf.Tensor.
  """
  if axis is None:
    ndims = t.get_shape().ndims
    if ndims is None:
      raise ValueError('Cannot run on tensor with dynamic ndims')
    dims = []
    for i in range(ndims):
      try:
        dim = int(t.get_shape()[i])
      except:
        dim = tf.shape(t)[i]
      dims.append(dim)
    return dims
  else:
    try:
      dim = int(t.get_shape()[axis])
    except:
      dim = tf.shape(t)[i]
    return dim
