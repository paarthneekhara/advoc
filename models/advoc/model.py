class Model(object):
  def __init__(self, mode, *args, **kwargs):
    self.mode = mode

  def __call__(self):
    raise Exception('Abstract method')

  def train_loop(self):
    raise Exception('Abstract method')

  def eval_ckpt(self, ckpt_fp):
    raise Exception('Abstract method')


class Modes(object):
  TRAIN = 'train'
  EVAL = 'eval'
  INFER = 'infer'
