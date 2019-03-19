class AudioModel(object):
  def __init__(self,
      mode,
      spectral,
      train_batch_size,
      subseq_len=24,
      audio_fs=22050):
    self.mode = mode
    self.extract_type = 'r9y9_melspec' if spectral else None
    self.train_batch_size = train_batch_size
    self.subseq_len = subseq_len
    self.audio_fs = audio_fs

  def __call__(self, x_spec, x_wave):
    raise Exception('Abstract method')

  def get_global_variables(self):
    raise Exception('Abstract method')

  def pretrain_hook(self, sess):
    pass

  def train_loop(self, sess):
    raise Exception('Abstract method')

  def eval_ckpt(self, sess):
    raise Exception('Abstract method')


class Modes(object):
  TRAIN = 'train'
  EVAL = 'eval'
  INFER = 'infer'
