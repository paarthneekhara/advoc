import advoc
import tensorflow as tf

class SpectralUtil(object):
  NFFT = 1024
  NHOP = 256
  FMIN = 125.
  FMAX = 7600.
  NMELS = 80
  fs = 16000

  def __init__(self):
    meltrans = advoc.spectral.create_mel_filterbank(
            self.fs, self.NFFT, fmin=self.FMIN, fmax=self.FMAX, n_mels=self.NMELS)
    invmeltrans = advoc.spectral.create_inverse_mel_filterbank(
            self.fs, self.NFFT, fmin=self.FMIN, fmax=self.FMAX, n_mels=self.NMELS)

    self.meltrans = tf.constant(meltrans, dtype = 'float32')
    self.invmeltrans = tf.constant(invmeltrans, dtype = 'float32')

  def mag_to_mel_linear_spec(self, mag_spec):
    linear_mel =  tf.expand_dims(
      tf.tensordot(mag_spec[:,:,:,0], tf.transpose(self.meltrans), axes = 1 ), -1)
    return linear_mel

  def mel_linear_to_mag_spec(self, mel_spec, transform = 'inverse'):
    if transform == 'inverse':
      transform_mat = tf.transpose(self.invmeltrans)
    elif transform == 'transposed':
      transform_mat = meltrans
    else:
      raise NotImplementedError()
    mag_spec =  tf.expand_dims(
      tf.tensordot(mel_spec[:,:,:,0], transform_mat, axes = 1 ), -1)
    return mag_spec