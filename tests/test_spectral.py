import os
import pickle
import unittest

import numpy as np
from scipy.signal import hilbert as sphilbert
import tensorflow as tf

import advoc.audioio as audioio
import advoc.spectral as spectral


AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
WAV_SC09 = os.path.join(AUDIO_DIR, 'sc09.wav')
WAV_MONO = os.path.join(AUDIO_DIR, 'mono.wav')
WAV_MONO_R9Y9 = os.path.join(AUDIO_DIR, 'mono_22k_r9y9.pkl')


class TestSpectralModule(unittest.TestCase):

  def setUp(self):
    _, self.wav_sc09_16 = audioio.decode_audio(WAV_SC09, fastwav=True)
    _, self.wav_mono_22 = audioio.decode_audio(WAV_MONO, fs=22050)
    _, self.wav_mono_24 = audioio.decode_audio(WAV_MONO, fs=24000)


  def test_stft(self):
    x = self.wav_sc09_16
    self.assertEqual(x.shape, (16000, 1, 1), 'invalid wav length')

    X = spectral.stft(x, 1024, 256, pad_end=True)
    self.assertEqual(X.dtype, np.complex128)
    self.assertEqual(X.shape, (63, 513, 1), 'invalid shape')

    X = spectral.stft(x, 1024, 256, pad_end=False)
    self.assertEqual(X.shape, (60, 513, 1), 'invalid shape')

    x = np.pad(x, [[0, 384], [0, 0], [0, 0]], 'constant')
    X = spectral.stft(x, 1024, 256, pad_end=True)
    self.assertEqual(X.shape, (64, 513, 1), 'invalid shape')

    X_mag = np.abs(X)
    self.assertEqual(X_mag.dtype, np.float64)
    self.assertAlmostEqual(np.sum(X_mag), 2148.755, 3, 'invalid spec')
    self.assertAlmostEqual(np.sum(X_mag[33]), 55.455, 3, 'invalid spec')
    self.assertAlmostEqual(np.sum(X_mag[40]), 20.347, 3, 'invalid spec')


  def test_stft_tf(self):
    with tf.Graph().as_default():
      x = tf.placeholder(tf.float32, [None, None, 1, None])
      X = spectral.stft_tf(x, 1024, 256, pad_end=True)

      self.assertEqual(X.dtype, tf.complex64)
      self.assertEqual(X.get_shape().as_list(), [None, None, 513, None], 'invalid shape')

      config = tf.ConfigProto(device_count={'GPU': 0})
      with tf.Session(config=config) as sess:
        _x = self.wav_sc09_16[np.newaxis]
        self.assertEqual(_x.shape, (1, 16000, 1, 1), 'invalid wav length')

        _X = sess.run(X, {x: _x})
        self.assertEqual(_X.dtype, np.complex64)
        self.assertEqual(_X.shape, (1, 63, 513, 1), 'invalid shape')

        _x = np.pad(_x, [[0, 0], [0, 384], [0, 0], [0, 0]], 'constant')
        _x = np.concatenate([self.wav_mono_22[np.newaxis, :16384], _x], axis=0)
        _X = sess.run(X, {x: _x})
        self.assertEqual(_X.shape, (2, 64, 513, 1), 'invalid shape')

        _X_mag = np.abs(_X)
        self.assertEqual(_X_mag.dtype, np.float32)
        self.assertAlmostEqual(np.sum(_X_mag[0]), 160.600, 3, 'invalid spec')
        self.assertAlmostEqual(np.sum(_X_mag[1]), 2148.754, 3, 'invalid spec')
        self.assertAlmostEqual(np.sum(_X_mag[1, 33]), 55.455, 3, 'invalid spec')
        self.assertAlmostEqual(np.sum(_X_mag[1, 40]), 20.347, 3, 'invalid spec')


  def test_tacotron2(self):
    melspec = spectral.waveform_to_tacotron2_melspec(self.wav_mono_24)
    self.assertEqual(melspec.dtype, np.float64)
    self.assertEqual(melspec.shape, (300, 80, 1), 'invalid shape')

    self.assertAlmostEqual(np.sum(melspec), 131.469, 3, 'invalid spec')
    self.assertAlmostEqual(np.sum(melspec[200]), 0.644, 3, 'invalid spec')
    self.assertAlmostEqual(np.sum(melspec[40]), 0., 3, 'invalid spec')


  def test_r9y9(self):
    self.assertEqual(self.wav_mono_22.shape, (82432, 1, 1), 'invalid shape')

    melspec = spectral.waveform_to_r9y9_melspec(self.wav_mono_22)
    self.assertEqual(melspec.dtype, np.float64)
    self.assertEqual(melspec.shape, (322, 80, 1), 'invalid shape')

    # This array came directly from the r9y9/wavenet_vocoder codebase.
    # Its shape is [80, 325].
    with open(WAV_MONO_R9Y9, 'rb') as f:
      r9y9_melspec = pickle.load(f)

    # R9Y9 code pads by ((nfft-nhop) // nhop) == (3 * nhop) on both sides.
    # We pad by 3 frames at end.
    # Therefore, we should skip comparison of first 3 frames.
    r9y9_melspec = np.swapaxes(r9y9_melspec, 0, 1)[3:, :, np.newaxis]

    self.assertTrue(np.array_equal(melspec, r9y9_melspec), 'not equal r9y9')


  def test_r9y9_tf(self):
    with tf.Graph().as_default():
      x = tf.placeholder(tf.float32, [None, None, 1, None])
      melspec = spectral.waveform_to_r9y9_melspec_tf(x)

      self.assertEqual(melspec.dtype, tf.float32)
      self.assertEqual(melspec.get_shape().as_list(), [None, None, 80, None], 'invalid shape')

      config = tf.ConfigProto(device_count={'GPU': 0})
      with tf.Session(config=config) as sess:
        _x = self.wav_mono_22[np.newaxis]
        np.random.seed(0)
        _noise = np.random.uniform(low=-1, high=1, size=_x.shape)
        _x = np.concatenate([_noise, _x], axis=0)

        self.assertEqual(_x.shape, (2, 82432, 1, 1), 'invalid shape')

        _melspec = sess.run(melspec, {x: _x})

        self.assertEqual(_melspec.dtype, np.float32)
        self.assertEqual(_melspec.shape, (2, 322, 80, 1), 'invalid shape')
        self.assertAlmostEqual(np.sum(_melspec[0]), 18319.934, 3, 'incorrect sum')
        self.assertAlmostEqual(np.sum(_melspec[1]), 5121.489, 3, 'incorrect sum')

        # This array came directly from the r9y9/wavenet_vocoder codebase.
        # Its shape is [80, 325].
        with open(WAV_MONO_R9Y9, 'rb') as f:
          r9y9_melspec = pickle.load(f)

        # R9Y9 code pads by ((nfft-nhop) // nhop) == (3 * nhop) on both sides.
        # We pad by 3 frames at end.
        # Therefore, we should skip comparison of first 3 frames.
        r9y9_melspec = np.swapaxes(r9y9_melspec, 0, 1)[np.newaxis, 3:, :, np.newaxis]

        err = np.sum(np.abs(_melspec[1:].astype(np.float64) - r9y9_melspec))
        self.assertAlmostEqual(err, 0.00731311, 8, 'not equal r9y9')


  def test_inverse_r9y9(self):
    self.assertEqual(self.wav_mono_22.shape, (82432, 1, 1), 'invalid shape')

    melspec = spectral.waveform_to_r9y9_melspec(self.wav_mono_22)
    inv_melspec = spectral.r9y9_melspec_to_waveform(melspec, waveform_len=82432)
    self.assertEqual(inv_melspec.shape, self.wav_mono_22.shape, 'invalid shape')

    x_env = np.abs(sphilbert(self.wav_mono_22[:, 0, 0]))
    x_inv_env = np.abs(sphilbert(inv_melspec[:, 0, 0]))
    env_l1 = np.mean(np.abs(x_env - x_inv_env))
    self.assertAlmostEqual(env_l1, 0.01737, 4, 'bad envelope after inverse')


if __name__ == '__main__':
  unittest.main()
