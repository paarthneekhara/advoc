import os
import tempfile
import unittest

import numpy as np

from advoc.audioio import decode_audio, save_as_wav


AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio')
WAV_MONO = os.path.join(AUDIO_DIR, 'mono.wav')
WAV_STEREO = os.path.join(AUDIO_DIR, 'stereo.wav')
MP3_MONO = os.path.join(AUDIO_DIR, 'mono.mp3')
MP3_STEREO = os.path.join(AUDIO_DIR, 'stereo.mp3')


class TestSpectralMethods(unittest.TestCase):

  def test_scipy_decode_audio(self):
    fs, x = decode_audio(WAV_MONO, fastwav=True)
    self.assertEqual(fs, 44100, 'incorrect sample rate')
    self.assertEqual(x.shape, (164864, 1, 1), 'incorrect shape')
    self.assertAlmostEqual(x.min(), -0.47483748, 8, 'incorrect min value')
    self.assertAlmostEqual(x.max(), 0.39728996, 8, 'incorrect max value')

    with self.assertRaises(ValueError, msg='should not be able to resample'):
      decode_audio(WAV_MONO, fs=22050, fastwav=True)

    fs, x = decode_audio(WAV_MONO, normalize=True, fastwav=True)
    self.assertAlmostEqual(np.abs(x).max(), 1., 8, 'incorrect peak value')

    fs, x = decode_audio(WAV_STEREO, fastwav=True)
    self.assertEqual(x.shape, (164864, 1, 2), 'incorrect shape')

    fs, x = decode_audio(WAV_STEREO, mono=True, fastwav=True)
    self.assertEqual(x.shape, (164864, 1, 1), 'incorrect shape')

    with self.assertRaises(ValueError, msg='should not be able to decode mp3'):
      decode_audio(MP3_MONO, fastwav=True)


  def test_librosa_decode_audio(self):
    fs, x = decode_audio(WAV_MONO)
    self.assertEqual(fs, 44100, 'incorrect sample rate')
    self.assertEqual(x.shape, (164864, 1, 1), 'incorrect shape')
    self.assertAlmostEqual(x.min(), -0.474823, 6, 'incorrect min value')
    self.assertAlmostEqual(x.max(), 0.397278, 6, 'incorrect max value')

    fs, x = decode_audio(WAV_MONO, fs=22050)
    self.assertEqual(fs, 22050, 'incorrect sample rate')
    self.assertEqual(x.shape, (82432, 1, 1), 'incorrect shape')

    fs, x = decode_audio(MP3_MONO)
    self.assertEqual(fs, 44100, 'incorrect sample rate')
    self.assertEqual(x.shape, (164864, 1, 1), 'incorrect shape')
    self.assertAlmostEqual(x.min(), -0.47714233, 8, 'incorrect min value')
    self.assertAlmostEqual(x.max(), 0.39419556, 8, 'incorrect max value')

    fs, x = decode_audio(MP3_MONO, normalize=True)
    self.assertAlmostEqual(np.abs(x).max(), 1., 8, 'incorrect peak value')

    fs, x = decode_audio(MP3_STEREO)
    self.assertEqual(x.shape, (164864, 1, 2), 'incorrect shape')

    fs, x = decode_audio(MP3_STEREO, mono=True)
    self.assertEqual(x.shape, (164864, 1, 1), 'incorrect shape')


  def test_save_as_wav(self):
    fs, x = decode_audio(WAV_MONO, fastwav=True)

    with tempfile.NamedTemporaryFile() as tf:
      with self.assertRaises(ValueError, msg='should not be able to save incorrect dims'):
        save_as_wav(tf.name, fs, x[:, 0])

      with self.assertRaises(ValueError, msg='should not be able to save features'):
        save_as_wav(tf.name, fs, np.concatenate([x, x], axis=1))

      with self.assertRaises(NotImplementedError, msg='should not be able to save stereo'):
        save_as_wav(tf.name, fs, np.concatenate([x, x], axis=2))

      save_as_wav(tf.name, fs, x)
      fs2, x2 = decode_audio(tf.name, fastwav=True)

      self.assertTrue(np.array_equal(x, x2), 'should be lossless after save')


if __name__ == '__main__':
  unittest.main()
