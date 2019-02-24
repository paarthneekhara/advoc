import os
import pickle
import unittest

import numpy as np

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

    X = spectral.stft(x, 1024, 256, pad='center')
    self.assertEqual(X.dtype, np.complex128)
    self.assertEqual(X.shape, (64, 513, 1), 'invalid shape')

    X = spectral.stft(x, 1024, 256, pad=None)
    self.assertEqual(X.shape, (60, 513, 1), 'invalid shape')

    x = np.pad(x, [[0, 384], [0, 0], [0, 0]], 'constant')
    X = spectral.stft(x, 1024, 256, pad='center')
    self.assertEqual(X.shape, (65, 513, 1), 'invalid shape')

    X_mag = np.abs(X)
    self.assertEqual(X_mag.dtype, np.float64)
    self.assertAlmostEqual(np.sum(X_mag), 2150.650, 3, 'invalid spec')
    self.assertAlmostEqual(np.sum(X_mag[33]), 36.699, 3, 'invalid spec')
    self.assertAlmostEqual(np.sum(X_mag[40]), 33.598, 3, 'invalid spec')


  def test_tacotron2(self):
    melspec = spectral.waveform_to_tacotron2_feats(self.wav_mono_24)
    self.assertEqual(melspec.dtype, np.float64)
    self.assertEqual(melspec.shape, (301, 80, 1), 'invalid shape')

    self.assertAlmostEqual(np.sum(melspec), 131.469, 3, 'invalid spec')
    self.assertAlmostEqual(np.sum(melspec[200]), 0.895, 3, 'invalid spec')
    self.assertAlmostEqual(np.sum(melspec[40]), 0., 3, 'invalid spec')


  def test_r9y9(self):
    with open('/home/cdonahue/advoc/advoc/tests/audio/mono_22k.pkl', 'rb') as f:
      wav_ref = pickle.load(f)
    melspec = spectral.waveform_to_r9y9_feats(self.wav_mono_22)
    self.assertEqual(melspec.dtype, np.float64)
    self.assertEqual(melspec.shape, (323, 80, 1), 'invalid shape')

    # This file came directly from the r9y9/wavenet_vocoder codebase.
    with open(WAV_MONO_R9Y9, 'rb') as f:
      r9y9_melspec = pickle.load(f)

    # R9Y9 code pads by ((nfft-nhop) // nhop) == (3 * nhop) on both sides.
    # We pad by (nfft // 2) == (2 * nhop) on both sides.
    # Therefore, we should skip comparison of first and last frame.
    r9y9_melspec = np.swapaxes(r9y9_melspec, 0, 1)[1:-1, :, np.newaxis]

    self.assertTrue(np.array_equal(melspec, r9y9_melspec), 'not equal r9y9')


if __name__ == '__main__':
  unittest.main()
