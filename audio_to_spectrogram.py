if __name__ == '__main__':
  from argparse import ArgumentParser
  import glob
  import numpy as np
  import os
  from tqdm import tqdm

  from advoc.audioio import decode_audio
  from advoc.spectral import waveform_to_r9y9_melspec

  parser = ArgumentParser()

  parser.add_argument('--data_dir', type=str, required=True,
      help='Directory of audio files')
  parser.add_argument('--out_dir', type=str, required=True,
      help='Directory for spectrograms')
  parser.add_argument('--data_fast_wav',
      action='store_true', dest='data_fast_wav',
      help='If set, provides faster loading of standard WAV files via scipy')

  parser.set_defaults(
      data_dir=None,
      out_dir=None,
      data_fast_wav=False)

  args = parser.parse_args()

  if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

  wave_fps = glob.glob(os.path.join(args.data_dir, '*'))
  for i, wave_fp in tqdm(enumerate(wave_fps)):
    wave_fn = os.path.splitext(os.path.split(wave_fp)[1])[0]
    spec_fn = wave_fn + '.npy'
    spec_fp = os.path.join(args.out_dir, spec_fn)

    _, wave = decode_audio(
        wave_fp,
        fs=22050,
        fastwav=args.data_fast_wav,
        mono=True,
        normalize=True)

    spec = waveform_to_r9y9_melspec(wave)

    np.save(spec_fp, spec)
