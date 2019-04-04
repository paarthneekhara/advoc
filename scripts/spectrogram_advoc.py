# This script vocodes a directory of spectrograms into waveforms using adversarial vocoding.

if __name__ == '__main__':
  from argparse import ArgumentParser
  import glob
  import numpy as np
  import os
  from tqdm import tqdm

  from advoc.audioio import save_as_wav
  from advoc.spectral import r9y9_melspec_to_waveform

  parser = ArgumentParser()

  parser.add_argument('--spec_dir', type=str, required=True,
      help='Directory of audio files')
  parser.add_argument('--out_dir', type=str, required=True,
      help='Directory for spectrograms')
  parser.add_argument('--model_ckpt', type=str,
      help='Adversarial vocoder checkpoint')

  parser.set_defaults(
      spec_dir=None,
      out_dir=None,
      model_ckpt=None)

  args = parser.parse_args()

  if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

  heuristic = False
  if args.model_ckpt is None:
    print('Warning: Model checkpoint not specified, using pseudoinverse+LWS heuristic to vocode')
    heuristic = True

  spec_fps = glob.glob(os.path.join(args.spec_dir, '*.npy'))
  for i, spec_fp in tqdm(enumerate(spec_fps)):
    spec_fn = os.path.splitext(os.path.split(spec_fp)[1])[0]
    wave_fn = spec_fn + '.wav'
    wave_fp = os.path.join(args.out_dir, wave_fn)

    spec = np.load(spec_fp)

    if heuristic:
      wave = r9y9_melspec_to_waveform(spec)
    else:
      raise NotImplementedError('Paarth fix this')

    save_as_wav(wave_fp, 22050, wave)
