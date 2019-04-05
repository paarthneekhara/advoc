# This script vocodes a directory of spectrograms into waveforms using adversarial vocoding.

if __name__ == '__main__':
  from argparse import ArgumentParser
  import glob
  import numpy as np
  import os
  from tqdm import tqdm
  import tensorflow as tf
  from advoc.audioio import save_as_wav
  from advoc.spectral import r9y9_melspec_to_waveform, magspec_to_waveform_lws
  from advoc.spectral import create_inverse_mel_filterbank

  #TODO move to advoc.spectral
  def tacotron_mel_to_mag(X_mel_dbnorm, invmeltrans):
    norm_min_level_db = -100
    norm_ref_level_db = 20
    
    X_mel_db = (X_mel_dbnorm * -norm_min_level_db) + norm_min_level_db
    X_mel = np.power(10, (X_mel_db + norm_ref_level_db) / 20)
    X_mag = np.dot(X_mel, invmeltrans.T)
    return X_mag

  parser = ArgumentParser()

  parser.add_argument('--spec_dir', type=str, required=True,
      help='Directory of audio files')
  parser.add_argument('--out_dir', type=str, required=True,
      help='Directory for spectrograms')
  parser.add_argument('--model_ckpt', type=str,
      help='Adversarial vocoder checkpoint')
  parser.add_argument('--meta_fp', type=str,
      help='Met graph filepath')
  parser.add_argument('--fs', type=int,
      help='sampling rate')
  parser.add_argument('--subseq_len', type=int,
      help="model subseq length")

  parser.set_defaults(
      spec_dir=None,
      out_dir=None,
      model_ckpt=None,
      meta_fp=None,
      fs=22050,
      subseq_len=256
      )

  args = parser.parse_args()

  if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

  heuristic = False
  if args.model_ckpt is None:
    print('Warning: Model checkpoint not specified, using pseudoinverse+LWS heuristic to vocode')
    heuristic = True
  else:
    gen_graph = tf.Graph()
    with gen_graph.as_default():
      gan_saver = tf.train.import_meta_graph(args.meta_fp)
    gen_sess = tf.Session(graph=gen_graph)
    print("Restoring")
    gan_saver.restore(gen_sess, args.model_ckpt)
    gen_mag_spec = gen_graph.get_tensor_by_name('generator/decoder_1/strided_slice_1:0')
    x_mag_input = gen_graph.get_tensor_by_name('ExpandDims_1:0')

  inv_mel_filterbank = create_inverse_mel_filterbank(
      args.fs, 1024, fmin=125, fmax=7600, n_mels=80)

  spec_fps = glob.glob(os.path.join(args.spec_dir, '*.npy'))
  for i, spec_fp in tqdm(enumerate(spec_fps)):
    spec_fn = os.path.splitext(os.path.split(spec_fp)[1])[0]
    wave_fn = spec_fn + '.wav'
    wave_fp = os.path.join(args.out_dir, wave_fn)

    spec = np.load(spec_fp)

    if heuristic:
      wave = r9y9_melspec_to_waveform(spec)
    else:
      subseq_len = args.subseq_len
      X_mag = tacotron_mel_to_mag(spec[:,:,0], inv_mel_filterbank)
      x_mag_original_length = X_mag.shape[0]
      x_mag_target_length = int(X_mag.shape[0] / subseq_len ) * subseq_len + subseq_len
      X_mag = np.pad(X_mag, ([0,x_mag_target_length - X_mag.shape[0]], [0,0]), 'constant')
      num_examples = int(x_mag_target_length/subseq_len)
      X_mag = np.reshape(X_mag, [num_examples, subseq_len, 513, 1])
      gen_mags = []
      for n in range(num_examples):
        _gen = gen_sess.run([gen_mag_spec], feed_dict = {
            x_mag_input : X_mag[n:n+1]
            })[0]
        gen_mags.append(_gen[0])
      gen_mag = np.concatenate(gen_mags, axis = 0)
      gen_mag = gen_mag[0:x_mag_original_length]
      wave = magspec_to_waveform_lws(gen_mag.astype('float64'), 1024, 256)
    
    save_as_wav(wave_fp, args.fs, wave)
