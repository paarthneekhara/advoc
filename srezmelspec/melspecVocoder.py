import numpy as np
import tensorflow as tf
import lws
from advoc import audioio
from advoc import spectral
from model import Modes
from srezModel import SrezMelSpec
from util import override_model_attrs
from argparse import ArgumentParser
import spectral_util
import os
import glob
import scipy.io.wavfile

def main():
  parser = ArgumentParser()
  parser.add_argument('--input_dir', type=str)
  parser.add_argument('--output_dir', type=str)
  parser.add_argument('--meta_fp', type=str)
  parser.add_argument('--ckpt_fp', type=str)
  parser.add_argument('--heuristic', type=str)
  parser.add_argument('--n_mels', type=int)
  parser.add_argument('--fs', type=int)
  parser.add_argument('--subseq_len', type=int)

  parser.set_defaults( 
    input_file=None,
    output_dir=None,
    ckpt_fp=None,
    meta_fp=None,
    heuristic="lws",
    n_mels=80,
    fs=22050,
    subseq_len = 256
    )
  args = parser.parse_args()

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  gen_graph = tf.Graph()
  with gen_graph.as_default():
    gan_saver = tf.train.import_meta_graph(args.meta_fp)

  gen_sess = tf.Session(graph=gen_graph)
  print("Restoring")
  gan_saver.restore(gen_sess, args.ckpt_fp)
  gen_mag_spec = gen_graph.get_tensor_by_name('generator/decoder_1/strided_slice_1:0')
  x_mag_input = gen_graph.get_tensor_by_name('ExpandDims_1:0')

  su = spectral_util.SpectralUtil(n_mels = args.n_mels, fs = args.fs)

  spec_fps = glob.glob(os.path.join(args.input_dir, '*.npy'))
  for fidx, fp in enumerate(spec_fps):
    _mel_spec = np.load(fp)[:,:,0]
    X_mag = su.tacotron_mel_to_mag(_mel_spec)
    x_mag_original_length = X_mag.shape[0]
    x_mag_target_length = int(X_mag.shape[0] / subseq_len ) * subseq_len + subseq_len
    X_mag = np.pad(X_mag, ([0,x_mag_target_length - X_mag.shape[0]], [0,0]), 'constant')
    num_examples = int(x_mag_target_length/subseq_len)
    X_mag = np.reshape(X_mag, [num_examples, subseq_len, 513, 1])
    gen_mags = []
    heuristic_mags = []
    for n in range(num_examples):
      _gen, _heur = gen_sess.run([gen_mag_spec, x_mag_input], feed_dict = {
          x_mag_input : X_mag[n:n+1]
          })
      gen_mags.append(_gen[0])
      heuristic_mags.append(_heur[0])
    gen_mag = np.concatenate(gen_mags, axis = 0)
    heur_mag = np.concatenate(heuristic_mags, axis = 0)

    _gen_audio = su.audio_from_mag_spec(gen_mag)
    gen_mag = gen_mag[0:x_mag_original_length]

    if args.heuristic == 'lws':
      _gen_audio = spectral.magspec_to_waveform_lws(gen_mag, 1024, 256)
    elif args.heuristic == 'gl':
      _gen_audio = spectral.magspec_to_waveform_griffin_lim(gen_mag, 1024, 256)
    else:
      raise NotImplementedError()

    fn = fp.split("/")[-1][:-3] + "wav"
    output_file_name = os.path.join(args.output_dir, fn)
    print("Writing", fidx, output_file_name)
    audioio.save_as_wav(output_file_name, args.fs, _gen_audio)  

if __name__ == '__main__':
  main()