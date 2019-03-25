import tensorflow as tf
from advoc.loader import decode_extract_and_batch
from model import Modes
from util import override_model_attrs
import numpy as np
import time
import advoc.spectral
from srezModel import SrezMelSpec
from spectral_util import SpectralUtil

def magnitude_to_linear_mel():
  NFFT = 1024
  NHOP = 256
  FMIN = 125.
  FMAX = 7600.
  NMELS = 80

  meltrans = advoc.spectral.create_mel_filterbank(
            fs, NFFT, fmin=FMIN, fmax=FMAX, n_mels=NMELS)
  invmeltrans = advoc.spectral.create_inverse_mel_filterbank(
            fs, NFFT, fmin=FMIN, fmax=FMAX, n_mels=NMELS)



def train(fps, args):
  # Initialize model
  
  model = SrezMelSpec(Modes.TRAIN)
  model, summary = override_model_attrs(model, args.model_overrides)
  print('-' * 80)
  print(summary)
  print('-' * 80)

  # Load data
  with tf.name_scope('loader'):
    x_mag_spec, x_wav = decode_extract_and_batch(
      fps,
      batch_size=model.train_batch_size,
      subseq_len=model.subseq_len,
      audio_fs=model.audio_fs,
      audio_mono=True,
      audio_normalize=False,
      decode_fastwav=args.data_fastwav,
      decode_parallel_calls=4,
      extract_type='mag_spec',
      extract_parallel_calls=8,
      repeat=True,
      shuffle=True,
      shuffle_buffer_size=512,
      subseq_randomize_offset=True,
      subseq_overlap_ratio=args.data_overlap_ratio,
      subseq_pad_end=True,
      prefetch_size=64 * 4,
      prefetch_gpu_num=0)

  # Create model
  spectral = SpectralUtil()
  
  x_mel_spec = spectral.mag_to_mel_linear_spec(x_mag_spec)
  x_inverted_mag_spec = spectral.mel_linear_to_mag_spec(x_mel_spec, transform = 'inverse')

  model(x_inverted_mag_spec, x_mag_spec)

  #Train
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_ckpt_every_nsecs,
      save_summaries_secs=args.train_summary_every_nsecs) as sess:
    while not sess.should_stop():
      model.train_loop(sess)

def eval(fps, args):
  raise NotImplementedError()


def infer(fps, args):
  raise NotImplementedError()

    

if __name__ == '__main__':
  from argparse import ArgumentParser
  import glob
  import os

  parser = ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'eval', 'infer'])
  parser.add_argument('train_dir', type=str)

  parser.add_argument('--data_dir', type=str, required=True)
  parser.add_argument('--data_fastwav', dest='data_fastwav', action='store_true')
  parser.add_argument('--data_overlap_ratio', type=float)
  parser.add_argument('--model_overrides', type=str)
  parser.add_argument('--train_ckpt_every_nsecs', type=int)
  parser.add_argument('--infer_batch_size', type=int)
  parser.add_argument('--train_summary_every_nsecs', type=int)
  parser.add_argument('--eval_dataset_name', type=str)
  parser.add_argument('--eval_wavenet_meta_fp', type=str)
  parser.add_argument('--eval_wavenet_ckpt_fp', type=str)
  parser.add_argument('--infer_dataset_name', type=str)
  parser.add_argument('--infer_ckpt_path', type=str)

  parser.set_defaults(
      mode=None,
      train_dir=None,
      data_dir=None,
      data_fastwav=False,
      data_overlap_ratio=0.25,
      model_overrides=None,
      train_ckpt_every_nsecs=360,
      train_summary_every_nsecs=60,
      infer_batch_size=1,
      eval_dataset_name=None,
      eval_wavenet_meta_fp=None,
      eval_wavenet_ckpt_fp=None,
      infer_dataset_name=None,
      infer_ckpt_path=None
      )

  args = parser.parse_args()

  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  fps = glob.glob(os.path.join(args.data_dir, '*'))
  print('Found {} audio files'.format(len(fps)))

  if args.mode == 'train':
    train(fps, args)
  elif args.mode == 'eval':
    eval(fps, args)
  elif args.mode == 'infer':
    infer(fps, args)
  else:
    raise NotImplementedError()

