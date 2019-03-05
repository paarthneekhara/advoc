import tensorflow as tf

from advoc.loader import decode_extract_and_batch
from model import Modes
from util import override_model_attrs

from wavenet_auto import Wavenet
from wavenet_vocoder import WavenetVocoder

_NAMED_MODELS = {
    'wavenet': Wavenet,
    'wavenet_vocoder': WavenetVocoder
}

def train(fps, args):
  # Initialize model
  model = _NAMED_MODELS[args.model](Modes.TRAIN)
  model, summary = override_model_attrs(model, args.model_overrides)
  print('-' * 80)
  print(summary)
  print('-' * 80)

  # Load data
  with tf.name_scope('loader'):
    # TODO: figure out how to handle prefetching to GPU with multiple GPUs
    x_spec, x_wav = decode_extract_and_batch(
      fps,
      batch_size=model.train_batch_size,
      subseq_len=model.subseq_len,
      audio_fs=model.audio_fs,
      audio_mono=True,
      audio_normalize=False,
      decode_fastwav=args.data_fastwav,
      decode_parallel_calls=4,
      extract_type=model.extract_type,
      extract_parallel_calls=None if model.extract_type is None else 8,
      repeat=True,
      shuffle=True,
      shuffle_buffer_size=512,
      subseq_randomize_offset=True,
      subseq_overlap_ratio=args.data_overlap_ratio,
      subseq_pad_end=True,
      prefetch_size=model.train_batch_size * 4,
      prefetch_gpu_num=0)

  # Create model
  model(x_spec, x_wav)

  # Train
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.train.MonitoredTrainingSession(
      config=config,
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_ckpt_every_nsecs,
      save_summaries_secs=args.train_summary_every_nsecs) as sess:
    while not sess.should_stop():
      model.train_loop(sess)


if __name__ == '__main__':
  from argparse import ArgumentParser
  import glob
  import os

  parser = ArgumentParser()

  parser.add_argument('mode', type=str, choices=['train', 'eval'])
  parser.add_argument('train_dir', type=str)

  parser.add_argument('--data_dir', type=str, required=True)
  parser.add_argument('--data_librosa', dest='data_fastwav', action='store_false')
  parser.add_argument('--data_overlap_ratio', type=float)

  parser.add_argument('--model', type=str, choices=list(_NAMED_MODELS.keys()))
  parser.add_argument('--model_overrides', type=str)

  parser.add_argument('--train_ckpt_every_nsecs', type=int)
  parser.add_argument('--train_summary_every_nsecs', type=int)

  parser.set_defaults(
      mode=None,
      train_dir=None,
      data_dir=None,
      data_fastwav=True,
      data_overlap_ratio=0.25,
      model='wavenet',
      model_overrides=None,
      train_ckpt_every_nsecs=360,
      train_summary_every_nsecs=60)

  args = parser.parse_args()

  if not os.path.isdir(args.train_dir):
    os.makedirs(args.train_dir)

  fps = glob.glob(os.path.join(args.data_dir, '*'))
  print('Found {} audio files'.format(len(fps)))

  if args.mode == 'train':
    train(fps, args)
