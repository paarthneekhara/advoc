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

  model(x_inverted_mag_spec, x_mag_spec, x_wav)

  #Train
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_ckpt_every_nsecs,
      save_summaries_secs=args.train_summary_every_nsecs) as sess:
    
    _step = 0
    while not sess.should_stop() and _step < args.max_steps:
      _step = model.train_loop(sess)

  print("Done!")

def eval(fps, args):
  if args.eval_dataset_name is not None:
    eval_dir = os.path.join(args.train_dir,
        'eval_{}'.format(args.eval_dataset_name))
  else:
    eval_dir = os.path.join(args.train_dir, 'eval_valid')
  if not os.path.isdir(eval_dir):
    os.makedirs(eval_dir)
  model = SrezMelSpec(Modes.TRAIN)
  model, summary = override_model_attrs(model, args.model_overrides)
  print('-' * 80)
  print(summary)
  print('-' * 80)

  with tf.name_scope('loader'):
    x_mag_spec, x_wav = decode_extract_and_batch(
      fps,
      batch_size=model.eval_batch_size,
      subseq_len=model.subseq_len,
      audio_fs=model.audio_fs,
      audio_mono=True,
      audio_normalize=False,
      decode_fastwav=args.data_fastwav,
      decode_parallel_calls=4,
      extract_type='mag_spec',
      extract_parallel_calls=8,
      repeat=False,
      shuffle=False,
      shuffle_buffer_size=None,
      subseq_randomize_offset=False,
      subseq_overlap_ratio=0.,
      subseq_pad_end=True,
      prefetch_size=None,
      prefetch_gpu_num=None)
  
  spectral = SpectralUtil()
  x_mel_spec = spectral.mag_to_mel_linear_spec(x_mag_spec)
  x_inverted_mag_spec = spectral.mel_linear_to_mag_spec(x_mel_spec, transform = 'inverse')

  with tf.variable_scope("generator") as vs:
    if model.generator_type == "pix2pix":
      gen_mag_spec = model.build_generator(x_inverted_mag_spec)
    elif model.generator_type == "linear":
      gen_mag_spec = model.build_linear_generator(x_inverted_mag_spec)
    else:
      raise NotImplementedError()
      
    G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)

  gen_loss_L1 = tf.reduce_mean(tf.abs(x_mag_spec - gen_mag_spec))
  gan_step = tf.train.get_or_create_global_step()
  gan_saver = tf.train.Saver(var_list=G_vars + [gan_step], max_to_keep=1)


  all_gen_loss_L1 = tf.placeholder(tf.float32, [None])

  summaries = [
    tf.summary.scalar('gen_loss_L1', tf.reduce_mean(all_gen_loss_L1)),
  ]
  summaries = tf.summary.merge(summaries)

  # Create summary writer
  summary_writer = tf.summary.FileWriter(eval_dir)
  ckpt_fp = None
  _best_gen_loss_l1 = np.inf

  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      ckpt_fp = latest_ckpt_fp
      print('Evaluating {}'.format(ckpt_fp))

      with tf.Session() as sess:
        gan_saver.restore(sess, latest_ckpt_fp)
        _step = sess.run(gan_step)
        _all_gen_loss_L1 = []

        while True:
          try:
            _gen_loss_L1, _gen_mag_spec, _x_mag_spec = sess.run([gen_loss_L1, gen_mag_spec, x_mag_spec])
          except tf.errors.OutOfRangeError:
            break

          _all_gen_loss_L1.append(_gen_loss_L1)

        _all_gen_loss_L1 = np.array(_all_gen_loss_L1)
        
        _summaries = sess.run(summaries, 
          {
            all_gen_loss_L1: _all_gen_loss_L1, 
          }
        )
        summary_writer.add_summary(_summaries, _step)
        _gen_loss_L1_np = np.mean(_all_gen_loss_L1)

        if _gen_loss_L1_np < _best_gen_loss_l1:
          gan_saver.save(sess, os.path.join(eval_dir, 'best_gen_loss_l1'), _step)
          print("Saved best gen loss l1!")
      print('Done!')
    time.sleep(1)

def infer(fps, args):
  if args.infer_dataset_name is not None:
    infer_dir = os.path.join(args.train_dir,
        'infer_{}'.format(args.infer_dataset_name))
  else:
    infer_dir = os.path.join(args.train_dir, 'infer_valid')
  if not os.path.isdir(infer_dir):
    os.makedirs(infer_dir)

  model = SrezMelSpec(Modes.INFER)
  model, summary = override_model_attrs(model, args.model_overrides)
  print('-' * 80)
  print(summary)
  print('-' * 80)

  with tf.name_scope('loader'):
    x_mag_spec, x_wav = decode_extract_and_batch(
      fps,
      batch_size=args.infer_batch_size,
      subseq_len= model.subseq_len,
      audio_fs=model.audio_fs,
      audio_mono=True,
      audio_normalize=False,
      decode_fastwav=args.data_fastwav,
      decode_parallel_calls=4,
      extract_type='mag_spec',
      extract_parallel_calls=8,
      repeat=False,
      shuffle=False,
      shuffle_buffer_size=None,
      subseq_randomize_offset=False,
      subseq_overlap_ratio=0.,
      subseq_pad_end=True,
      prefetch_size=None,
      prefetch_gpu_num=0)

  spectral = SpectralUtil()
  x_mel_spec = spectral.mag_to_mel_linear_spec(x_mag_spec)
  x_inverted_mag_spec = spectral.mel_linear_to_mag_spec(x_mel_spec, transform = 'inverse')

  with tf.variable_scope("generator") as vs:
    if model.generator_type == "pix2pix":
      gen_mag_spec = model.build_generator(x_inverted_mag_spec)
    elif model.generator_type == "linear":
      gen_mag_spec = model.build_linear_generator(x_inverted_mag_spec)
    else:
      raise NotImplementedError()
    G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)

  step = tf.train.get_or_create_global_step()
  gan_saver = tf.train.Saver(var_list=G_vars + [step], max_to_keep=1)
  
  input_audio = tf.py_func( spectral.audio_from_mag_spec, [x_inverted_mag_spec[0]], tf.float32, stateful=False)
  target_audio = tf.py_func( spectral.audio_from_mag_spec, [x_mag_spec[0]], tf.float32, stateful=False)
  gen_audio = tf.py_func( spectral.audio_from_mag_spec, [gen_mag_spec[0]], tf.float32, stateful=False)

  input_audio = tf.reshape(input_audio, [1, 66304, 1, 1] )
  target_audio = tf.reshape(target_audio, [1, 66304, 1, 1] )
  gen_audio = tf.reshape(gen_audio, [1, 66304, 1, 1] )

  summaries = [
    tf.summary.audio('infer_x_wav', x_wav[:, :, 0, :], model.audio_fs),
    tf.summary.audio('infer_gen_audio', gen_audio[:, :, 0, :], model.audio_fs),
    tf.summary.audio('target_audio', target_audio[:, :, 0, :], model.audio_fs),
    tf.summary.audio('infer_input_audio', input_audio[:, :, 0, :], model.audio_fs)
  ]

  summaries = tf.summary.merge(summaries)
  # Create saver and summary writer
  summary_writer = tf.summary.FileWriter(infer_dir)

  if args.infer_ckpt_path is not None:
    # Infering From a particular Checkpoint
    ckpt_fp = args.infer_ckpt_path
    print('Infereing From {}'.format(ckpt_fp))

    with tf.Session() as sess:
      gan_saver.restore(sess, ckpt_fp)
      _step = sess.run(step)
      # Just one batch at a time
      while True:
        try:
          _summaries= sess.run(summaries)
          summary_writer.add_summary(_summaries, _step)
        except tf.errors.OutOfRangeError:
          break
      print('Done!')

  else:
    # Continuous Inference
    ckpt_fp = None
    while True:
      with tf.Session() as sess:
        latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
        if latest_ckpt_fp != ckpt_fp:
          ckpt_fp = latest_ckpt_fp
          print('Infereing From {}'.format(ckpt_fp))
          gan_saver.restore(sess, ckpt_fp)
          _step = sess.run(step)
          
          while True:
            try:
              _summaries= sess.run(summaries)
              summary_writer.add_summary(_summaries, _step)
            except tf.errors.OutOfRangeError:
              break
          print("Done!")
        time.sleep(1)

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
  parser.add_argument('--max_steps', type=int)
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
      max_steps=100000,
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

