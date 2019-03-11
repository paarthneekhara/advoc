import tensorflow as tf
from advoc.loader import decode_extract_and_batch
from model import Modes
from util import override_model_attrs
from vocoderGANPatches import VocoderGAN
import numpy as np
import time

def train(fps, args):
  # Initialize model
  model = VocoderGAN(Modes.TRAIN)
  model, summary = override_model_attrs(model, args.model_overrides)
  print('-' * 80)
  print(summary)
  print('-' * 80)

  # Load data
  with tf.name_scope('loader'):
    x_spec, x_wav = decode_extract_and_batch(
      fps,
      batch_size=model.train_batch_size,
      subseq_len=model.subseq_len,
      audio_fs=model.audio_fs,
      audio_mono=True,
      audio_normalize=False,
      decode_fastwav=args.data_fastwav,
      decode_parallel_calls=4,
      extract_type='r9y9_melspec',
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
  model(x_wav, x_spec)

  # Train
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=args.train_dir,
      save_checkpoint_secs=args.train_ckpt_every_nsecs,
      save_summaries_secs=args.train_summary_every_nsecs) as sess:
    while not sess.should_stop():
      model.train_loop(sess)

def eval(fps, args):
  if args.eval_dataset_name is not None:
    eval_dir = os.path.join(args.train_dir,
        'eval_{}'.format(args.eval_dataset_name))
  else:
    eval_dir = os.path.join(args.train_dir, 'eval_valid')
  if not os.path.isdir(eval_dir):
    os.makedirs(eval_dir)

  # Initialize model
  model = VocoderGAN(Modes.EVAL)
  model, summary = override_model_attrs(model, args.model_overrides)
  print('-' * 80)
  print(summary)
  print('-' * 80)

  # Load data
  with tf.name_scope('loader'):
    x_spec, x_wav = decode_extract_and_batch(
      fps,
      batch_size=model.eval_batch_size,
      subseq_len=model.subseq_len,
      audio_fs=model.audio_fs,
      audio_mono=True,
      audio_normalize=False,
      decode_fastwav=args.data_fastwav,
      decode_parallel_calls=4,
      extract_type='r9y9_melspec',
      extract_parallel_calls=8,
      repeat=False,
      shuffle=False,
      shuffle_buffer_size=None,
      subseq_randomize_offset=False,
      subseq_overlap_ratio=0.,
      subseq_pad_end=True,
      prefetch_size=None,
      prefetch_gpu_num=None)

  # Create GAN generation graph
  z = tf.random.normal([model.eval_batch_size, 1, 1, model.zdim], dtype=tf.float32)
  z_tiled = z * tf.constant(1.0, shape=[model.eval_batch_size, 64, 1, model.zdim])
  
  # Generator
  with tf.variable_scope('G') as vs:
    G_z = model.build_generator(x_spec, z_tiled)

  wav_l1 = tf.reduce_mean(tf.abs(x_wav - G_z))
  wav_l2 = tf.reduce_mean(tf.square(x_wav - G_z))

  gen_spec = tf.contrib.signal.stft(G_z[:,:,0,0], 1024, 256, pad_end=True)
  gen_spec_mag = tf.abs(gen_spec)

  target_spec = tf.contrib.signal.stft(x_wav[:,:,0,0], 1024, 256, pad_end=True)
  target_spec_mag = tf.abs(target_spec)

  spec_l1 = tf.reduce_mean(tf.abs(target_spec_mag - gen_spec_mag))
  spec_l2 = tf.reduce_mean(tf.square(target_spec_mag - gen_spec_mag))

  # WaveNet eval
  wavenet_graph = tf.Graph()
  with wavenet_graph.as_default():
    wavenet_saver = tf.train.import_meta_graph(args.eval_wavenet_meta_fp)
  wavenet_sess = tf.Session(graph=wavenet_graph)
  wavenet_saver.restore(wavenet_sess, args.eval_wavenet_ckpt_fp)
  wavenet_step = wavenet_graph.get_tensor_by_name('global_step:0')
  wavenet_chop = wavenet_graph.get_tensor_by_name('input_chop:0')
  wavenet_chopped = wavenet_graph.get_tensor_by_name('chopped:0')
  wavenet_input_wave = wavenet_graph.get_tensor_by_name('input_wave:0')
  wavenet_avg_nll = wavenet_graph.get_tensor_by_name('avg_nll:0')
  print('Loaded WaveNet (step {})'.format(wavenet_sess.run(wavenet_step)))

  G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
  gan_step = tf.train.get_or_create_global_step()
  
  gan_saver = tf.train.Saver(var_list=G_vars + [gan_step], max_to_keep=1)
  gan_saver2 = tf.train.Saver(var_list=G_vars + [gan_step], max_to_keep=1)
  gan_saver3 = tf.train.Saver(var_list=G_vars + [gan_step], max_to_keep=1)
  
  all_wav_l1 = tf.placeholder(tf.float32, [None])
  all_wav_l2 = tf.placeholder(tf.float32, [None])
  all_spec_l1 = tf.placeholder(tf.float32, [None])
  all_spec_l2 = tf.placeholder(tf.float32, [None])
  wavenet_all_nll = tf.placeholder(tf.float32, [None])

  summaries = [
    tf.summary.scalar('wav_l1', tf.reduce_mean(all_wav_l1)),
    tf.summary.scalar('wav_l2', tf.reduce_mean(all_wav_l2)),
    tf.summary.scalar('spec_l1', tf.reduce_mean(all_spec_l1)),
    tf.summary.scalar('spec_l2', tf.reduce_mean(all_spec_l2)),
    tf.summary.scalar('wavenet_nll', tf.reduce_mean(wavenet_all_nll))
  ]
  summaries = tf.summary.merge(summaries)

  # Create summary writer
  summary_writer = tf.summary.FileWriter(eval_dir)

  ckpt_fp = None
  _best_wav_l1 = np.inf
  _best_spec_l2 = np.inf
  _best_wavenet_nll = np.inf
  while True:
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    if latest_ckpt_fp != ckpt_fp:
      ckpt_fp = latest_ckpt_fp
      print('Evaluating {}'.format(ckpt_fp))

      with tf.Session() as sess:
        gan_saver.restore(sess, latest_ckpt_fp)

        _step = sess.run(gan_step)

        _all_G_z_slices = []
        _all_wav_l1 = []
        _all_wav_l2 = []
        _all_spec_l1 = []
        _all_spec_l2 = []

        while True:
          try:
            _G_z, _wav_l1, _wav_l2, _spec_l1, _spec_l2 = sess.run([
              G_z,
              wav_l1, 
              wav_l2, 
              spec_l1, 
              spec_l2])

          except tf.errors.OutOfRangeError:
            break

          for _G_z_waveform in _G_z:
            _G_z_slices = wavenet_sess.run(wavenet_chopped, {wavenet_chop: _G_z_waveform})
            _all_G_z_slices.append(_G_z_slices)

          _all_wav_l1.append(_wav_l1)
          _all_wav_l2.append(_wav_l2)
          _all_spec_l1.append(_spec_l1)
          _all_spec_l2.append(_spec_l2)
        
        _wavenet_all_nll = []
        _all_G_z_slices = np.concatenate(_all_G_z_slices, axis=0)
        for i in range(0, _all_G_z_slices.shape[0], 16):
          _wavenet_input_wave = _all_G_z_slices[i:i+16]
          if _wavenet_input_wave.shape[0] == 16:
            _wavenet_avg_nll = wavenet_sess.run(wavenet_avg_nll, {wavenet_input_wave: _wavenet_input_wave})
            _wavenet_all_nll.append(_wavenet_avg_nll)

        _wavenet_all_nll = np.concatenate(_wavenet_all_nll, axis=0)

        _all_wav_l1 = np.array(_all_wav_l1)
        _all_wav_l2 = np.array(_all_wav_l2)
        _all_spec_l1 = np.array(_all_spec_l1)
        _all_spec_l2 = np.array(_all_spec_l2)

        _summaries = sess.run(summaries, 
          {
            all_wav_l1: _all_wav_l1, 
            all_wav_l2: _all_wav_l2, 
            all_spec_l1: _all_spec_l1, 
            all_spec_l2: _all_spec_l2,
            wavenet_all_nll: _wavenet_all_nll
          }
        )
        summary_writer.add_summary(_summaries, _step)

        _all_wav_l1_np = np.mean(_all_wav_l1)
        _all_spec_l2_np = np.mean(_all_spec_l2)
        _wavenet_all_nll = np.mean(_wavenet_all_nll)

        if _wavenet_all_nll < _best_wavenet_nll:
          gan_saver.save(sess, os.path.join(eval_dir, 'best_nll'), _step)
          _best_wavenet_nll = _wavenet_all_nll
          print("Saved best wavenet NLL!")
        
        if _all_wav_l1_np < _best_wav_l1:
          gan_saver2.save(sess, os.path.join(eval_dir, 'best_wav_l1'), _step)
          _best_wav_l1 = _all_wav_l1_np
          print("Saved best wav l1!")

        if _all_spec_l2_np < _best_spec_l2:
          gan_saver3.save(sess, os.path.join(eval_dir, 'best_spec_l2'), _step)
          _best_spec_l2 = _all_spec_l2_np
          print("Saved best spec l2!")

      print('Done!')

    time.sleep(1)

  wavenet_sess.close()


def infer(fps, args):
  if args.infer_dataset_name is not None:
    infer_dir = os.path.join(args.train_dir,
        'infer_{}'.format(args.eval_dataset_name))
  else:
    infer_dir = os.path.join(args.train_dir, 'infer_valid')
  if not os.path.isdir(infer_dir):
    os.makedirs(infer_dir)

  # Initialize model
  model = VocoderGAN(Modes.INFER)
  model, summary = override_model_attrs(model, args.model_overrides)
  print('-' * 80)
  print(summary)
  print('-' * 80)

  # Load data
  with tf.name_scope('loader'):
    x_spec, x_wav = decode_extract_and_batch(
      fps,
      batch_size=1,
      subseq_len= model.subseq_len * 8, #8 seconds
      audio_fs=model.audio_fs,
      audio_mono=True,
      audio_normalize=False,
      decode_fastwav=args.data_fastwav,
      decode_parallel_calls=4,
      extract_type='r9y9_melspec',
      extract_parallel_calls=8,
      repeat=False,
      shuffle=False,
      shuffle_buffer_size=None,
      subseq_randomize_offset=False,
      subseq_overlap_ratio=0.,
      subseq_pad_end=True,
      prefetch_size=None,
      gpu_num=None)

  # Create GAN generation graph
  x_spec_sliced = tf.reshape(x_spec, [8, model.subseq_len, x_spec.shape[2], 1 ])
  z = tf.random.normal([8, 1, 1, model.zdim], dtype=tf.float32)
  z_tiled = z * tf.constant(1.0, shape=[8, model.subseq_len, 1, model.zdim])

  # Generator
  with tf.variable_scope('G') as vs:
    G_z = model.build_generator(x_spec_sliced, z_tiled)

  G_z_tiled = tf.reshape(G_z, x_wav.shape)
  G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
  step = tf.train.get_or_create_global_step()
  step_op = tf.assign(step, step+1)
  gan_saver = tf.train.Saver(var_list=G_vars, max_to_keep=1)

  summaries = [
    tf.summary.audio('infer_x_wav', x_wav[:, :, 0, :], model.audio_fs),
    tf.summary.audio('infer_G_z', G_z_tiled[:, :, 0, :], model.audio_fs),
  ]

  summaries = tf.summary.merge(summaries)
  # Create saver and summary writer
  summary_writer = tf.summary.FileWriter(infer_dir)

  
    
  ckpt_fp = args.infer_ckpt_path
  print('Infereing From {}'.format(ckpt_fp))

  with tf.Session() as sess:
    gan_saver.restore(sess, ckpt_fp)
    sess.run(step.initializer)
    # _step = sess.run(gan_step)
    while True:
      try:
        _summaries, _step, _ = sess.run([summaries, step, step_op])
        summary_writer.add_summary(_summaries, _step)
      except tf.errors.OutOfRangeError:
        break
    print('Done!')
    

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

