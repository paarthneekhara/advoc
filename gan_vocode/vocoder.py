import tensorflow as tf
from advoc.loader import decode_extract_and_batch
from model import Modes
from util import override_model_attrs
import numpy as np
import time
import advoc.spectral
from argparse import ArgumentParser
from advoc import audioio
import os
import glob
from vocoderGANPatches import VocoderGAN

def main():
  parser = ArgumentParser()
  parser.add_argument('--input_dir', type=str)
  parser.add_argument('--output_dir', type=str)
  parser.add_argument('--meta_fp', type=str)
  parser.add_argument('--ckpt_fp', type=str)
  parser.add_argument('--n_mels', type=int)
  parser.add_argument('--fs', type=int)
  parser.add_argument('--model_overrides', type=int)

  parser.set_defaults( 
    input_file=None,
    output_dir=None,
    ckpt_fp=None,
    meta_fp=None,
    n_mels=80,
    fs=22050,
    model_overrides=None
    )
  args = parser.parse_args()

  if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

  model = VocoderGAN(Modes.INFER)
  model, summary = override_model_attrs(model, args.model_overrides)
  print('-' * 80)
  print(summary)
  print('-' * 80)

  x_spec = tf.placeholder(tf.float32,  [1, model.subseq_len, args.n_mels, 1 ])
  z = tf.random.normal([1, 1, 1, model.zdim], dtype=tf.float32)
  z_tiled = z * tf.constant(1.0, shape=[1, model.subseq_len, 1, model.zdim])

  with tf.variable_scope('G') as vs:
    G_z = model.build_generator(x_spec, z_tiled)
    G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
  step = tf.train.get_or_create_global_step()
  gan_saver = tf.train.Saver(var_list=G_vars + [step], max_to_keep=1)

  import time
  
  with tf.Session() as sess:
    print("Restoring")
    gan_saver.restore(sess, args.ckpt_fp)
    print("Restored")
    start = time.time()
    spec_fps = glob.glob(os.path.join(args.input_dir, '*.npy'))
    for fidx, fp in enumerate(spec_fps):
      _mel_spec = np.load(fp)[:,:,0]
      original_length = _mel_spec.shape[0]
      target_length = int(original_length / model.subseq_len ) * model.subseq_len + model.subseq_len
      num_examples = int(target_length/model.subseq_len)
      _mel_spec = np.pad(_mel_spec, ([0,target_length - original_length], [0,0]), 'constant')
      mel_spec_np = np.reshape(_mel_spec, [num_examples, model.subseq_len, args.n_mels, 1])
      gen_audios = []
      for n in range(num_examples):
        _gen_audio = sess.run([G_z], feed_dict = {
          x_spec : mel_spec_np[n:n+1]
          })[0]
        gen_audios.append(_gen_audio)
      
      gen_audio = np.concatenate(gen_audios, axis = 1)
      gen_audio = gen_audio[:,0:original_length*256,:,:]
      fn = fp.split("/")[-1][:-3] + "wav"
      output_file_name = os.path.join(args.output_dir, fn)
      print("Writing", fidx, output_file_name)
      audioio.save_as_wav(output_file_name, args.fs, gen_audio[0])
  end = time.time()
  print("Execution Time in Seconds", end - start)

if __name__ == '__main__':
  main()

'''

export CUDA_VISIBLE_DEVICES="-1" 
python vocoder.py \
--input_dir /data2/advoc/Clock/ClockSpectrogram \
--ckpt_fp /data2/paarth/TrainDir/vocoder/new/spec_reg_1Patchedl2_PS2/eval_valid/best_spec_l2-146054 \
--output_dir /data2/paarth/dump; \


'''