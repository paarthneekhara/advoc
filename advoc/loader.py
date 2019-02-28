import numpy as np
import tensorflow as tf

from advoc.audioio import decode_audio
from advoc.spectral import waveform_to_r9y9_melspec_tf


def decode_extract_and_batch(
    fps,
    batch_size,
    subseq_len,
    audio_fs,
    audio_mono=True,
    audio_normalize=True,
    decode_fastwav=False,
    decode_parallel_calls=1,
    extract_type='r9y9_melspec',
    extract_parallel_calls=1,
    repeat=False,
    shuffle=False,
    shuffle_buffer_size=None,
    subseq_randomize_offset=False,
    subseq_overlap_ratio=0,
    subseq_pad_end=False,
    prefetch_size=None,
    prefetch_gpu_num=None):
  """Decodes audio file paths into mini-batches of samples.

  Args:
    fps: List of audio file paths.
    batch_size: Number of items in the batch.
    subseq_len: Length of the subsequences in feature timesteps.
    audio_fs: Sample rate for decoded audio files.
    audio_mono: If false, preserves multichannel (all files must have same).
    audio_normalize: If false, do not normalize audio waveforms.
    decode_fastwav: If true, uses scipy to decode standard wav files.
    decode_parallel_calls: Number of parallel decoding threads.
    extract_type: Type of features to extract (None for no features)
    extract_parallel_calls: Number of parallel extraction threads.
    repeat: If true (for training), continuously iterate through the dataset.
    shuffle: If true (for training), buffer and shuffle the subsequences.
    subseq_randomize_offset: If true, randomize starting position for subseq.
    pad_end: If true, allows zero-padded examples at the end.

  Returns:
    A tuple of np.float32 tensors representing audio and feature subsequences.
      audio: [batch_size, ?, 1, nch]
  """
  # Create dataset of filepaths
  dataset = tf.data.Dataset.from_tensor_slices(fps)

  # Shuffle all filepaths every epoch
  if shuffle:
    dataset = dataset.shuffle(buffer_size=len(fps))

  # Repeat
  if repeat:
    dataset = dataset.repeat()

  def _decode_audio_shaped(fp):
    _decode_audio_closure = lambda _fp: decode_audio(
      _fp,
      fs=audio_fs,
      mono=audio_mono,
      normalize=audio_normalize,
      fastwav=decode_fastwav)[1]

    audio = tf.py_func(
        _decode_audio_closure,
        [fp],
        tf.float32,
        stateful=False)
    audio.set_shape([None, 1, 1 if audio_mono else None])

    return audio

  # Decode audio
  dataset = dataset.map(
      _decode_audio_shaped,
      num_parallel_calls=decode_parallel_calls)

  # Extract features
  if extract_type is None:
    feature_fs = audio_fs
    subseq_pad_val = 0.
    dataset = dataset.map(lambda x: (x, x))
  elif extract_type == 'r9y9_melspec':
    nhop = 256

    def _extract_feats_shaped(wav):
      return waveform_to_r9y9_melspec_tf(wav[tf.newaxis], fs=audio_fs)[0]

    feature_fs = audio_fs / nhop
    subseq_pad_val = 0.
    dataset = dataset.map(
        lambda x: (_extract_feats_shaped(x), x),
        num_parallel_calls=extract_parallel_calls)
  else:
    raise ValueError()

  # Extract paired audio and features
  def _parallel_subseq(features, audio):
    # Calculate hop size
    assert subseq_overlap_ratio >= 0
    subseq_hop = int(round(subseq_len * (1. - subseq_overlap_ratio)))
    if subseq_hop < 1:
      raise ValueError('Overlap ratio too high')

    # Audio is [nsamps, 1, nch] at audio_fs Hz
    # Features is [ntsteps, ?, nch] at feature_fs Hz
    # Calculate ratio which is equal to the number of samples per tstep.
    nsamps_per_tstep = float(audio_fs) / float(feature_fs)
    audio_subseq_len = subseq_len * nsamps_per_tstep
    audio_subseq_len = int(round(audio_subseq_len) + 1e-4)
    audio_subseq_hop = subseq_hop * nsamps_per_tstep
    audio_subseq_hop = int(round(audio_subseq_hop) + 1e-4)

    """
    print(audio_fs)
    print(audio_subseq_len)
    print(audio_subseq_hop)
    print(feature_fs)
    print(subseq_len)
    print(subseq_hop)
    """

    # Retrieve nsamps and ntsteps
    # TODO: Check that these values are sane wrt audio_fs and feature_fs.
    nsamps = tf.shape(audio)[0]
    ntsteps = tf.shape(features)[0]

    # Randomize starting phase:
    if subseq_randomize_offset:
      start = tf.random_uniform([], maxval=subseq_len, dtype=tf.int32)
      start_audio = tf.cast(start, np.float32) * nsamps_per_tstep
      start_audio = tf.cast(tf.round(start_audio + 1e-4), np.int32)
      audio = audio[start_audio:]
      features = features[start:]

    # Extract subsequences
    # TODO: Only compute subseqs once when not extracting features
    feature_subseqs = tf.contrib.signal.frame(
        features,
        subseq_len,
        subseq_hop,
        pad_end=subseq_pad_end,
        pad_value=subseq_pad_val,
        axis=0)
    audio_subseqs = tf.contrib.signal.frame(
        audio,
        audio_subseq_len,
        audio_subseq_hop,
        pad_end=subseq_pad_end,
        pad_value=0,
        axis=0)

    # TODO: Make sure first dim is equal (same number of subseqs)
    """
    print_op = tf.print(
        tf.shape(audio),
        tf.shape(features),
        tf.shape(audio_subseqs),
        tf.shape(feature_subseqs))
    with tf.control_dependencies([print_op]):
      audio_subseqs = tf.identity(audio_subseqs)
    """

    return feature_subseqs, audio_subseqs

  def _parallel_subseq_dataset_wrapper(features, audio):
    feature_subseqs, audio_subseqs = _parallel_subseq(features, audio)
    return tf.data.Dataset.zip((
      tf.data.Dataset.from_tensor_slices(feature_subseqs),
      tf.data.Dataset.from_tensor_slices(audio_subseqs),
    ))

  # Extract parallel subsequences from both audio and features
  dataset = dataset.flat_map(_parallel_subseq_dataset_wrapper)

  # Shuffle examples
  if shuffle:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

  # Make batches
  dataset = dataset.batch(batch_size, drop_remainder=True)

  # Queue up a number of batches on the CPU side
  if prefetch_size is not None:
    dataset = dataset.prefetch(prefetch_size)
    if prefetch_gpu_num is not None and prefetch_gpu_num >= 0:
      dataset = dataset.apply(
          tf.data.experimental.prefetch_to_device(
            '/device:GPU:{}'.format(prefetch_gpu_num)))

  # Get tensors
  iterator = dataset.make_one_shot_iterator()

  x_feats, x_audio = iterator.get_next()

  return tf.stop_gradient(x_feats), tf.stop_gradient(x_audio)
