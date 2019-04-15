import numpy as np
import tensorflow as tf

from advoc.audioio import decode_audio
from advoc.spectral import waveform_to_melspec_tf, stft_tf


def decode_extract_and_batch(
    fps,
    batch_size,
    slice_len,
    audio_fs=22050,
    audio_mono=True,
    audio_normalize=False,
    decode_fastwav=False,
    decode_parallel_calls=1,
    extract_type=None,
    extract_nfft=1024,
    extract_nhop=256,
    extract_parallel_calls=1,
    repeat=False,
    shuffle=False,
    shuffle_buffer_size=None,
    slice_first_only=False,
    slice_randomize_offset=False,
    slice_overlap_ratio=0,
    slice_pad_end=False,
    prefetch_size=None,
    prefetch_gpu_num=None):
  """Decodes audio files directly into [b, slice_len, nfeats, nch] batches.

  This is a monstrous function signature. However, this method needs to do a 
  lot of stuff and there's no useful intermediate results, so we made it one
  big block. It is a bit cumbersome to configure but the functionality of 
  decoding directly into tensors is convenient enough to merit this.

  Args:
    fps: List of audio file paths.
    batch_size: Number of items in the batch.
    slice_len: Length of the slices. If extract_type=None this is samples.
    audio_fs: Sample rate for decoded audio files.
    audio_mono: If false, preserves multichannel (all files must have same).
    audio_normalize: If true, normalize audio waveforms.
    decode_fastwav: If true, uses scipy to quickly decode standard wav files.
    decode_parallel_calls: Number of parallel decoding threads.
    extract_type: Type of spectral features to extract: None, magspec, melspec.
    extract_nfft: STFT window size for feature extraction.
    extract_nhop: STFT hop size for feature extraction.
    extract_parallel_calls: Number of parallel extraction threads.
    repeat: If true (for training), continuously iterate through the dataset.
    shuffle: If true (for training), buffer and shuffle the slices.
    shuffle_buffer_size: Size of buffer for shuffling.
    slice_first_only: If true, only use first slice from each audio file.
    slice_randomize_offset: If true, randomize starting position for slice.
    slice_overlap_ratio: Ratio of overlap between feature slices.
    slice_pad_end: If true, zero pad features.
    prefetch_size: If a number, prefetch this many batches.
    prefetch_gpu_num: If a number, prefetch to this GPU num.

  Returns:
    A tuple of np.float32 tensors representing audio and feature slices.
      audio: [batch_size, ?, 1, nch]
      features: [batch_size, ? // nhop, nfeats, nch]
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
    slice_pad_val = 0.
    dataset = dataset.map(lambda x: (x, x))
  elif extract_type == 'melspec':
    def _extract_feats_shaped(wav):
      return waveform_to_melspec_tf(
          wav[tf.newaxis],
          fs=audio_fs,
          nfft=extract_nfft,
          nhop=extract_nhop)[0]

    feature_fs = audio_fs / extract_nhop
    slice_pad_val = 0.
    dataset = dataset.map(
        lambda x: (_extract_feats_shaped(x), x),
        num_parallel_calls=extract_parallel_calls)
  elif extract_type == 'magspec':
    def _extract_feats_shaped(wav):
      spec = stft_tf(
          wav[tf.newaxis],
          nfft=extract_nfft,
          nhop=extract_nhop)[0]
      return tf.abs(spec)

    feature_fs = audio_fs / extract_nhop
    slice_pad_val = 0.
    dataset = dataset.map(
        lambda x: (_extract_feats_shaped(x), x),
        num_parallel_calls=extract_parallel_calls)
  else:
    raise ValueError()

  # Extract paired audio and features
  def _parallel_slice(features, audio):
    # Calculate hop size
    if slice_overlap_ratio < 0:
      raise ValueError('Slice overlap must be nonnegative')
    slice_hop = int(round(slice_len * (1. - slice_overlap_ratio)))
    if slice_hop < 1:
      raise ValueError('Overlap ratio too high')

    # Audio is [nsamps, 1, nch] at audio_fs Hz
    # Features is [ntsteps, ?, nch] at feature_fs Hz
    # Calculate ratio which is equal to the number of samples per tstep.
    nsamps_per_tstep = float(audio_fs) / float(feature_fs)
    audio_slice_len = slice_len * nsamps_per_tstep
    audio_slice_len = int(round(audio_slice_len) + 1e-4)
    audio_slice_hop = slice_hop * nsamps_per_tstep
    audio_slice_hop = int(round(audio_slice_hop) + 1e-4)

    # Retrieve nsamps and ntsteps
    # TODO: Check that these values are sane wrt audio_fs and feature_fs.
    nsamps = tf.shape(audio)[0]
    ntsteps = tf.shape(features)[0]

    # Randomize starting phase:
    if slice_randomize_offset:
      start = tf.random_uniform([], maxval=slice_len, dtype=tf.int32)
      start_audio = tf.cast(start, np.float32) * nsamps_per_tstep
      start_audio = tf.cast(tf.round(start_audio + 1e-4), np.int32)
      audio = audio[start_audio:]
      features = features[start:]

    # Extract slices
    # TODO: Only compute slices once when not extracting features
    feature_slices = tf.contrib.signal.frame(
        features,
        slice_len,
        slice_hop,
        pad_end=slice_pad_end,
        pad_value=slice_pad_val,
        axis=0)
    audio_slices = tf.contrib.signal.frame(
        audio,
        audio_slice_len,
        audio_slice_hop,
        pad_end=slice_pad_end,
        pad_value=0,
        axis=0)

    if slice_first_only:
      feature_slices = feature_slices[:1]
      audio_slices = audio_slices[:1]

    # TODO: Make sure first dim is equal (same number of slices)

    return feature_slices, audio_slices

  def _parallel_slice_dataset_wrapper(features, audio):
    feature_slices, audio_slices = _parallel_slice(features, audio)
    return tf.data.Dataset.zip((
      tf.data.Dataset.from_tensor_slices(feature_slices),
      tf.data.Dataset.from_tensor_slices(audio_slices),
    ))

  # Extract parallel slices from both audio and features
  dataset = dataset.flat_map(_parallel_slice_dataset_wrapper)

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
