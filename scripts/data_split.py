from argparse import ArgumentParser
import glob
import os
import shutil

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument('--source_dir', type=str)
  parser.add_argument('--out_dir', type=str)
  
  parser.set_defaults(
    out_dir=None,
    source_dir=None
  )
  args = parser.parse_args()
  
  OUT_DIR = args.out_dir
  if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)

  splits = [('valid', 0.05), ('test', 0.05)]
  wavs = sorted(glob.glob(os.path.join(args.source_dir, "*.wav")))
  
  split_to_len = {}
  for split_name, split_proportion in splits:
    split_len = int(split_proportion * len(wavs))
    split_to_len[split_name] = split_len
  split_to_len['train'] = len(wavs) - sum(split_to_len.values())

  idx = 0
  for split_name in ['train'] + [s[0] for s in splits]:
    split_dir = os.path.join(OUT_DIR, split_name)
    os.makedirs(split_dir)

    split_len = split_to_len[split_name]

    split_fps = wavs[idx:idx+split_len]

    for fp in split_fps:
      shutil.copy(fp, split_dir)

    idx += split_len