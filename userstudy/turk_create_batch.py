import glob
import os
import random
import shutil
import uuid

name = '00_userstudy_pilot'
waveform_dir = '/data2/advoc/userstudy/Waveforms'
#waveform_dir = '/home/cdonahue/nuuserstudy/Waveforms'
#study_dir = './{}'.format(name)
study_dir = '/data2/advoc/turkhits/{}'.format(name)
url_templ = 'http://deepyeti.ucsd.edu/cdonahue/advocturkhits/{}/wavs/{{}}.wav'.format(name)

methods = [
    'LJSpeechTest_Real_Original',
    'LJSpeechTest_Tacotron2_R9Y9_Mel80_WaveNetVocoder'
]

start_i = 0
num = 100

wave_fps = sorted(glob.glob(os.path.join(waveform_dir, methods[0], '*.wav')))
wave_fns = [os.path.split(fp)[1] for fp in wave_fps]

# Randomize (consistently)
random.seed(0)
random.shuffle(wave_fns)

# Select a batch
wave_fns = wave_fns[start_i:start_i+num]
assert len(wave_fns) == num

# Make sure all methods have these same filenames
for method in methods:
  method_dir = os.path.join(waveform_dir, method)
  for wave_fn in wave_fns:
    method_wave_fp = os.path.join(method_dir, wave_fn)
    assert os.path.exists(method_wave_fp)

# Create output directory
if os.path.isdir(study_dir):
  shutil.rmtree(study_dir)
os.makedirs(study_dir)
study_wav_dir = os.path.join(study_dir, 'wavs')
os.makedirs(study_wav_dir)

# Strip revealing information (create UUIDs)
uuids = set()
uuid_key = []
urls = []
for method in methods:
  method_dir = os.path.join(waveform_dir, method)
  for wave_fn in wave_fns:
    method_wave_fp = os.path.join(method_dir, wave_fn)
    wave_uuid = uuid.uuid4().hex
    assert wave_uuid not in uuids
    uuid_key.append('{},{},{}'.format(wave_uuid, method, wave_fn))
    uuids.add(wave_uuid)
    out_wave_fp = os.path.join(study_wav_dir, '{}.wav'.format(wave_uuid))
    shutil.copyfile(method_wave_fp, out_wave_fp)
    urls.append(url_templ.format(wave_uuid))

# Create key
with open(os.path.join(study_dir, 'key.csv'), 'w') as f:
  f.write('\n'.join(sorted(uuid_key)))

# Create hit
urls = ['audio_url'] + sorted(urls)
with open(os.path.join(study_dir, 'hit.csv'), 'w') as f:
  f.write('\n'.join(urls))
