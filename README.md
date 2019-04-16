# Adversarial vocoding

Code from our paper *Expediting TTS Synthesis with Adversarial Vocoding*. Sound examples can be found [here](https://chrisdonahue.github.io/advoc_examples).

[Adversarial vocoding](#adversarial-vocoder-advoc) is a method for transforming perceptually-informed spectrograms (e.g. mel spectrograms) into audio. It can be used in combination with TTS systems which produce spectrograms (e.g. [Tacotron 2](https://github.com/Rayhane-mamah/Tacotron-2)). We also provide a method for [generating spectrograms with a GAN](#mel-spectrogram-gan), which can then be vocoded to audio using adversarial vocoding.

## Installation

To get started training adversarial vocoders, you must first install the lightweight `advoc` package. This package is a well-tested set of modules which handle audio IO, spectral processing, and heuristic vocoding in both `numpy` and `tensorflow`.

To install in a virtual environment, follow these instructions:

```
virtualenv -p python3 --no-site-packages advoc
cd advoc
git clone https://github.com/paarthneekhara/advoc.git
source bin/activate
cd advoc
pip install -e .
```

To install globally, run `sudo pip install -e .`.

To run our suite of tests to affirm reproducibility of our feature representations, heuristic inversion code, etc. run:

`python setup.py test`

Our setup script installed `tensorflow==1.13.1` which is linked to CUDA 10 by default. If you are using CUDA 9, you may want to instead use `tensorflow-gpu==1.12.0`:

```
pip uninstall tensorflow-gpu
pip install tensorflow-gpu==1.12.0
```

## Adversarial Vocoder (AdVoc)

An adversarial vocoder is a *magnitude estimation* method: it takes mel spectrograms (logarithmic frequency axis) and produces magnitude spectrograms (linear frequency axis). We pair these estimated magnitude spectrograms with phase estimates from [LWS](https://pypi.org/project/lws/) to produce audio.

### Training
To train the adversarial vocoder model on the LJ Speech Dataset, [download the data](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2) and extract the `.wav` files into `models/advoc/data/ljspeech/wavs`. Split this data into training, validation and test sets using `scripts/data_split.py` as follows:


```
cd scripts
python data_split.py \
  --source_dir models/advoc/data/ljspeech/wavs \
  --out_dir models/advoc/data/ljspeech/wavs_split
```

This script should create `train`, `valid`, and `test` directories in `models/advoc/data/ljspeech/wavs_split`.

Train the the adversarial vocoder model (AdVoc) on the training set as follows:

```
cd models/advoc
WORK_DIR=./train
export CUDA_VISIBLE_DEVICES="0"
python train_evaluate.py train \
  ${WORK_DIR} \
  --data_cfg ../../datacfg/ljspeech.txt \
  --data_dir ./data/ljspeech/wavs_split/train \
```

To train the smaller version of adversarial vocoder (AdVoc-small) use:

```
export CUDA_VISIBLE_DEVICES="0"
python train_evaluate.py train \
  ${WORK_DIR}$ \
  --data_cfg ../../datacfg/ljspeech.txt \
  --data_dir ./data/ljspeech/wavs_split/train \
  --model_type small
```

For custom datasets, see [here](#dataset-configuration).

#### Monitoring and continuous evaluation
Training logs can be visualized and audio samples can be listened to by launching tensorboard in the `WORK_DIR` as follows:

```
tensorboard --logdir=${WORK_DIR}$
```

To back up checkpoints every hour (GAN training may occasionally collapse so it's good to have backups)

```
python backup.py $WORK_DIR$ 60
```

To evaluate each checkpoint on the validation set, run the following:

```
export CUDA_VISIBLE_DEVICES="-1"
python train_evaluate.py eval \
  ${WORK_DIR}$ \
  --data_cfg ../../datacfg/ljspeech.txt \
  --data_dir ./data/ljspeech/wavs_split/valid \
```

### Inference

Extract mel-spectrograms for audio files from the test dataset using `scripts/audio_to_spectrogram.py` as follows:

```
cd scripts
python audio_to_spectrogram.py \
  --wave_dir ../models/advoc/data/ljspeech/wavs_split/test \
  --out_dir ../models/advoc/data/ljspeech/mel_specs/test \
  --data_fast_wav
```

Running this script should save the extracted mel-spectrogram in `models/advoc/data/ljspeech/mel_specs/test` as `.npy` files. 

The mel-spectrograms can be vocoded either using the pre-trained models provided at the bottom of this page or training the model from scratch using the steps given above. To vocode mel-spectrograms from an AdVoc checkpoint, use `scripts/spectrogram_advoc.py`:

```
cd scripts
export CUDA_VISIBLE_DEVICES="0"
python spectrogram_advoc.py \
--spec_dir ../models/advoc/data/ljspeech/mel_specs/test \
--out_dir ../models/advoc/data/ljspeech/vocoded_output/test \
--model_ckpt <PATH TO PRETRAINED CKPT> \
--meta_fp <PATH TO MODEL METAGRAPH>
```

The above command should save the vocoded audio in `models/advoc/data/ljspeech/vocoded_output/test`.


## Mel spectrogram GAN

Adversarial vocoding can be used to factorize audio generation into `P(spectrogram) * P(audio | spectrogram)`. This is useful because it is currently easier to generate spectrograms with GANs than raw audio. In our paper, we show that this factorized strategy can be used to achieve state-of-the-art results on unsupervised generation of small-vocabulary speech.

### Training a new model

To train a mel spectrogram GAN on spoken digits, first [download the SC09 dataset](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/sc09.tar.gz) and unzip to `models/melspecgan/data`. Then, run the following from this directory:

```
cd models/melspecgan
WORK_DIR=./sc09melspecgan
export CUDA_VISIBLE_DEVICES="0"
python train.py train \
  ${WORK_DIR} \
  --data_cfg ../../datacfg/sc09.txt \
  --data_dir ./data/sc09/train \
```

Then train an adversarial vocoder on this same dataset

```
cd models/advoc
WORK_DIR=./sc09advoc
export CUDA_VISIBLE_DEVICES="0"
python train_evaluate.py train \
  ${WORK_DIR} \
  --data_cfg ../../datacfg/sc09.txt \
  --data_dir ../melspecgan/data/sc09/train \
```

To train on a different dataset, see [here](#dataset-configuration).

### Inference

To generate mel spectrograms of spoken digits with MelspecGAN, first [download our pretrained checkpoint](https://drive.google.com/open?id=1oNBB-MSP28uHkqVOtYa6c3AfAQyEQZ0b) and extract to `scripts/sc09_melspecgan`. Then, use `scripts/generate_spectrogram.py` as follows:

```
cd scripts
python generate_spectrogram.py \
  --out_dir ./melspecgan_gen \
  --ckpt_fp ./sc09_melspecgan/best_score-55089 \
  --n 1000 \
```

You can also run on a MelspecGAN you trained by changing `--ckpt_fp`.

## Dataset configuration

Both models in this repository (MelspecGAN and Adversarial Vocoder) use configuration files to set data processing properties appropriate for particular datasets. These configuration files affect the loader which streams batches of examples directly from audio files (requires no preprocessing). The loader works by decoding individual audio files and extracting a variable number of "slices" (fixed-length examples) from each.

We provide configuration files for LJSpeech (`datacfg/ljspeech.txt`) and SC09 (`datacfg/sc09.txt`), but you may need to create your own if you want to use a different dataset. If you create one, you will need to set the following properties in the config file:

- `sample_rate`: The number of audio samples per second.
- `fastwav`: Set to `1` to use `scipy` to load WAV files, `0` to use `librosa` to load arbitrary audio files. Scipy is faster but only works for standard WAV files (16-bit PCM or 32-bit float), and does not support resampling. Librosa is slower but supports many types of audio files and resampling.
- `normalize`: Set to `1` to normalize each audio file, `0` to skip normalization.
- `slice_first_only`: Set to `1` to only use the first slice from each audio file, `0` to use as many slices as possible. Enabling this is appropriate for sound effects datasets, disabling is appropriate for datasets of longer audio files.
- `slice_randomize_offset`: Set to `1` to randomize the starting position for slicing, `0` to always start at the beginning. Enabling this is more appropriate for datasets of longer audio files.
- `slice_pad_end`: Set to `1` to zero-pad the end of each audio file to produce as many slices as possible, `0` to ignore incomplete slices at the end of each audio file.

## Pretrained checkpoints

- [LJSpeech Advoc Mel80](https://drive.google.com/open?id=1fyYugd73xofb6jU2m4GoKbCOVBaYc-zH)
- [LJSpeech Advoc Mel40](https://drive.google.com/open?id=1YAqCHrlDThpL71uZqSKa4onohOHKfO8H)
- [LJSpeech Advoc Mel20](https://drive.google.com/open?id=1uLTtY4PH6BC-DAmBWS0WZAHy7YVTo-ZI)
- [LJSpeech Advoc (small) Mel80](https://drive.google.com/open?id=126qWSsW7W8ofowETA4bFUjqddUzU7fhb)
- [SC09 MelSpecGan Mel80](https://drive.google.com/open?id=12X7B6bup2ObFckYlZt_14GFLFYdQcX-a)
- [SC09 Advoc Mel80](https://drive.google.com/open?id=1oNBB-MSP28uHkqVOtYa6c3AfAQyEQZ0b)
