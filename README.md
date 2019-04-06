# Adversarial vocoding

Code from our paper *Expediting TTS Synthesis with Adversarial Vocoding*. Sound examples can be found [here](https://chrisdonahue.github.io/advoc_examples).

## Installation

There are multiple training scripts in this repository, but they all require the lightweight `advoc` package. This package is a well-tested set of modules which handle audio IO, spectral processing, and heuristic vocoding in both `numpy` and `tensorflow`.

To install globally, use `sudo pip install -e .` from this directory.

To install in a virtual environment, follow these instructions:

```
virtualenv -p python3 --no-site-packages advoc
cd advoc
git clone git@github.com:chrisdonahue/advoc.git
source bin/activate
cd advoc
pip install -e .
```

To run our suite of tests to affirm reproducibility of our feature representations, heuristic inversion code, etc. run:

`python setup.py test`

## Adversarial Vocoder (AdVoc)

### Training
To reproduce the experiments in our paper, [download the LJ Speech Dataset][https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2] and extract the ```.wav``` files to ```models/advoc/data/ljspeech/wavs```. Split this data into training, validation and test sets using ```scripts/data_split.py``` as follows:


```
cd scripts
python data_split.py \
--source_dir models/advoc/data/ljspeech/wavs \
--out_dir models/advoc/data/ljspeech/wavs_split
```

This script should create ```train, valid and test``` directories in ```models/advoc/data/ljspeech/wavs_split```.

Train the the adversarial vocoder model on the training set as follows:

```
cd models/advoc
WORK_DIR=./train
python train.py train \
  ${WORK_DIR} \
  --data_dir ./data/ljspeech/wavs_split/train \
  --data_fastwav \
```

To train the smaller version of adversarial vocoder (AdVoc-small) add the argument use:

```
python train_evaluate.py train \
  ${WORK_DIR} \
  --data_dir ./data/ljspeech/wavs_split/train \
  --data_fastwav \
  --model_type small
```


### Inference

Generate mel-spectrograms for audio files from the test dataset using ```scripts/audio_to_spectrogram.py``` as follows:

```
cd scripts
python audio_to_spectrogram.py \
--wave_dir ../models/advoc/data/ljspeech/wavs_split/test \
--out_dir ../models/advoc/data/ljspeech/mel_specs/test \
--data_fast_wav
```

The above command should save the extracted mel-spectrogram features to ```models/advoc/data/ljspeech/mel_specs/test```

To vocode mel-spectrograms using a pretrained model, use ```scripts/spectrogram_advoc.py```:

```
cd scripts
python spectrogram_advoc.py \
--spec_dir ../models/advoc/data/ljspeech/mel_specs/test \
--out_dir ../models/advoc/data/ljspeech/vocoded_output/test \
--model_ckpt <PATH TO PRETRAINED CKPT>
--meta_fp <PATH TO MODEL METAGRAPH>
```

The above command should save the vocoded audio in ```models/advoc/data/ljspeech/vocoded_output/test```.


## Mel spectrogram GAN

Adversarial vocoding can be used to factorize unsupervised generation of audio into `P(spectrogram) * P(audio | spectrogram)`. This is useful because there are many GAN primitives for generating images, and spectrograms are (somewhat) image-like, so we can leverage pre-existing work. In our paper, we show that this factorized strategy can be used to achieve state-of-the-art results on unsupervised generation of small-vocabulary speech.

### Training a new model

To train a mel spectrogram GAN, first [download the SC09 dataset](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/sc09.tar.gz) and unzip to `models/melspecgan/data`. Then, run the following from this directory:

```
cd models/melspecgan
WORK_DIR=./train
python train.py train \
  ${WORK_DIR} \
  --data_dir ./data/sc09/train \
```

## Pretrained checkpoints

- [LJSpeech Advoc Mel80](https://drive.google.com/open?id=1fyYugd73xofb6jU2m4GoKbCOVBaYc-zH)
- [LJSpeech Advoc Mel40](https://drive.google.com/open?id=1YAqCHrlDThpL71uZqSKa4onohOHKfO8H)
- [LJSpeech Advoc Mel20](https://drive.google.com/open?id=1uLTtY4PH6BC-DAmBWS0WZAHy7YVTo-ZI)
- [LJSpeech Advoc (small) Mel80](https://drive.google.com/open?id=126qWSsW7W8ofowETA4bFUjqddUzU7fhb)
- [SC09 MelSpecGan Mel80](https://drive.google.com/open?id=12X7B6bup2ObFckYlZt_14GFLFYdQcX-a)
- [SC09 Advoc Mel80](https://drive.google.com/open?id=1oNBB-MSP28uHkqVOtYa6c3AfAQyEQZ0b)
