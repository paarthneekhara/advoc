# Adversarial vocoding (`advoc`)

Code from our paper *Expediting TTS Synthesis with Adversarial Vocoding*. Sound examples can be found [here](https://chrisdonahue.github.io/advoc_examples).

## Installation

There are multiple training scripts in this repository, but they all require the lightweight `advoc` package. To install globally, use `sudo pip install -e .` from this directory. To install in a virtual environment, follow these instructions:

```
virtualenv -p python3 --no-site-packages advoc
cd advoc
git clone git@github.com:chrisdonahue/advoc.git
source bin/activate
cd advoc
pip install -e .
```

## Mel spectrogram GAN (MelSpecGan)

Adversarial vocoding can be used to factorize unsupervised generation of audio into `P(spectrogram) * P(audio | spectrogram)`. This is useful because there are many GAN primitives for generating images, and spectrograms are (somewhat) image-like. In our paper, we show that this factorized strategy can be used to achieve state-of-the-art results on unsupervised generation of small-vocabulary speech.

To train a mel spectrogram GAN, first [download the SC09 dataset](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/sc09.tar.gz) and unzip to `melspecgan/data`. Then, run the following from this directory:

```
cd melspecgan
WORK_DIR=./train
python train.py train \
  ${WORK_DIR} \
  --data_dir ./data/sc09/train \
```

## Testing

`python setup.py test`
