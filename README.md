# Adversarial vocoding (`advoc`)

## Installation

```
virtualenv -p python3 --no-site-packages advoc
cd advoc
git clone git@github.com:chrisdonahue/advoc.git
source bin/activate
cd advoc
pip install -e .
```

If you want to support more types of audio decoding (including MP3s), use the following command which will install `librosa`:
`pip install -e .[audio_decoding]`

## Testing

`python setup.py test`
