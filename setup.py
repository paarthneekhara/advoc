#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
  long_description = f.read()

setup(
    name='advoc',
    version='0.1.0',
    description='Vocode with generative adversarial networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chrisdonahue/advoc',
    packages=find_packages(),
    install_requires=[
      'numpy>=1.16.0',
      'tensorflow-gpu==1.13.1',
      'librosa==0.6.3',
      'lws==1.2',
      'tqdm>=4.31.1',
      'scipy>=1.0.0',
    ],
    test_suite='tests'
)
