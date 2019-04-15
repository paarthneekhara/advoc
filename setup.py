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
      'tensorflow-gpu>=1.12.0',
      'librosa==0.6.3',
      'lws==1.2',
      'numpy>=1.16.0',
      'scipy>=1.0.0',
      'tqdm>=4.31.1'
    ],
    test_suite='tests'
)
