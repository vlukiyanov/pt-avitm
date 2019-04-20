# pt-avitm
[![Build Status](https://travis-ci.org/vlukiyanov/pt-avitm.svg?branch=master)](https://travis-ci.org/vlukiyanov/pt-avitm) [![codecov](https://codecov.io/gh/vlukiyanov/pt-avitm/branch/master/graph/badge.svg)](https://codecov.io/gh/vlukiyanov/pt-avitm)

PyTorch implementation of a version of the Autoencoding Variational Inference For Topic Models (AVITM) algorithm. Compatible with PyTorch 1.0.0 and Python 3.6 or 3.7 with or without CUDA.

This follows (*or attempts to; note this implementation is unofficial*) the algorithm described in "Autoencoding Variational Inference For Topic Models" of Akash Srivastava, Charles Sutton (https://arxiv.org/abs/1703.01488).

Currently this is work in progress, and is lacking more tests and further examples.

## Examples

You can find a number of examples in the examples directory.

## Usage

The simplest way to use the library is using the sklearn-compatible API, as below.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

from ptavitm.sklearn_api import ProdLDATransformer

pipeline = make_pipeline(
    CountVectorizer(
        stop_words='english',
        max_features=2500,
        max_df=0.9
    ),
    ProdLDATransformer()
)

pipeline.fit(texts)
result = pipeline.transform(texts)
```

## Other implementations of AVITM and similar

* Original TensorFlow: https://github.com/akashgit/autoencoding_vi_for_topic_models 
* PyTorch: https://github.com/hyqneuron/pytorch-avitm
