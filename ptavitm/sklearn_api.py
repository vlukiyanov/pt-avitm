from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Sparse2Corpus
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import torch
from torch.optim import Adam
from torch.utils.data.sampler import WeightedRandomSampler

from ptavitm.model import train, predict
from ptavitm.vae import ProdLDA
from ptavitm.utils import CountTensorDataset


# TODO decide how a partial_fit method API might work and implement, also more thought into score and logging


class ProdLDATransformer(TransformerMixin, BaseEstimator):
    def __init__(self,
                 cuda=None,
                 batch_size=200,
                 epochs=80,
                 hidden1_dimension=100,
                 hidden2_dimension=100,
                 topics=50,
                 lr=0.001,
                 samples=20000,
                 score_num=7,
                 score_type='coherence') -> None:
        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden1_dimension = hidden1_dimension
        self.hidden2_dimension = hidden2_dimension
        self.topics = topics
        self.lr = lr
        self.samples = samples
        self.autoencoder = None
        self.score_type = score_type
        self.score_num = score_num
        if self.score_type not in ['coherence']:
            raise ValueError('score_type must be "coherence"')

    def fit(self, X, y=None) -> None:
        documents, features = X.shape
        ds = CountTensorDataset(X.astype(np.float32))
        self.autoencoder = ProdLDA(
            in_dimension=features,
            hidden1_dimension=self.hidden1_dimension,
            hidden2_dimension=self.hidden2_dimension,
            topics=self.topics
        )
        if self.cuda:
            self.autoencoder.cuda()
        ae_optimizer = Adam(
            self.autoencoder.parameters(),
            lr=self.lr,
            betas=(0.99, 0.999)
        )
        train(
            ds,
            self.autoencoder,
            cuda=self.cuda,
            validation=None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            optimizer=ae_optimizer,
            sampler=WeightedRandomSampler(torch.ones(documents), min(documents, self.samples)),
            silent=True,
            num_workers=0  # TODO causes a bug to change this on Mac
        )

    def transform(self, X):
        if self.autoencoder is None:
            raise NotFittedError
        self.autoencoder.eval()
        ds = CountTensorDataset(X.astype(np.float32))
        output = predict(
            ds,
            self.autoencoder,
            cuda=self.cuda,
            encode=True,
            silent=True,
            batch_size=self.batch_size,
            num_workers=0  # TODO causes a bug to change this on Mac
        )
        return output.cpu().numpy()

    def score(self, X, y=None, sample_weight=None) -> float:
        # TODO this needs further testing for correctness, WIP
        if self.autoencoder is None:
            raise NotFittedError
        self.autoencoder.eval()
        corpus = Sparse2Corpus(X, documents_columns=False)
        decoder_weight = self.autoencoder.decoder.linear.weight.detach().cpu()
        id2word = {index: str(index) for index in range(X.shape[1])}
        topics = [
            [str(item.item()) for item in topic]
            for topic in decoder_weight.topk(min(self.score_num, X.shape[1]), dim=0)[1].t()
        ]
        cm = CoherenceModel(
            topics=topics,
            corpus=corpus,
            dictionary=Dictionary.from_corpus(corpus, id2word),
            coherence='u_mass'
        )
        return cm.get_coherence()
