import numpy as np
from scipy.sparse import csr_matrix

from ptavitm.utils import CountTensorDataset


def test_shape():
    test_matrix = csr_matrix(np.eye(10))
    dataset = CountTensorDataset(test_matrix)
    assert len(dataset) == 10
    for item in range(10):
        assert isinstance(dataset[item], tuple)
        assert len(dataset[item]) == 1
        assert len(dataset[item][0].shape) == 1
        assert dataset[item][0].shape[0] == 10
