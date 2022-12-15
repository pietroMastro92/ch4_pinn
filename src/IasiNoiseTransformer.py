import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from IasiNoiseOperator import IasiNoiseOperator

class IasiNoiseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, iasi_noise_matrix, iasi_feature_name ='IASI Channel '):
        #SAVE ONLY THE TRIANGULAR LOWER PART
        self.iasi_noise_matrix = np.asfortranarray(np.tril(iasi_noise_matrix)).astype(np.float32)
        self.iasi_feature_name = iasi_feature_name

    def fit(self, X):
        return self

    def transform(self, X):
        #RECONSTRUCT NOISE FROM TRIANGULAR LOWER PART
        noise_matrix = self.iasi_noise_matrix + self.iasi_noise_matrix.T - np.diag(self.iasi_noise_matrix.diagonal())
        iasiNoiseOperator = IasiNoiseOperator(noise_matrix)
        X_ = X.copy()
        filter_constrain = X_.columns[[bool(re.compile(self.iasi_feature_name.rstrip()).search(x)) for x in X_.columns.values]]
        #ADD NOISE
        X_[filter_constrain] = iasiNoiseOperator.addNoise(X_[filter_constrain])
        return X_