import numpy as np
import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.base import BaseEstimator, TransformerMixin


class IasiPerBandPcaTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, orth_base_bnd1, orth_base_bnd2,
                 pc_bnd1=90, pc_bnd2=120, #MORE PCS IN CH4 ABSORPTION BAND (ARE THE DEFAULT EUMETSAT VALUES)
                 iasi_feature_name='IASI Channel '):

        self.pc_bnd1 = pc_bnd1
        self.pc_bnd2 = pc_bnd2
        self.orth_base_bnd1 = orth_base_bnd1
        self.orth_base_bnd2 = orth_base_bnd2

        self.iasi_feature_name = iasi_feature_name


    def fit(self, X):

        return self


    def transform(self, X):

        X_ = X.copy()

        # RECONSTRUCT NOISE FROM TRIANGULAR LOWER PART

        iasi_channels = X_.filter(regex=self.iasi_feature_name).astype(np.float32)

        columns_name = iasi_channels.columns.to_list()

        iasi_channels = iasi_channels.to_numpy()


        # PER BAND1 and BAND2 COMPUTATION

        n1_bnd1 = self.orth_base_bnd1.k1 - 1

        n2_bnd1 = n1_bnd1 + self.orth_base_bnd1.kn

        iasi_channels_bnd1 = iasi_channels[:, n1_bnd1:n2_bnd1]

        pcs_bnd1 = np.dot(self.orth_base_bnd1.E[:, :self.pc_bnd1].T,

                          np.dot(np.linalg.inv(self.orth_base_bnd1.N), iasi_channels_bnd1.T)).T


        n1_bnd2 = self.orth_base_bnd2.k1 - 1

        n2_bnd2 = n1_bnd2 + self.orth_base_bnd2.kn

        iasi_channels_bnd2 = iasi_channels[:, n1_bnd2:n2_bnd2]

        pcs_bnd2 = np.dot(self.orth_base_bnd2.E[:, :self.pc_bnd2].T,

                          np.dot(np.linalg.inv(self.orth_base_bnd2.N)  , iasi_channels_bnd2.T)).T


        # PCA

        X_ = X_.drop(columns_name, axis=1)  # DROP COLUMNS RELATED TO IASI CHANNELS

        iasi_pcs_bnd1 = pd.DataFrame(data=pcs_bnd1[0:, 0:],

                                     index=X_.index,  # use same indices as before

                                     columns=["B1 IASI PC " + str(i + 1) for i in

                                              range(np.shape(pcs_bnd1)[1])])

        iasi_pcs_bnd2 = pd.DataFrame(data=pcs_bnd2[0:, 0:],

                                     index=X_.index,  # use same indices as before

                                     columns=["B2 IASI PC " + str(i + 1) for i in

                                              range(np.shape(pcs_bnd2)[1])])

        iasi_pcs = pd.concat([iasi_pcs_bnd1, iasi_pcs_bnd2], axis=1)


        X_ = pd.concat([X_, iasi_pcs], axis=1)

        return X_

