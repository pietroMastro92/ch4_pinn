import numpy as np
class IasiNoiseOperator():

    def __init__(self, iasi_cov_matrix):
        # iasi_cov_matrix mxm iasi covariance matrix.
        self.iasi_cov_matrix = np.asfortranarray(iasi_cov_matrix).astype(np.float32)

    def addNoise(self, iasi_spectra):
        # iasi_spectra  nxm matrix, n = number of measurements, m = number of IASI channels.
        rowShape = iasi_spectra.shape[0]
        columnShape = iasi_spectra.shape[1]
        noise = self.computeNoise(rowShape, columnShape, self.iasi_cov_matrix)
        return iasi_spectra + noise

    def computeNoise(self, row_shape, column_shape, iasi_cov_matrix):
        # COMPUTE THE IASI GAUSSIAN NOISE MATRIX
        mu, sigma = 0., 1.  # mean and standard deviation
        random_matrix = \
            np.asfortranarray(np.random.normal(mu, sigma, size=(row_shape, column_shape))).astype(np.float32)
        return random_matrix.dot(iasi_cov_matrix)