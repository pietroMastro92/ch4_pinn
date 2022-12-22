"""
This is the demo code that uses hydra to access the parameters in under the directory config.
Author: Pietro Mastro
"""

import time
import hydra
import logging
import pandas as pd
import numpy as np
from joblib import load, dump
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from hydra.utils import to_absolute_path as abspath
from IasiNoiseTransformer import IasiNoiseTransformer
from IasiPerBandPcaTransformer import IasiPerBandPcaTransformer
from Costants import Costants

@hydra.main(config_path="../config", config_name="main")
def process_data(config: DictConfig):

    """Function to process the data"""
    raw_path = abspath(config.raw.path)
    iasi_noise_path = abspath(config.raw.path_iasi_noise)
    orth_base_path = abspath(config.raw.path_orth_base)
    
    logging.info(f"Process data using {raw_path}")

    # ? SHOULD WE LOAD/SAVE DATA USING A SPECIFIC FILE FORMAT
    # * FOR THE MOMENT WE ARE USING A PANDAS DATAFRAME SAVED WITH JOBLIB IN WICH THERE ARE ALL THE DATA NEEDED
    # * THE DATAFRAME HAS BEEN GENERATED USING THE 20% OF THE TOTAL DATASET OF SIMULATED MEASUREMENT FROM SIGMA-IASI
    logging.info("LOAD RAW DATASET")
    raw_dataset = load(raw_path)
    iasi_noise = load(iasi_noise_path)
    orth_base = load(orth_base_path)

    processing_columns = config.process.processing_columns 
    #*SPLIT INPUT/OUTPUT USING COLUMNS NAME
    logging.info("INPUT/OUTPUT DATA SPLITTING")
    X_train = raw_dataset.filter(
                                regex=f'({processing_columns.iasi_ch}|'
                                      f'{processing_columns.vza}|'
                                      f'{processing_columns.surface_pressure})'
                                )
    
    y_train = raw_dataset.filter(regex=f'{processing_columns.ch4}').to_numpy()

    #*PROCESSING OF INPUT DATA
    logging.info("INPUT PREPROCESSING START")
    ts = time.time()
    X_train, input_preprocessor = process_input(X_train, iasi_noise, 
                                                orth_base, 
                                                processing_columns.iasi_pcs_bnd1,
                                                processing_columns.iasi_pcs_bnd2)
    tf = time.time()
    logging.info("INPUT PREPROCESSING FINISH. TIME ELAPSED: %0.3fs" % tf)

    #*SAVE PROCESSED DATA
    processed_path_input = abspath(config.processed.path_input)
    save(X_train, processed_path_input)
    processed_path_output = abspath(config.processed.path_output)
    save(y_train, processed_path_output)


def process_input(X, iasi_noise_matrix, orth_base, iasi_pcs_bnd1, iasi_pcs_bnd2):
    costants = Costants()
    #*DEFINING A PROCESSING PIPELINE
    iasi_scaler_features_bnd1 = [costants.IASI_PC_BND1 + " " + str(i + 1) for i in range(iasi_pcs_bnd1)]
    iasi_scaler_features_bnd2 = [costants.IASI_PC_BND2 + " " + str(i + 1) for i in range(iasi_pcs_bnd2)]
    vza_scaler_features = [costants.VZA]
    surface_pressure_scaler_features = [costants.SURFACE_PRESSURE]

    scaler_preprocessor = ColumnTransformer(
                                            transformers=[('iasi1', StandardScaler(), iasi_scaler_features_bnd1),
                                                        ('iasi2', StandardScaler(), iasi_scaler_features_bnd2),
                                                        ('vza', StandardScaler(), vza_scaler_features),
                                                        ('surface_pressure', StandardScaler(),
                                                        surface_pressure_scaler_features)]
                                            )   
    

    preprocessing_pipeline = Pipeline(
                             steps=[('iasi_noise', IasiNoiseTransformer(iasi_noise_matrix)),
                                    ('iasi_pca', IasiPerBandPcaTransformer(orth_base[costants.EUMETSAT_BND1],
                                                                           orth_base[costants.EUMETSAT_BND2])),
                                    ('standard_scaling', scaler_preprocessor)
                                    ])
    X_processed = preprocessing_pipeline.fit_transform(X)
    return X_processed, preprocessing_pipeline

def save(data, path):
    start = time.time()
    with open(path, 'wb') as f:
        dump(data, f, compress='zlib')
        zlib_dump_duration = time.time() - start
    logging.info("Zlib dump duration: %0.3fs" % zlib_dump_duration)


if __name__ == "__main__":
    logging.getLogger(__name__)
    logging.basicConfig(filename='logging_output.log', level=logging.DEBUG)
    process_data()
